import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.checkpoint import filter_head_free_state_dict, load_head_free_checkpoint
from parta.provenance import ResolvedRunContract, checkpoint_shard_digest, write_run_status
from parta.state_head import (
    ObjectStatePredictions,
    SetSlotStateHead,
    StateHeadConfig,
    extract_visual_prefix_hidden,
)
from parta.state_loss import ObjectStateSetLoss, StateLossConfig, StateTargets
from parta.training import run_a1o_side_branch
from parta.t0 import (
    REQUIRED_T0_CHECKS,
    GradientBatchRecord,
    T0Report,
    assert_component_shared_gradient_norms,
    assert_exact_frame_contract,
    compare_tensors,
    summarize_gradient_calibration,
)


def _predictions(batch=1, slots=6, categories=4, frames=4):
    generator = torch.Generator().manual_seed(42)
    tensors = [
        torch.randn(batch, slots, generator=generator, requires_grad=True),
        torch.randn(batch, slots, categories, generator=generator, requires_grad=True),
        torch.randn(batch, slots, 3, generator=generator, requires_grad=True),
        torch.randn(batch, slots, 3, generator=generator, requires_grad=True),
        torch.randn(batch, slots, frames, generator=generator, requires_grad=True),
        torch.randn(batch, slots, 8, generator=generator, requires_grad=True),
    ]
    return ObjectStatePredictions(*tensors)


def _target(order=None, empty=False, source="adt"):
    if empty:
        count = 0
        order = torch.empty(0, dtype=torch.long)
    else:
        count = 3
        order = torch.arange(count) if order is None else torch.as_tensor(order)
    categories = torch.tensor([1, 2, 0], dtype=torch.long)[:count][order]
    centers = torch.tensor([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])[:count][order]
    extents = torch.tensor([[1.0, 1, 1], [2.0, 1, 1], [1.0, 3, 1]])[:count][order]
    visibility = torch.tensor(
        [[1.0, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]]
    )[:count][order]
    return StateTargets(
        categories=categories,
        centers_world_m=centers,
        extents_m=extents,
        visibility=visibility,
        category_valid=torch.ones(count, dtype=torch.bool),
        center_valid=torch.ones(count, dtype=torch.bool),
        extent_valid=torch.ones(count, dtype=torch.bool),
        visibility_valid=torch.ones(count, 4, dtype=torch.bool),
        scene_scale_m=torch.tensor(10.0),
        source_dataset=source,
        scene_id=f"{source}-scene",
    )


def test_hungarian_and_loss_are_gt_permutation_invariant():
    predictions = _predictions()
    criterion = ObjectStateSetLoss(StateLossConfig(smooth_l1_beta=0.2))
    original = criterion(predictions, [_target()])
    permuted = criterion(predictions, [_target(order=[2, 0, 1])])
    for name in ("loss_state", "loss_existence", "loss_category", "loss_center", "loss_extent", "loss_visibility"):
        torch.testing.assert_close(original[name], permuted[name])


def test_empty_gt_only_has_all_negative_existence_loss():
    predictions = _predictions()
    result = ObjectStateSetLoss(StateLossConfig())(predictions, [_target(empty=True)])
    assert result["loss_existence"].item() > 0
    for name in ("loss_category", "loss_center", "loss_extent", "loss_visibility"):
        assert result[name].item() == 0
    result["loss_state"].backward()
    assert predictions.existence_logits.grad is not None
    assert predictions.category_logits.grad is None


def test_all_masked_optional_fields_are_finite_and_backward():
    predictions = _predictions()
    target = _target()
    target.category_valid.zero_()
    target.center_valid.zero_()
    target.extent_valid.zero_()
    target.visibility_valid.zero_()
    result = ObjectStateSetLoss(StateLossConfig())(predictions, [target])
    assert torch.isfinite(result["loss_state"])
    assert result["loss_category"].item() == 0
    result["loss_state"].backward()
    assert predictions.existence_logits.grad is not None


def test_visual_tap_contract_and_set_head_backward():
    hidden = torch.randn(2, 80, 16, requires_grad=True)
    visual_mask = torch.zeros(2, 80, dtype=torch.bool)
    visual_mask[:, :32] = True
    counts = [[2] * 16, [2] * 16]
    ids = [list(range(16)), list(range(100, 116))]
    tap = extract_visual_prefix_hidden(hidden, visual_mask, counts, ids)
    assert tap.hidden.shape == (2, 32, 16)
    assert tap.frame_token_spans[0, -1].tolist() == [30, 32]
    head = SetSlotStateHead(
        StateHeadConfig(
            hidden_size=16,
            num_categories=4,
            num_slots=384,
            num_layers=1,
            num_heads=4,
            ffn_dim=32,
            max_frames=32,
        )
    )
    output = head(tap)
    assert output.existence_logits.shape == (2, 384)
    output.existence_logits.mean().backward()
    assert hidden.grad is not None


def test_head_free_filter_and_load_audit():
    model = torch.nn.Linear(3, 2)
    state = copy.deepcopy(model.state_dict())
    state["parta_state_head.slot_queries"] = torch.randn(4, 3)
    filtered, dropped = filter_head_free_state_dict(state)
    assert "parta_state_head.slot_queries" not in filtered
    assert dropped == ("parta_state_head.slot_queries",)
    audit = load_head_free_checkpoint(torch.nn.Linear(3, 2), state)
    assert audit.passed
    assert audit.loaded_shared_keys == 2
    assert not load_head_free_checkpoint(torch.nn.Linear(3, 2), model.state_dict()).passed
    assert not load_head_free_checkpoint(
        torch.nn.Linear(3, 2),
        state,
        expected_state_head_keys=("parta_state_head.wrong",),
    ).passed


def test_a1o_runtime_rejects_mope_even_after_attach():
    model = torch.nn.Module()
    model.parta_state_head = torch.nn.Identity()
    model._mope_encoder = object()
    with pytest.raises(ValueError, match="MoPE-free"):
        run_a1o_side_branch(
            model,
            visual_state_hidden=torch.empty(0),
            visual_state_valid_mask=torch.empty(0),
            frame_token_counts=[],
            frame_ids=[],
            targets=[],
            loss_config=StateLossConfig(),
        )


def test_t0_exact_contract_comparison_gradient_summary_and_report(tmp_path):
    ids = torch.tensor([[1, 2]])
    masks = torch.tensor([[True, True]])
    spans = torch.tensor([[[0, 2], [2, 4]]])
    assert_exact_frame_contract(ids, ids.clone(), masks, masks.clone(), spans, spans.clone())
    with pytest.raises(AssertionError):
        assert_exact_frame_contract(ids, ids + 1, masks, masks, spans, spans)

    comparison = compare_tensors(torch.ones(3), torch.ones(3) + 1e-7)
    assert comparison.passed
    pattern = [(1, 100), (2, 101), (100, 100), (101, 200), (102, 300)]
    records = []
    for index in range(50):
        qa, state = pattern[index % len(pattern)]
        records.append(
            GradientBatchRecord(
                source_dataset="adt" if index % 2 == 0 else "hypersim",
                qa_gradient_norm=qa,
                state_gradient_norm=state,
            )
        )
    summary = summarize_gradient_calibration(records)
    assert summary["passed"]
    assert summary["statistics"]["qa_over_state"]["median"] == pytest.approx(0.34)
    assert summary["lambda_state_candidate"] == pytest.approx(0.034)

    report_path = tmp_path / "t0.json"
    report = T0Report("synthetic", "abc")
    for name in REQUIRED_T0_CHECKS:
        report.add_boolean(name, True)
    report.add_comparison("question_invariance", comparison)
    report.add_boolean(
        "gradient_calibration",
        summary["passed"],
        **{key: value for key, value in summary.items() if key != "passed"},
    )
    payload = report.finalize(str(report_path))
    assert payload["status"] == "complete_passed"
    assert json.loads(report_path.read_text())["status"] == "complete_passed"


def test_t0_missing_check_and_component_gradient_are_hard_failures(tmp_path):
    report = T0Report("incomplete", "abc")
    report.add_boolean("finite", True)
    with pytest.raises(AssertionError, match="missing"):
        report.finalize(str(tmp_path / "failed.json"))
    payload = json.loads((tmp_path / "failed.json").read_text())
    assert payload["status"] == "complete_failed"
    with pytest.raises(AssertionError, match="component"):
        assert_component_shared_gradient_norms(
            {"existence": 1.0, "center": 0.0},
            ("existence", "center", "extent"),
        )


def test_d55_run_contract_checkpoint_and_status(tmp_path):
    shard_a = tmp_path / "model-1.bin"
    shard_b = tmp_path / "model-2.bin"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    checkpoint_digest = checkpoint_shard_digest([shard_b, shard_a])
    digest = "a" * 64
    base = dict(
        run_id="a1o-seed42",
        experiment="A1-O",
        seed=42,
        resolved_config={"num_slots": 384, "video_min_frames": 16, "video_max_frames": 32},
        manifest_sha256=digest,
        initialization_sha256=digest,
        code_revision="deadbeef",
        exact_frame_binding_sha256=digest,
        output_dir=str(tmp_path),
    )
    running = ResolvedRunContract(**base)
    status_path = tmp_path / "provenance.json"
    write_run_status(running, status_path)
    payload = json.loads(status_path.read_text())
    assert payload["status"] == "running"
    assert len(payload["resolved_config_sha256"]) == 64
    assert len(payload["run_fingerprint"]) == 64
    with pytest.raises(ValueError, match="checkpoint"):
        ResolvedRunContract(**base, status="complete").to_payload()
    complete = ResolvedRunContract(
        **base, status="complete", checkpoint_sha256=checkpoint_digest
    )
    write_run_status(complete, status_path)
    assert json.loads(status_path.read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="terminal"):
        write_run_status(complete, status_path)
