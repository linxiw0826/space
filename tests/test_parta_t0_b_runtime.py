import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.t0_b_runtime import (
    T0BBatchObservation,
    T0BThresholds,
    build_t0_b_report,
    finalize_t0_b_report,
    validate_t0_a_initialization_transaction,
    parameter_gradient_norm,
    nested_state_digest,
)
from parta.checkpoint import (
    ResumeContract, capture_rng_state, load_training_checkpoint, save_training_checkpoint,
)
from parta.provenance import stable_sha256


def _observations(count=30, *, bad_index=None):
    sources = ("adt", "hypersim", "scannetppv2")
    rows = []
    for index in range(count):
        value = 0.0 if index == bad_index else 1.0
        rows.append(T0BBatchObservation(
            batch_index=index, qa_id=f"q{index}", source_dataset=sources[index % 3],
            qa_loss=1.0, state_loss=2.0, qa_gradient_norm=value,
            state_gradient_norm=1.0, shared_gradient_parameter_count=1,
            head_gradient_parameter_count=1, enabled_components=("existence", "category"),
            masked_components=("center", "extent", "visibility"),
            component_losses={"existence": 1.0, "category": 1.0}, matching_valid=True,
            component_valid_counts={"existence": 1, "category": 1, "center": 0, "extent": 0, "visibility": 0},
            matched_pairs=1, gt_objects=1, exact_frame_consistent=True,
            actual_frame_count=16,
        ))
    return rows


def test_gpu_report_is_machine_decidable_and_source_balanced(tmp_path):
    report = build_t0_b_report(
        _observations(), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert report["status"] == "complete_passed"
    assert report["source_batch_counts"] == {"adt": 10, "hypersim": 10, "scannetppv2": 10}
    assert report["gradient_calibration"]["lambda_state_candidate"] == pytest.approx(0.1)
    path = tmp_path / "report.json"
    finalize_t0_b_report(report, str(path))
    assert json.loads(path.read_text())["formal_gpu_evidence"] is True


def test_cpu_mock_never_becomes_formal_pass():
    report = build_t0_b_report(
        _observations(), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="awaiting_gpu",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert report["status"] == "awaiting_gpu"
    assert report["formal_gpu_evidence"] is False


def test_failed_gpu_gate_is_nonzero_at_finalize(tmp_path):
    report = build_t0_b_report(
        _observations(bad_index=0), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    with pytest.raises(AssertionError, match="shared_qa_gradients"):
        finalize_t0_b_report(report, str(tmp_path / "failed.json"))


def test_t0_b_rejects_unbound_t0_a_initialization(tmp_path):
    checkpoint = tmp_path / "t0.pt"
    torch.save({"model": {}}, checkpoint)
    checkpoint_sha = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema_version": "parta_t0_report_v1",
                                  "status": "complete_passed"}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "complete", "experiment": "parta-t0-a",
                                  "code_revision": "abc", "checkpoint_sha256": "s" * 64}))
    provenance = tmp_path / "provenance.json"
    payload = {
        "status": "complete_passed", "a1_checkpoint_role": "initialization_no_optimizer_updates",
        "a1_checkpoint_optimizer_steps": 0,
        "a1_checkpoint_artifact": {"ordered_shards": [{"sha256": checkpoint_sha}]},
        "a1_checkpoint_state_sha256": "s" * 64,
        "parameter_sha256_before_backward": "s" * 64,
        "parameter_sha256_after_backward": "s" * 64,
        "git_revision": "abc", "checkpoint_sha256": "g" * 64,
        "vggt_checkpoint_sha256": "v" * 64,
        "manifest_sha256": {"adt": "a" * 64, "hypersim": "h" * 64},
        "exact_frame_binding_sha256": "e" * 64,
    }
    provenance.write_text(json.dumps(payload))
    kwargs = dict(
        report_path=report, provenance_path=provenance, run_status_path=status,
        checkpoint_path=checkpoint, current_code_revision="abc",
        guide_artifact_sha256="g" * 64, vggt_artifact_sha256="v" * 64,
        current_manifest_inputs={
            "adt": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "a" * 64}}},
            "hypersim": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "h" * 64}}},
            "scannetppv2": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "p" * 64}}},
        },
    )
    assert validate_t0_a_initialization_transaction(**kwargs)[
        "t0_a_checkpoint_optimizer_steps"
    ] == 0
    payload["a1_checkpoint_optimizer_steps"] = 1
    provenance.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="optimizer_steps"):
        validate_t0_a_initialization_transaction(**kwargs)
    payload["a1_checkpoint_optimizer_steps"] = 0
    provenance.write_text(json.dumps(payload))
    kwargs["current_manifest_inputs"]["adt"]["files"][
        "qa_manifest_exact_verified.jsonl"
    ]["sha256"] = "x" * 64
    with pytest.raises(ValueError, match="manifest.adt"):
        validate_t0_a_initialization_transaction(**kwargs)


def test_parameter_gradient_norm_counts_only_connected_parameters():
    connected = torch.nn.Parameter(torch.tensor(2.0))
    unused = torch.nn.Parameter(torch.tensor(3.0))
    norm, count = parameter_gradient_norm(connected.square(), (connected, unused), retain_graph=False)
    assert norm == pytest.approx(4.0)
    assert count == 1


def test_cli_cpu_mock_writes_awaiting_gpu_atomic_artifacts(tmp_path):
    output = tmp_path / "t0b"
    script = Path(__file__).resolve().parents[1] / "scripts" / "parta" / "run_t0_b.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output), "--cpu-mock", "--batches", "30"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "t0_b_report.json").read_text())["status"] == "awaiting_gpu"
    assert json.loads((output / "run_status.json").read_text())["status"] == "awaiting_gpu"
    resolved = json.loads((output / "resolved_config.json").read_text())
    digest = stable_sha256(resolved)
    assert json.loads((output / "t0_b_report.json").read_text())["resolved_config_sha256"] == digest
    assert json.loads((output / "provenance.json").read_text())["resolved_config_sha256"] == digest
    assert json.loads((output / "run_status.json").read_text())["resolved_config_sha256"] == digest


def test_thresholds_are_cli_overridable_but_bounded():
    T0BThresholds(minimum_batches=20, maximum_batches=50).validate(20)
    with pytest.raises(ValueError, match="requested T0-B batches"):
        T0BThresholds().validate(19)


def test_missing_expected_source_is_a_hard_failure():
    rows = [row for row in _observations() if row.source_dataset != "scannetppv2"]
    report = build_t0_b_report(
        rows, requested_batches=len(rows),
        thresholds=T0BThresholds(minimum_batches=20, maximum_batches=50),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert not report["checks"]["source_registry_exact"]["passed"]
    assert report["status"] == "complete_failed"


def test_component_mask_loss_inconsistency_is_a_hard_failure():
    rows = _observations()
    bad = rows[0]
    rows[0] = T0BBatchObservation(
        **(vars(bad) | {
            "enabled_components": ("existence",),
            "masked_components": ("category", "center", "extent", "visibility"),
            "component_losses": {"existence": 1.0, "category": 1.0},
        })
    )
    report = build_t0_b_report(
        rows, requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert not report["checks"]["component_mask_consistency"]["passed"]


def test_checkpoint_resume_restores_tensor_optimizer_scheduler_counter_and_rng(tmp_path):
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    contract = ResumeContract("a1o", "1" * 64, "2" * 64, "3" * 64)
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        global_step=7, epoch=2, sampler_position=11, contract=contract,
    )
    expected = (
        nested_state_digest(model.state_dict()),
        nested_state_digest(optimizer.state_dict()),
        nested_state_digest(scheduler.state_dict()),
        nested_state_digest(capture_rng_state()),
    )
    with torch.no_grad():
        model.weight.add_(99)
    torch.rand(3)
    counters = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        expected_contract=contract,
    )
    actual = (
        nested_state_digest(model.state_dict()),
        nested_state_digest(optimizer.state_dict()),
        nested_state_digest(scheduler.state_dict()),
        nested_state_digest(capture_rng_state()),
    )
    assert actual == expected
    assert counters == {"global_step": 7, "epoch": 2, "sampler_position": 11}
