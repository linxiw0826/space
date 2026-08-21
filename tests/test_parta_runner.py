import copy
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers.utils import ModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.checkpoint import (
    ResumeContract,
    export_head_free_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    load_head_free_artifact,
    fsdp_optimizer_state_to_load,
)
from parta.distributed import DistributedContext
from parta.runner import (
    PartATrainBatch,
    PartATrainConfig,
    PartATrainer,
    SharedForwardOutput,
    assert_matched_fairness,
    config_sha256,
    matched_fairness_payload,
    seed_matched_run,
    attach_a1o_head_without_advancing_shared_rng,
    SourceBalancedCursor,
    validate_single_step_execution_contract,
)
from parta.state_head import StateHeadConfig
from parta.state_loss import StateLossConfig, StateTargets
from parta.training import (
    attach_a1o_state_head, consume_a1o_forward_result,
    install_a1o_forward_integration, prepare_a1o_forward_request,
)
from parta.training_log import JsonlTrainingLogger


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(8, 8)
        self.qa = torch.nn.Linear(8, 1)

    def forward(self, features, return_tap=False):
        hidden = self.shared(features)
        return SharedForwardOutput(
            qa_loss=self.qa(hidden).square().mean(),
            visual_state_hidden=hidden if return_tap else None,
            visual_state_valid_mask=torch.ones(hidden.shape[:2], dtype=torch.bool)
            if return_tap
            else None,
        )


@dataclass
class TinyHFOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    visual_state_hidden: torch.Tensor | None = None
    visual_state_valid_mask: torch.Tensor | None = None


class TinyHFModel(TinyModel):
    def forward(self, features, return_tap=False):
        hidden = self.shared(features)
        logits = self.qa(hidden)
        return TinyHFOutput(
            loss=logits.square().mean(),
            logits=logits,
            visual_state_hidden=hidden if return_tap else None,
            visual_state_valid_mask=torch.ones(hidden.shape[:2], dtype=torch.bool)
            if return_tap else None,
        )


def _forward(model, inputs, return_tap):
    return model(inputs["features"], return_tap=return_tap)


def _target():
    return StateTargets(
        categories=torch.tensor([1]),
        centers_world_m=torch.tensor([[1.0, 2.0, 3.0]]),
        extents_m=torch.tensor([[1.0, 1.0, 1.0]]),
        visibility=torch.ones(1, 16),
        category_valid=torch.ones(1, dtype=torch.bool),
        center_valid=torch.ones(1, dtype=torch.bool),
        extent_valid=torch.ones(1, dtype=torch.bool),
        visibility_valid=torch.ones(1, 16, dtype=torch.bool),
        scene_scale_m=torch.tensor(10.0),
        source_dataset="adt",
        scene_id="scene",
    )


def _batch():
    return PartATrainBatch(
        model_inputs={"features": torch.randn(1, 16, 8)},
        targets=[_target()],
        source_datasets=["adt"],
        frame_ids=[list(range(16))],
        frame_token_counts=[[1] * 16],
        media_kinds=["video"],
        expected_frame_binding_sha256=["a" * 64],
    )


def _trainer(tmp_path, arm):
    torch.manual_seed(11)
    model = TinyModel()
    if arm == "a1o":
        attach_a1o_state_head(
            model,
            StateHeadConfig(
                hidden_size=8,
                num_categories=4,
                num_slots=384,
                num_layers=1,
                num_heads=2,
                ffn_dim=16,
            ),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = PartATrainConfig(arm=arm, learning_rate=1e-3, max_steps=2)
    common = matched_fairness_payload(
        config,
        manifest_sha256="1" * 64,
        initialization_sha256="2" * 64,
        exact_frame_binding_sha256="3" * 64,
        trainable_shared_parameter_names=["qa.bias", "qa.weight", "shared.bias", "shared.weight"],
    )
    trainer = PartATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        forward_adapter=_forward,
        logger=JsonlTrainingLogger(tmp_path / f"{arm}.jsonl"),
        manifest_sha256="1" * 64,
        resolved_config_sha256=config_sha256(config),
        matched_contract_sha256=assert_matched_fairness(common, common),
    )
    return trainer


def test_a0_and_a1o_paths_and_log_schema(tmp_path):
    a0 = _trainer(tmp_path, "a0")
    assert not hasattr(a0.model, "parta_state_head")
    a0_record = a0.train_step(_batch())
    assert a0_record.state_loss == 0.0
    assert a0_record.peak_cuda_memory_bytes is None

    a1o = _trainer(tmp_path, "a1o")
    a1o_record = a1o.train_step(_batch())
    assert a1o_record.state_loss > 0
    payload = json.loads((tmp_path / "a1o.jsonl").read_text().splitlines()[0])
    required = {
        "qa_loss", "state_loss", "total_loss", "loss_existence", "loss_category",
        "loss_center", "loss_extent", "loss_visibility", "grad_norm",
        "effective_gt_objects", "source_dataset", "actual_frame_count",
        "samples_per_second", "frames_per_second", "peak_cuda_memory_bytes",
    }
    assert required <= payload.keys()


def test_a1o_side_branch_executes_inside_parent_forward():
    model = TinyModel()
    attach_a1o_state_head(model, StateHeadConfig(
        hidden_size=8, num_categories=4, num_slots=384, num_layers=1,
        num_heads=2, ffn_dim=16,
    ))
    install_a1o_forward_integration(model)
    prepare_a1o_forward_request(
        model, frame_token_counts=[[1] * 16], frame_ids=[list(range(16))],
        media_kinds=["video"], targets=[_target()], loss_config=__import__(
            "parta.state_loss", fromlist=["StateLossConfig"]
        ).StateLossConfig(),
    )
    output = model(torch.randn(1, 16, 8), return_tap=True)
    branch = consume_a1o_forward_result(model)
    assert branch.predictions.existence_logits.shape == (1, 384)
    assert torch.isfinite(branch.losses["loss_state"])
    # The zero-valued graph anchor makes the head reachable from the actual
    # forward return without changing the QA scalar.
    assert output.qa_loss.grad_fn is not None


def test_a1o_model_output_exposes_exact_non_detached_state_loss_mapping():
    model = TinyHFModel()
    attach_a1o_state_head(model, StateHeadConfig(
        hidden_size=8, num_categories=4, num_slots=384, num_layers=1,
        num_heads=2, ffn_dim=16,
    ))
    install_a1o_forward_integration(model)
    prepare_a1o_forward_request(
        model, frame_token_counts=[[1] * 16], frame_ids=[list(range(16))],
        media_kinds=["video"], targets=[_target()], loss_config=StateLossConfig(),
    )
    output = model(torch.randn(1, 16, 8), return_tap=True)
    branch = consume_a1o_forward_result(model)
    assert output["loss"] is output.loss
    assert output["parta_state_loss"] is branch.losses["loss_state"]
    assert output["parta_state_loss"].grad_fn is not None


def test_a1o_ddp_find_unused_supports_multiple_forward_backward_steps():
    """Regression for DDP's head parameter "marked ready twice" failure."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the default process group")

    with tempfile.NamedTemporaryFile() as rendezvous:
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{rendezvous.name}", rank=0, world_size=1
        )
        try:
            model = TinyModel()
            attach_a1o_state_head(model, StateHeadConfig(
                hidden_size=8, num_categories=4, num_slots=384, num_layers=1,
                num_heads=2, ffn_dim=16,
            ))
            install_a1o_forward_integration(model)
            ddp = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True
            )
            optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-4)

            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                target = _target()
                # Exercise a changing autograd graph: the category and
                # visibility components are entirely masked on the middle
                # iteration and active on the surrounding iterations.
                if step == 1:
                    target.category_valid.zero_()
                    target.visibility_valid.zero_()
                prepare_a1o_forward_request(
                    ddp,
                    frame_token_counts=[[1] * 16],
                    frame_ids=[list(range(16))],
                    media_kinds=["video"],
                    targets=[target],
                    loss_config=StateLossConfig(),
                )
                output = ddp(torch.randn(1, 16, 8), return_tap=True)
                branch = consume_a1o_forward_result(ddp)
                total = output.qa_loss + 0.02 * branch.losses["loss_state"]
                total.backward()
                if step == 1:
                    assert model.parta_state_head.visibility.bias.grad is None
                    assert model.parta_state_head.category.bias.grad is None
                else:
                    assert model.parta_state_head.visibility.bias.grad is not None
                    assert model.parta_state_head.category.bias.grad is not None
                optimizer.step()
        finally:
            torch.distributed.destroy_process_group()


def test_matched_identity_rejects_non_whitelisted_drift():
    left = matched_fairness_payload(
        PartATrainConfig(arm="a0"), manifest_sha256="1" * 64,
        initialization_sha256="2" * 64, exact_frame_binding_sha256="3" * 64,
        trainable_shared_parameter_names=["x"],
    )
    right = matched_fairness_payload(
        PartATrainConfig(arm="a1o"), manifest_sha256="1" * 64,
        initialization_sha256="2" * 64, exact_frame_binding_sha256="3" * 64,
        trainable_shared_parameter_names=["x"],
    )
    assert assert_matched_fairness(left, right)
    right["seed"] = 7
    with pytest.raises(ValueError, match="seed"):
        assert_matched_fairness(left, right)


def test_matched_execution_identity_rejects_world_size_and_strategy_drift():
    base = dict(distributed_strategy="ddp", ddp_find_unused_parameters=True,
                world_size=2, per_rank_batch_size=1,
                effective_global_batch_size=2, source_content_identity={"adt": "abc"})
    kwargs = dict(manifest_sha256="1" * 64, initialization_sha256="2" * 64,
                  exact_frame_binding_sha256="3" * 64,
                  trainable_shared_parameter_names=["x"])
    a0 = matched_fairness_payload(PartATrainConfig(arm="a0"), **kwargs,
                                  execution_contract=base)
    a1o = matched_fairness_payload(PartATrainConfig(arm="a1o"), **kwargs,
                                   execution_contract=base)
    assert assert_matched_fairness(a0, a1o)
    assert a0["execution_contract"]["ddp_find_unused_parameters"] is True
    drift = copy.deepcopy(a1o)
    drift["execution_contract"]["world_size"] = 1
    drift["execution_contract"]["distributed_strategy"] = "none"
    with pytest.raises(ValueError, match="execution_contract"):
        assert_matched_fairness(a0, drift)


def test_gradient_accumulation_is_fail_closed_across_cli_config_and_execution():
    with pytest.raises(ValueError, match="gradient accumulation >1"):
        PartATrainConfig(arm="a0", gradient_accumulation_steps=2).validate()
    config = PartATrainConfig(arm="a0")
    valid = {"gradient_accumulation_steps": 1, "effective_global_batch_size": 2}
    validate_single_step_execution_contract(
        cli_gradient_accumulation_steps=1, config=config,
        execution_contract=valid, world_size=2,
    )
    for cli, execution in ((2, valid), (1, {**valid, "gradient_accumulation_steps": 2}),
                           (1, {**valid, "effective_global_batch_size": 4})):
        with pytest.raises(ValueError):
            validate_single_step_execution_contract(
                cli_gradient_accumulation_steps=cli, config=config,
                execution_contract=execution, world_size=2,
            )
    matched = matched_fairness_payload(
        config, manifest_sha256="1" * 64, initialization_sha256="2" * 64,
        exact_frame_binding_sha256="3" * 64,
        trainable_shared_parameter_names=["x"], execution_contract=valid,
    )
    assert matched["gradient_accumulation_steps"] == 1
    assert matched["execution_contract"]["gradient_accumulation_steps"] == 1
    assert matched["execution_contract"]["effective_global_batch_size"] == 2


def test_fsdp_optimizer_restore_uses_official_conversion_seam():
    calls = []

    class FakeFSDP:
        @staticmethod
        def optim_state_dict_to_load(model, optimizer, state):
            calls.append((model, optimizer, state))
            return {"converted": state}

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = fsdp_optimizer_state_to_load(
        model, optimizer, {"full": 1}, fsdp_api=FakeFSDP
    )
    assert result == {"converted": {"full": 1}}
    assert calls == [(model, optimizer, {"full": 1})]


def test_checkpoint_resume_restores_full_transaction_and_rng(tmp_path):
    trainer = _trainer(tmp_path, "a0")
    trainer.train_step(_batch())
    trainer.epoch = 2
    path = tmp_path / "checkpoint.pt"
    trainer.save(str(path))
    expected_model = copy.deepcopy(trainer.model.state_dict())
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))

    resumed = _trainer(tmp_path, "a0")
    resumed.resume(str(path))
    actual_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual_random == expected_random
    assert (resumed.global_step, resumed.epoch, resumed.sampler_position) == (1, 2, 1)
    for key, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected_model[key])

    wrong = ResumeContract("a0", "9" * 64, resumed.resume_contract.resolved_config_sha256,
                           resumed.resume_contract.matched_contract_sha256)
    with pytest.raises(ValueError, match="resume contract mismatch"):
        load_training_checkpoint(
            path, model=resumed.model, optimizer=resumed.optimizer,
            scheduler=resumed.scheduler, expected_contract=wrong,
        )


def test_head_free_export_loads_into_a0_without_constructing_head(tmp_path):
    trainer = _trainer(tmp_path, "a1o")
    source = tmp_path / "a1o.pt"
    trainer.save(str(source))
    destination = tmp_path / "a1o-drop.pt"
    audit = export_head_free_checkpoint(source, destination)
    assert audit.passed and audit.dropped_state_head_keys
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    clean = TinyModel()
    clean.load_state_dict(payload["model"], strict=True)
    assert not hasattr(clean, "parta_state_head")
    assert payload["qa_forward_contract"] == "a0_shared_forward_v1"
    audit, report = load_head_free_artifact(TinyModel(), destination)
    assert audit.passed
    assert report["missing_keys"] == report["unexpected_keys"] == []


def test_a1o_head_init_preserves_shared_rng_stream_and_weights():
    seed_matched_run(42)
    a0 = TinyModel()
    expected_random = torch.rand(4)
    expected_shared = copy.deepcopy(a0.state_dict())

    seed_matched_run(42)
    a1o = TinyModel()
    attach_a1o_head_without_advancing_shared_rng(
        a1o,
        StateHeadConfig(hidden_size=8, num_categories=4, num_slots=384,
                        num_layers=1, num_heads=2, ffn_dim=16),
        seed=42,
    )
    actual_random = torch.rand(4)
    torch.testing.assert_close(actual_random, expected_random)
    for key, value in expected_shared.items():
        torch.testing.assert_close(a1o.state_dict()[key], value)


def test_epoch_cursor_resume_is_exact_and_reshuffles():
    rows = ([{"source_sampling_key": "adt"} for _ in range(7)]
            + [{"source_sampling_key": "hypersim"} for _ in range(7)])
    uninterrupted = SourceBalancedCursor(rows, seed=42)
    sequence = [uninterrupted.next_index() for _ in range(10)]
    interrupted = SourceBalancedCursor(rows, seed=42)
    prefix = [interrupted.next_index() for _ in range(6)]
    resumed = SourceBalancedCursor(rows, seed=42, epoch=interrupted.epoch,
                                   position=interrupted.position)
    assert prefix + [resumed.next_index() for _ in range(4)] == sequence
    first = SourceBalancedCursor(rows, seed=42, epoch=0).order()
    second = SourceBalancedCursor(rows, seed=42, epoch=1).order()
    assert first != second


def test_non_primary_logger_and_checkpoint_writes_are_suppressed(tmp_path):
    trainer = _trainer(tmp_path, "a0")
    trainer.is_primary = False
    trainer.logger = JsonlTrainingLogger(tmp_path / "rank1.jsonl", enabled=False)
    trainer.train_step(_batch())
    trainer.save(str(tmp_path / "rank1.pt"))
    assert not (tmp_path / "rank1.jsonl").exists()
    assert not (tmp_path / "rank1.pt").exists()
    assert not DistributedContext(rank=1, local_rank=1, world_size=2).is_primary


def test_engineering_checkpoint_is_non_promotable(tmp_path):
    trainer = _trainer(tmp_path, "a0")
    trainer.resume_contract = ResumeContract(
        "a0", "1" * 64, config_sha256(trainer.config), "3" * 64,
        transaction_kind="engineering", promotable=False,
    )
    path = tmp_path / "engineering.pt"
    trainer.save(str(path))
    formal = _trainer(tmp_path / "formal", "a0")
    with pytest.raises(ValueError, match="resume contract mismatch"):
        formal.resume(str(path))
