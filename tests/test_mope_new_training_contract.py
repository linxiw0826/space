from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import pytest
from safetensors.torch import save_file

from model.mope_new_encoder import load_projector_warmstart
from model.mope_projector import MoPEProjectorCrossAttn
from train_framework.train_space_mope_new import (
    E04A_PROJECTOR_PARAMETERS,
    configure_trainability,
    validate_resume_scope,
)
from train_framework.checkpoint_rotation import (
    predelete_for_two_slot_rotation,
    validate_deepspeed_resume_checkpoint,
)
from train_framework.data import mope_data_wrapper
from scripts.idea1_feature.verify_e04a_checkpoint import verify


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        inner = nn.Module()
        inner.add_module("_mope_encoder", nn.Linear(2, 2))
        inner.add_module("_mope_projector", MoPEProjectorCrossAttn(768, 16))
        inner.add_module("llm", nn.Linear(16, 16))
        self.model = inner


def save_projector(root, projector, mutate=None):
    state = {f"model._mope_projector.{key}": value.clone() for key, value in projector.state_dict().items()}
    if mutate:
        mutate(state)
    torch.save(state, root / "pytorch_model.bin")


def test_crossattn_accepts_final515k_8_tokens():
    projector = MoPEProjectorCrossAttn(768, 16)
    result = projector(torch.randn(1, 8, 768), torch.randn(1, 9, 16))
    assert result.shape == (1, 9, 16)


def test_trainability_contracts():
    model = FakeModel()
    counts = configure_trainability(model, "e02c-new")
    assert counts["encoder"] == 0 and counts["projector"] > 0 and counts["other"] > 0
    with pytest.raises(ValueError, match="unknown"):
        configure_trainability(FakeModel(), "e00b-new")


def test_e04a_freezes_everything_except_exact_projector_contract():
    model = FakeModel()
    model.model._mope_projector = MoPEProjectorCrossAttn(768, 2560)
    model.register_parameter("stray_parameter", nn.Parameter(torch.ones(3)))

    counts = configure_trainability(model, "e04a-new")

    assert counts == {
        "encoder": 0,
        "projector": E04A_PROJECTOR_PARAMETERS,
        "other": 0,
    }
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 8
    assert all(name.startswith("model._mope_projector.") for name in trainable)
    assert model.stray_parameter.requires_grad is False


def test_e04a_projector_is_ungated_and_exact_zero_residual_at_init():
    projector = MoPEProjectorCrossAttn(768, 2560)
    assert projector.use_gate is False
    assert not hasattr(projector, "gate_mlp")
    assert torch.count_nonzero(projector.out_proj.weight).item() == 0
    assert torch.count_nonzero(projector.out_proj.bias).item() == 0

    image = torch.randn(1, 5, 2560)
    result = projector(torch.randn(1, 8, 768), image)
    assert torch.equal(result, image)


def test_projector_warmstart_is_strict_and_trainable(tmp_path):
    source = MoPEProjectorCrossAttn(768, 16)
    with torch.no_grad():
        source.out_proj.weight.fill_(0.25)
    save_projector(tmp_path, source)
    target = MoPEProjectorCrossAttn(768, 16)
    norms = load_projector_warmstart(target, tmp_path)
    assert norms["out_proj.weight"] > 0
    assert torch.equal(target.out_proj.weight, source.out_proj.weight)
    assert all(parameter.requires_grad for parameter in target.parameters())


@pytest.mark.parametrize("kind", ["missing", "shape"])
def test_projector_warmstart_rejects_invalid_state(tmp_path, kind):
    projector = MoPEProjectorCrossAttn(768, 16)
    def mutate(state):
        key = "model._mope_projector.out_proj.weight"
        if kind == "missing":
            state.pop(key)
        else:
            state[key] = torch.zeros(2, 2)
    save_projector(tmp_path, projector, mutate)
    with pytest.raises(RuntimeError):
        load_projector_warmstart(MoPEProjectorCrossAttn(768, 16), tmp_path)


def test_resume_must_stay_in_own_output(tmp_path):
    output = tmp_path / "e03a"
    validate_resume_scope(str(output / "checkpoint-1000"), str(output))
    with pytest.raises(ValueError):
        validate_resume_scope(str(tmp_path / "e00b" / "checkpoint-1000"), str(output))


def test_strict_final515k_loading_never_falls_back_to_zeros(monkeypatch):
    class Base:
        list_data_dict = [{"image": ["missing.jpg"]}]
        def __len__(self): return 1
        def __getitem__(self, index): return {}

    monkeypatch.setattr(mope_data_wrapper, "_STRICT_MOPE_LOADING", True)
    monkeypatch.setattr(
        mope_data_wrapper, "_load_mope_frames",
        lambda annotation, frames: (_ for _ in ()).throw(ValueError("missing mope_video")),
    )
    wrapped = mope_data_wrapper.MoPEDatasetWrapper(Base(), mope_all_frames=16)
    with pytest.raises(
        RuntimeError,
        match=r"dataset index=0 sample_id=UNKNOWN video=UNKNOWN",
    ):
        wrapped[0]


def test_e02c_launcher_refuses_implicit_resume_from_existing_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    output_dir = output_root / "train" / "e02c_mope_new_crossattn_joint_4b"
    (output_dir / "checkpoint-7269").mkdir(parents=True)
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(output_root),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e02c_mope_new_crossattn_joint.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing implicit resume" in result.stderr


def test_e02c_fresh_launcher_disables_transformers_auto_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(tmp_path / "output"),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e02c_mope_new_crossattn_joint.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--overwrite_output_dir" in result.stdout
    assert "--dataloader_num_workers 2" in result.stdout
    assert "--save_steps 500" in result.stdout


def test_e02c_three_gpu_launcher_preserves_effective_batch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(tmp_path / "output"),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "NPROC_PER_NODE": "3",
        "GRAD_ACCUM": "8",
        "CUDA_VISIBLE_DEVICES": "4,6,7",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e02c_mope_new_crossattn_joint.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--nproc_per_node=3" in result.stdout
    assert "--gradient_accumulation_steps 8" in result.stdout
    assert "effective_batch=48" in result.stdout


def test_e02c_launcher_allows_two_checkpoint_rotation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(tmp_path / "output"),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "SAVE_TOTAL_LIMIT": "2",
        "PREDELETE_OLDEST_CHECKPOINT": "1",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e02c_mope_new_crossattn_joint.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--save_total_limit 2" in result.stdout
    assert "--predelete_oldest_checkpoint True" in result.stdout
    assert "save_total_limit=2" in result.stdout
    assert "predelete_oldest=1" in result.stdout


def test_e04a_dry_run_is_e01_initialized_projector_only(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(output_root),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e04a_mope_new_e01_projector_only.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected_base = output_root / "train" / "e01_guide_4b"
    expected_output = output_root / "train" / "e04a_mope_new_e01_projector_only_4b"
    assert f"init_checkpoint={expected_base}" in result.stdout
    assert f"--model_name_or_path {expected_base}" in result.stdout
    assert f"--output_dir {expected_output}" in result.stdout
    assert "--tune_mm_vision False" in result.stdout
    assert "--tune_mm_mlp False" in result.stdout
    assert "--tune_mm_llm False" in result.stdout
    assert "--mope_use_gate False" in result.stdout
    assert "--mope_feed_causal_mask False" in result.stdout
    assert "--mope_feed_temporal_pe False" in result.stdout
    assert "--load_mope_projector_from_ckpt False" in result.stdout
    assert "--freeze_mope_projector False" in result.stdout
    assert "train_other=False gate=False" in result.stdout
    assert "--nproc_per_node=3" in result.stdout
    assert "--gradient_accumulation_steps 8" in result.stdout
    assert "grad_accum=8" in result.stdout
    assert "effective_batch=48" in result.stdout
    assert "warmstart=none" in result.stdout
    assert "--overwrite_output_dir" in result.stdout


def test_e04a_resume_must_be_inside_its_own_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(output_root),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
        "RESUME_FROM_CHECKPOINT": str(output_root / "train" / "e01_guide_4b" / "checkpoint-1000"),
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e04a_mope_new_e01_projector_only.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Resume must be inside this experiment's output directory" in result.stderr


def test_deepspeed_resume_preflight_requires_all_rank_shards(tmp_path):
    checkpoint = tmp_path / "checkpoint-3000"
    tag_dir = checkpoint / "global_step3000"
    tag_dir.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text('{"global_step":3000}')
    (checkpoint / "latest").write_text("global_step3000")
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"model")
    for rank in range(4):
        (tag_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"optim")

    audit = validate_deepspeed_resume_checkpoint(checkpoint, 4)
    assert audit["global_step"] == 3000
    assert audit["optimizer_shards"] == 4

    (tag_dir / "bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete DeepSpeed shards"):
        validate_deepspeed_resume_checkpoint(checkpoint, 4)


def test_e04a_posttrain_verifier_proves_frozen_backbone_and_learned_projector(tmp_path):
    base = tmp_path / "e01"
    candidate = tmp_path / "e04a"
    base.mkdir()
    candidate.mkdir()
    base_tensors = {
        "model.language_model.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "lm_head.weight": torch.arange(3, dtype=torch.float32).reshape(1, 3),
    }
    projector = MoPEProjectorCrossAttn(768, 2560).state_dict()
    projector["out_proj.weight"][0, 0] = 0.5
    candidate_tensors = {
        **{key: value.clone() for key, value in base_tensors.items()},
        **{f"model._mope_projector.{key}": value for key, value in projector.items()},
        "model._mope_encoder.contract": torch.tensor([2, 12, 8, 8, 1, 1, 16, 4, 4, 224, 1]),
    }
    for root, tensors in ((base, base_tensors), (candidate, candidate_tensors)):
        shard = "model-00001-of-00001.safetensors"
        save_file(tensors, root / shard)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: shard for key in tensors}
        }))

    report = verify(base, candidate)
    assert report["status"] == "complete_passed"
    assert report["frozen_tensors_compared_exactly"] == 2
    assert report["projector_norms"]["out_proj.weight"] == pytest.approx(0.5)


def test_e04a_verifier_is_runnable_without_training_pythonpath():
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(root / "scripts/idea1_feature/verify_e04a_checkpoint.py"), "--help"],
        cwd=Path("/"),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--base" in result.stdout and "--candidate" in result.stdout


def test_e04a_requires_completed_e01_top_level_not_nested_fallback(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output"
    e01_root = output_root / "train" / "e01_guide_4b"
    nested = e01_root / "checkpoint-7269"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}")
    (nested / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}'
    )
    (nested / "model-00001-of-00001.safetensors").write_text("weights")
    mope_checkpoint = tmp_path / "checkpoint-50.pth"
    mope_checkpoint.write_text("mope")
    env = {
        **os.environ,
        "SPACE_OUTPUT_ROOT": str(output_root),
        "SPACE_LOG_ROOT": str(tmp_path / "logs"),
        "MOPE_NEW_CKPT": str(mope_checkpoint),
        "DRY_RUN": "1",
    }
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/train/train_e04a_mope_new_e01_projector_only.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires the completed E-01 top-level HF checkpoint" in result.stderr


def test_two_slot_rotation_deletes_placeholder_and_keeps_complete_latest(tmp_path):
    output = tmp_path / "train"
    placeholder = output / "checkpoint-2000"
    latest = output / "checkpoint-3000"
    placeholder.mkdir(parents=True)
    (placeholder / "partial-shard.pt").write_bytes(b"partial")
    latest.mkdir()
    (latest / "trainer_state.json").write_text("{}")

    deleted = predelete_for_two_slot_rotation(output, 4000)

    assert deleted == [str(placeholder)]
    assert not placeholder.exists()
    assert latest.is_dir()


def test_two_slot_rotation_refuses_to_delete_when_latest_is_incomplete(tmp_path):
    output = tmp_path / "train"
    oldest = output / "checkpoint-2000"
    latest = output / "checkpoint-3000"
    oldest.mkdir(parents=True)
    (oldest / "trainer_state.json").write_text("{}")
    latest.mkdir()

    with pytest.raises(RuntimeError, match="newest recovery point is incomplete"):
        predelete_for_two_slot_rotation(output, 4000)

    assert oldest.is_dir()
    assert latest.is_dir()


def test_two_slot_rotation_refuses_existing_save_destination(tmp_path):
    output = tmp_path / "train"
    (output / "checkpoint-4000").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="destination already exists"):
        predelete_for_two_slot_rotation(output, 4000)
