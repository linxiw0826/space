from types import SimpleNamespace
import os
from pathlib import Path
import subprocess

import torch
from torch import nn
import pytest

from model.mope_new_encoder import load_projector_warmstart
from model.mope_projector import MoPEProjectorCrossAttn
from train_framework.train_space_mope_new import configure_trainability, validate_resume_scope
from train_framework.data import mope_data_wrapper


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
    with pytest.raises(RuntimeError, match="strict MoPE frame loading failed"):
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
