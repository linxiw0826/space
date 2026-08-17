from types import SimpleNamespace

import torch
from torch import nn
import pytest

from model.mope_new_encoder import load_projector_warmstart
from model.mope_projector import MoPEProjectorCrossAttn
from train_framework.train_space_mope_new import configure_trainability, validate_resume_scope


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


def test_crossattn_accepts_1568_tokens():
    projector = MoPEProjectorCrossAttn(768, 16)
    result = projector(torch.randn(1, 1568, 768), torch.randn(1, 9, 16))
    assert result.shape == (1, 9, 16)


def test_trainability_contracts():
    e00b = FakeModel()
    counts = configure_trainability(e00b, "e00b-new")
    assert counts["encoder"] == counts["other"] == 0 and counts["projector"] > 0
    for experiment in ("e02c-new", "e03a-new"):
        model = FakeModel()
        counts = configure_trainability(model, experiment)
        assert counts["encoder"] == 0 and counts["projector"] > 0 and counts["other"] > 0


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
