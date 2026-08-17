import torch
from torch import nn
import pytest

from model.mope_new_encoder import (
    clean_state_dict,
    encoder_only_state,
    extract_state_dict,
    validate_and_load_encoder,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.predictor = nn.Linear(2, 2)


@pytest.mark.parametrize("key", ["model", "module", "state_dict"])
def test_checkpoint_container_keys(key):
    state = {"encoder.weight": torch.ones(2, 2)}
    assert extract_state_dict({key: state}) == state
    assert extract_state_dict(state) == state


def test_prefix_cleanup_and_predictor_filter():
    state = {
        "module._orig_mod.encoder.weight": torch.ones(2, 2),
        "module.predictor.weight": torch.ones(2, 2),
    }
    cleaned = clean_state_dict(state)
    assert "encoder.weight" in cleaned
    assert set(encoder_only_state(cleaned)) == {"encoder.weight"}


def test_strict_encoder_load_accepts_predictor_omission():
    model = Tiny()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert validate_and_load_encoder(model, state)


def test_strict_encoder_load_rejects_missing_and_shape():
    model = Tiny()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state.pop("encoder.bias")
    with pytest.raises(RuntimeError, match="missing"):
        validate_and_load_encoder(model, state)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["encoder.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError, match="shape_mismatch"):
        validate_and_load_encoder(model, state)
