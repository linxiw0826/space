import torch
from torch import nn
import pytest

from model.mope_new_encoder import MoPENewEncoder, select_video_indices


class FakeNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.predictor = nn.Linear(2, 2)

    def encode(self, frames, token_mask=None, record_stats=True):
        base = torch.arange(1568 * 768, device=frames.device, dtype=frames.dtype)
        return base.view(1, 1568, 768).expand(frames.shape[0], -1, -1)


def factory(**kwargs):
    return FakeNative()


def make_checkpoint(path):
    model = FakeNative()
    torch.save({"model": model.state_dict()}, path)


@pytest.mark.parametrize("pool,shape", [("none", (2, 1568, 768)), ("temporal", (2, 8, 768)), ("mean", (2, 1, 768))])
def test_output_contract_and_frozen_eval(tmp_path, pool, shape):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    encoder = MoPENewEncoder(ckpt, pool_mode=pool, model_factory=factory)
    encoder.train()
    assert not encoder.training
    assert not encoder.encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    output = encoder(torch.zeros(2, 3, 16, 224, 224))
    assert output.shape == shape
    if pool == "temporal":
        raw = torch.arange(1568 * 768, dtype=torch.float32).view(8, 196, 768)
        assert torch.equal(output[0], raw.mean(1))


def test_input_contract_rejects_wrong_frames(tmp_path):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    encoder = MoPENewEncoder(ckpt, model_factory=factory)
    with pytest.raises(ValueError, match="expected"):
        encoder(torch.zeros(1, 3, 8, 224, 224))


def test_center_stride_sampling_and_short_uniform():
    assert select_video_indices(100, 16, 4).tolist() == list(range(18, 82, 4))
    short = select_video_indices(10, 16, 4)
    assert len(short) == 16 and short[0] == 0 and short[-1] == 9
