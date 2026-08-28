import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import pytest

from model.mope_new_encoder import (
    MoPENewEncoder,
    load_video_for_mope_new,
    select_video_indices,
)


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


def test_output_contract_and_frozen_eval(tmp_path):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    encoder = MoPENewEncoder(ckpt, model_factory=factory)
    encoder.train()
    assert not encoder.training
    assert not encoder.encoder.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    output = encoder(torch.zeros(2, 3, 16, 224, 224))
    assert output.shape == (2, 8, 768)
    raw = torch.arange(1568 * 768, dtype=torch.float32).view(8, 196, 768)
    assert torch.equal(output[0], raw.mean(1))


def test_input_contract_rejects_wrong_frames(tmp_path):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    encoder = MoPENewEncoder(ckpt, model_factory=factory)
    with pytest.raises(ValueError, match="expected"):
        encoder(torch.zeros(1, 3, 8, 224, 224))


def test_final515k_segment_sampling_and_short_video():
    assert select_video_indices(100).tolist() == [
        0, 8, 16, 24, 25, 33, 41, 49,
        50, 58, 66, 74, 75, 83, 91, 99,
    ]
    short = select_video_indices(10)
    assert len(short) == 16 and short[0] == 0 and short[-1] == 9


def test_video_loader_uses_exact_bounded_indexed_selection(monkeypatch):
    total = 10
    readers = []

    class FakeBatch:
        def __init__(self, frames):
            self.frames = frames

        def asnumpy(self):
            return self.frames

    class FakeReader:
        def __init__(self, _, ctx=None, num_threads=None):
            self.indices = None
            readers.append(self)

        def __len__(self):
            return total

        def get_batch(self, indices):
            self.indices = indices
            return FakeBatch(np.stack([
                np.full((2, 2, 3), index, dtype=np.uint8) for index in indices
            ]))

    fake_decord = SimpleNamespace(
        VideoReader=FakeReader,
        cpu=lambda index: ("cpu", index),
    )
    monkeypatch.setitem(sys.modules, "decord", fake_decord)
    result = load_video_for_mope_new("fake.mp4", input_size=2)

    assert result.shape == (3, 16, 2, 2)
    assert len(readers) == 1
    assert readers[0].indices == select_video_indices(total).tolist()


def test_factory_receives_exact_final515k_architecture(tmp_path):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    seen = {}

    def recording_factory(**kwargs):
        seen.update(kwargs)
        return FakeNative()

    MoPENewEncoder(ckpt, model_factory=recording_factory)
    assert seen == {
        "pretrained": False, "all_frames": 16, "tubelet_size": 2,
        "encoder_depth": 12, "dense_layers": 8, "num_experts": 8,
        "top_k": 1, "num_shared_experts": 1, "router_score_func": "sigmoid",
        "router_bias_update_speed": 0.0, "num_anchors": 1,
        "pos_embed_type": "3d_sincos",
        "predictor_pos_embed_type": "3d_sincos",
    }


@pytest.mark.parametrize("pool", ["none", "mean"])
def test_non_official_pooling_is_rejected(tmp_path, pool):
    ckpt = tmp_path / "checkpoint-50.pth"
    make_checkpoint(ckpt)
    with pytest.raises(ValueError, match="pool_mode"):
        MoPENewEncoder(ckpt, pool_mode=pool, model_factory=factory)
