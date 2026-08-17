"""Strict lmms-eval plugin for the updated MoPE CrossAttn experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from lmms_eval.api.registry import register_model

from src.eval.models.qwen3_vl_mope import Qwen3_VL_MY, Qwen3_VL_MoPE_CrossAttn
from src.model.mope_new_encoder import (
    DEFAULT_SOURCE_ROOT,
    MoPENewEncoder,
    POOL_IDS,
    load_saved_eval_components,
    load_video_for_mope_new,
)
from model.mope_patch import _patch_model_for_mope_crossattn
from model.mope_projector import MoPEProjectorCrossAttn


@register_model("qwen3_vl_mope_new_crossattn")
class Qwen3VLMoPENewCrossAttn(Qwen3_VL_MoPE_CrossAttn):
    def __init__(
        self,
        pretrained: str,
        mope_checkpoint_path: str,
        mope_source_root: str = str(DEFAULT_SOURCE_ROOT),
        mope_all_frames: int = 16,
        mope_sampling_rate: int = 4,
        mope_input_size: int = 224,
        mope_pool_mode: str = "none",
        **kwargs,
    ) -> None:
        mope_all_frames = int(mope_all_frames)
        mope_sampling_rate = int(mope_sampling_rate)
        mope_input_size = int(mope_input_size)
        if mope_pool_mode not in POOL_IDS:
            raise ValueError(f"invalid MoPE-new pool mode: {mope_pool_mode}")
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames
        self.mope_sampling_rate = mope_sampling_rate
        self.mope_input_size = mope_input_size
        self.mope_pool_mode = mope_pool_mode
        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size
        encoder = MoPENewEncoder(
            mope_checkpoint_path, source_root=mope_source_root,
            num_frames=mope_all_frames, sampling_rate=mope_sampling_rate,
            input_size=mope_input_size, pool_mode=mope_pool_mode,
        )
        projector = MoPEProjectorCrossAttn(mope_dim=768, llm_dim=llm_dim)
        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)
        load_saved_eval_components(encoder, projector, pretrained, encoder.contract)
        reference = next(self._model.parameters())
        inner._mope_encoder.to(device=reference.device, dtype=reference.dtype)
        inner._mope_projector.to(device=reference.device, dtype=reference.dtype)
        _patch_model_for_mope_crossattn(self._model)
        expected_tokens = 1568 if mope_pool_mode == "none" else 8 if mope_pool_mode == "temporal" else 1
        print(
            "[MoPE-new eval] "
            f"checkpoint={pretrained}, model=qwen3_vl_mope_new_crossattn, "
            f"mope_checkpoint={mope_checkpoint_path}, frames={mope_all_frames}, "
            f"sampling_rate={mope_sampling_rate}, pool={mope_pool_mode}, "
            f"expected_features=[B,{expected_tokens},768]",
            flush=True,
        )

    def _compute_mope_frames(self, visuals) -> Optional[torch.Tensor]:
        if not isinstance(visuals, (list, tuple)):
            visuals = [visuals]
        video = next(
            (item for item in visuals if isinstance(item, str)
             and item.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"))),
            None,
        )
        if video is None:
            raise ValueError("MoPE-new eval requires a video path; refusing to skip MoPE")
        frames = load_video_for_mope_new(
            video, num_frames=self.mope_all_frames,
            sampling_rate=self.mope_sampling_rate, input_size=self.mope_input_size,
        )
        return frames.unsqueeze(0)
