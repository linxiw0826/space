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
        mope_groups: int = 4,
        mope_frames_per_group: int = 4,
        mope_input_size: int = 224,
        mope_pool_mode: str = "temporal",
        **kwargs,
    ) -> None:
        mope_all_frames = int(mope_all_frames)
        mope_groups = int(mope_groups)
        mope_frames_per_group = int(mope_frames_per_group)
        mope_input_size = int(mope_input_size)
        if (mope_all_frames, mope_groups, mope_frames_per_group, mope_pool_mode) != (16, 4, 4, "temporal"):
            raise ValueError("final515k eval requires 16 frames sampled as 4x4 with temporal pooling")
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        # final515k is a strict two-encoder evaluation.  A bad/missing video
        # must never turn a sample into an unreported GUIDE-only evaluation.
        self._mope_eval_fail_closed = True
        self.mope_all_frames = mope_all_frames
        self.mope_groups = mope_groups
        self.mope_frames_per_group = mope_frames_per_group
        self.mope_input_size = mope_input_size
        self.mope_pool_mode = mope_pool_mode
        # Qwen3_VL_MY.model is the canonical Accelerate/DDP unwrap property.
        # self._model may already be DistributedDataParallel at this point and
        # therefore does not expose the transformer's ``model``/``config``.
        base_model = self.model
        inner = base_model.model
        llm_dim = base_model.config.text_config.hidden_size
        encoder = MoPENewEncoder(
            mope_checkpoint_path, source_root=mope_source_root,
            num_frames=mope_all_frames, groups=mope_groups,
            frames_per_group=mope_frames_per_group,
            input_size=mope_input_size, pool_mode=mope_pool_mode,
        )
        projector = MoPEProjectorCrossAttn(mope_dim=768, llm_dim=llm_dim)
        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)
        load_saved_eval_components(encoder, projector, pretrained, encoder.contract)
        reference = next(base_model.parameters())
        inner._mope_encoder.to(device=reference.device, dtype=reference.dtype)
        inner._mope_projector.to(device=reference.device, dtype=reference.dtype)
        # These modules are attached after Accelerate prepared the base model,
        # so inherit eval mode explicitly instead of relying on parent state.
        inner._mope_encoder.eval()
        inner._mope_projector.eval()
        _patch_model_for_mope_crossattn(base_model)
        print(
            "[MoPE-final515k eval] "
            f"checkpoint={pretrained}, model=qwen3_vl_mope_new_crossattn, "
            f"mope_checkpoint={mope_checkpoint_path}, frames={mope_all_frames}, "
            f"sampling=4x4, pos=3d_sincos, pool={mope_pool_mode}, "
            "expected_features=[B,8,768]",
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
            video, groups=self.mope_groups,
            frames_per_group=self.mope_frames_per_group,
            input_size=self.mope_input_size,
        )
        return frames.unsqueeze(0)
