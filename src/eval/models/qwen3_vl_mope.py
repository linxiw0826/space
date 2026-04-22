"""
qwen3_vl_mope.py — lmms-eval plugin for E-02a inference.

Registers model type ``qwen3_vl_mope`` via LMMS_EVAL_PLUGINS.

Inherits from Qwen3_VL_MY and applies _patch_model_for_mope() at __init__
time, then injects mope_frames via the sidecar mechanism in generate_until.

Usage (set before running lmms-eval):
    export LMMS_EVAL_PLUGINS="src.eval"
    export PYTHONPATH="${SPACE_ROOT}:${PYTHONPATH}"

Requires: lmms_eval in PYTHONPATH (set by eval_e02a_vsibench.sh via GUIDE_LMMS_EVAL).
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — ensure lmms_eval, qwenvl, and model packages are importable.
# __file__ is src/eval/models/qwen3_vl_mope.py
# parents[3] → the space/ project root
# ---------------------------------------------------------------------------
_SPACE_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _SPACE_ROOT / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Imports — must come after sys.path setup
# ---------------------------------------------------------------------------
from lmms_eval.api.registry import register_model  # noqa: E402
from lmms_eval.models.simple.qwen3_vl_my import Qwen3_VL_MY  # noqa: E402
from model.mope_patch import _patch_model_for_mope  # noqa: E402


def _load_mope_weights_from_pretrained(inner_model, pretrained_path: str) -> bool:
    """从 E-02a checkpoint 提取并加载 _mope_encoder / _mope_projector 权重。

    Checkpoint 中的 keys 格式：model._mope_encoder.*, model._mope_projector.*
    加载到 inner_model（Qwen3VLModel）时需去掉 'model.' 前缀。
    支持 safetensors（含分片）和 pytorch_model*.bin 两种格式。
    返回 True 表示加载到至少 1 个 MoPE 权重，False 表示失败或无 MoPE 权重。
    """
    ckpt_dir = Path(pretrained_path)
    state_dict = {}

    try:
        from safetensors.torch import load_file as _st_load_file
        shards = sorted(ckpt_dir.glob("*.safetensors"))
        if shards:
            for shard in shards:
                shard_data = _st_load_file(str(shard))
                for k, v in shard_data.items():
                    if k.startswith("model._mope_encoder.") or k.startswith("model._mope_projector."):
                        state_dict[k[len("model."):]] = v
    except ImportError:
        pass

    if not state_dict:
        import torch as _torch
        bin_files = sorted(ckpt_dir.glob("pytorch_model*.bin"))
        for bin_file in bin_files:
            bin_data = _torch.load(str(bin_file), map_location="cpu", weights_only=True)
            for k, v in bin_data.items():
                if k.startswith("model._mope_encoder.") or k.startswith("model._mope_projector."):
                    state_dict[k[len("model."):]] = v

    if not state_dict:
        print(f"[Qwen3_VL_MoPE] WARNING: no MoPE keys in checkpoint at {ckpt_dir}")
        return False

    msg = inner_model.load_state_dict(state_dict, strict=False)  # noqa: F841
    print(f"[Qwen3_VL_MoPE] Loaded {len(state_dict)} MoPE tensors from checkpoint.")
    return True


# ImageNet normalisation constants for MoPE encoder input
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@register_model("qwen3_vl_mope")
class Qwen3_VL_MoPE(Qwen3_VL_MY):
    """Qwen3-VL + MoPE inference model for lmms-eval.

    Loads an E-02a checkpoint that has _mope_encoder / _mope_projector
    weights saved alongside the base Qwen3-VL parameters, applies the
    MoPE monkey-patch to inner_model.forward(), and injects mope_frames
    via the sidecar mechanism on every generate_until call.

    Interface:
        __init__(pretrained, ..., mope_all_frames=8)
        generate_until(requests) -> List[str]
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames

        inner = self._model.model  # Qwen3VLModel / Qwen2_5_VLModel
        llm_dim = self._model.config.hidden_size

        from model.mope_encoder import MoPEEncoder
        from model.mope_projector import MoPEProjector

        encoder = MoPEEncoder(checkpoint_path=None, all_frames=self.mope_all_frames)
        projector = MoPEProjector(mope_dim=768, llm_dim=llm_dim)

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        if success:
            _patch_model_for_mope(self._model)
            print(
                f"[Qwen3_VL_MoPE] MoPE patch applied. "
                f"mope_all_frames={self.mope_all_frames}"
            )
        else:
            print(
                "[Qwen3_VL_MoPE] WARNING: MoPE weights could not be loaded from checkpoint "
                "— running as standard GUIDE model."
            )

    # ------------------------------------------------------------------
    # MoPE frame computation
    # ------------------------------------------------------------------

    def _compute_mope_frames(self, visuals) -> Optional[torch.Tensor]:
        """Sample T frames from the first video in visuals and return a
        [1, 3, T, 224, 224] float tensor in ImageNet-normalised space.

        Args:
            visuals: list of (str path | PIL.Image) or a single such item.

        Returns:
            Tensor of shape [1, 3, T, 224, 224] on CPU, dtype float32.
            None if no video path is found or any error occurs.
        """
        try:
            import decord  # import here to avoid hard dep at module level

            # Normalise to list
            if not isinstance(visuals, (list, tuple)):
                visuals = [visuals]

            video_path = None
            for v in visuals:
                if isinstance(v, str) and v.lower().endswith(_VIDEO_EXTENSIONS):
                    video_path = v
                    break

            if video_path is None:
                print("[Qwen3_VL_MoPE] No video path found in visuals — MoPE skipped for this sample.")
                return None

            T = self.mope_all_frames
            vr = decord.VideoReader(video_path)
            total = len(vr)

            if total >= T:
                indices = np.linspace(0, total - 1, T, dtype=int)
            else:
                # Repeat last frame to fill T slots
                indices = list(range(total))
                indices += [total - 1] * (T - total)
                indices = np.array(indices, dtype=int)

            mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

            frame_tensors = []
            for idx in indices:
                frame_np = vr[int(idx)].asnumpy()           # H x W x 3, uint8
                pil_frame = Image.fromarray(frame_np).convert("RGB")
                pil_frame = pil_frame.resize((224, 224), Image.BILINEAR)
                arr = np.array(pil_frame, dtype=np.float32) / 255.0  # H x W x 3
                t = torch.from_numpy(arr).permute(2, 0, 1)           # 3 x 224 x 224
                t = (t - mean) / std
                frame_tensors.append(t)

            # Stack: [T, 3, 224, 224] -> permute -> [3, T, 224, 224] -> unsqueeze -> [1, 3, T, 224, 224]
            frames = torch.stack(frame_tensors, dim=0)         # [T, 3, 224, 224]
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, 224, 224]
            return frames

        except Exception as exc:
            print(
                f"[Qwen3_VL_MoPE] WARNING: _compute_mope_frames failed "
                f"({type(exc).__name__}: {exc}). MoPE will be skipped for this sample."
            )
            return None

    # ------------------------------------------------------------------
    # generate_until — process one request at a time so each sample can
    # have its own mope_frames injected via the sidecar.
    # ------------------------------------------------------------------

    def generate_until(self, requests) -> List[str]:
        """Process requests one-by-one, injecting MoPE frames per sample.

        Each request element has the structure produced by lmms-eval's
        Collator: (context, gen_kwargs, doc_to_visual, doc_id, task, split).
        We extract visuals by calling doc_to_visual(self.task_dict[task][split][doc_id]).
        """
        results = []

        for request in requests:
            # ----------------------------------------------------------
            # Extract visuals from this single request.
            # request.args = (context, gen_kwargs, doc_to_visual, doc_id, task, split)
            # ----------------------------------------------------------
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

            try:
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            except Exception as exc:
                print(
                    f"[Qwen3_VL_MoPE] WARNING: could not extract visuals "
                    f"({type(exc).__name__}: {exc}). MoPE skipped."
                )
                visuals = None

            mope_frames = self._compute_mope_frames(visuals) if visuals is not None else None

            inner = self._model.model
            try:
                if mope_frames is not None:
                    # Move frames to the model's device and dtype
                    target_device = next(inner.parameters()).device
                    target_dtype = next(inner.parameters()).dtype
                    inner._pending_mope_frames = mope_frames.to(
                        device=target_device, dtype=target_dtype
                    )

                # Delegate actual generation to the parent class with a
                # single-element list so all batching/processor logic is reused.
                single_result = super().generate_until([request])
                results.extend(single_result)

            finally:
                # Always clean up the sidecar regardless of success/failure.
                inner._pending_mope_frames = None

        return results
