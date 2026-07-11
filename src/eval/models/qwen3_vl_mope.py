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
import torch.nn as nn
import os
# new-server cuDNN init workaround, gated by SPACE_DISABLE_CUDNN
if os.environ.get("SPACE_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
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
from model.mope_patch import (  # noqa: E402
    _patch_model_for_mope,
    _patch_model_for_mope_concat,
    _patch_model_for_mope_crossattn,
    _patch_model_for_mope_qformer,
)


def _read_feed_causal_mask() -> bool:
    """Read the E-16e feed-causal-mask eval switch from the environment.

    Returns True when ``MOPE_FEED_CAUSAL_MASK`` is set truthy (``1``/``true``,
    case-insensitive), else False. UNSET or ``0``/``false`` -> False, in which
    case the crossattn patch is called WITHOUT ``feed_causal_mask`` and the eval
    is byte-for-byte identical to the pre-E-16e path (bidirectional feed
    cross-attn = E-03a / E-16 / E-16b).

    Why this MUST be set for E-16e: E-16e TRAINS with a causal feed mask
    (``--mope_feed_causal_mask``), and the feed cross-attn also runs at
    inference. Leaving it OFF at eval would run the feed bidirectional ->
    train/test mismatch -> invalid E-16e scores. Set ``MOPE_FEED_CAUSAL_MASK=1``
    ONLY when evaluating an E-16e checkpoint; leave it unset for every other
    checkpoint (E-03a/E-16/E-16b) so their eval stays byte-equivalent.
    """
    return os.environ.get("MOPE_FEED_CAUSAL_MASK", "0").strip().lower() in ("1", "true")


def _read_temporal_pe() -> bool:
    """Read the E-16t feed temporal-PE eval switch from the environment.

    Returns True when ``MOPE_FEED_TEMPORAL_PE`` is set truthy (``1``/``true``,
    case-insensitive), else False. UNSET or ``0``/``false`` -> False, so the
    crossattn projector is built WITHOUT the temporal encoding and the eval is
    byte-for-byte identical to the pre-E-16t path (E-03a / E-16 / E-16e).

    Why this MUST be set for E-16t: an E-16t checkpoint TRAINS with
    ``--mope_feed_temporal_pe`` and the feed cross-attn (with the temporal code)
    also runs at inference. Leaving it OFF at eval would drop the temporal code
    at test time -> train/test mismatch. Set ``MOPE_FEED_TEMPORAL_PE=1`` ONLY
    when evaluating an E-16t checkpoint; leave it unset otherwise.
    """
    return os.environ.get("MOPE_FEED_TEMPORAL_PE", "0").strip().lower() in ("1", "true")


def _read_encoder_type() -> str:
    """Read the frozen-encoder backbone selection from the environment.

    Returns ``MOPE_ENCODER_TYPE`` lower-cased ('mope' default, or 'videomae').
    UNSET -> 'mope', byte-for-byte the original MoPE eval. This MUST match the
    encoder the checkpoint was TRAINED with (--mope_encoder_type): the frozen
    encoder weights are restored from the checkpoint's ``_mope_encoder.*`` keys
    with strict=False, so a mismatch (e.g. building the MoE MoPE encoder for a
    VideoMAE checkpoint) would silently fail to load the block MLP weights and
    produce a wrong (random-MLP) encoder. Set ``MOPE_ENCODER_TYPE=videomae``
    when evaluating an E-03a_vmae / E-16_vmae checkpoint.
    """
    val = os.environ.get("MOPE_ENCODER_TYPE", "mope").strip().lower()
    if val not in ("mope", "videomae"):
        raise ValueError(
            f"MOPE_ENCODER_TYPE='{val}' invalid; must be 'mope' or 'videomae'."
        )
    return val


def _build_mope_encoder(all_frames: int):
    """Build the frozen encoder (architecture only) per MOPE_ENCODER_TYPE.

    Weights are restored from the fine-tuned checkpoint's ``_mope_encoder.*``
    keys by the caller (``_load_mope_weights_from_pretrained``), exactly like the
    original MoPE path — so both branches pass ``checkpoint_path=None`` here.
    """
    encoder_type = _read_encoder_type()
    if encoder_type == "videomae":
        from model.mope_encoder import VideoMAEEncoder
        return VideoMAEEncoder(checkpoint_path=None, all_frames=all_frames)
    from model.mope_encoder import MoPEEncoder
    return MoPEEncoder(checkpoint_path=None, all_frames=all_frames)


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
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_encoder import MoPEEncoder
        from model.mope_projector import MoPEProjector

        encoder = MoPEEncoder(checkpoint_path=None, all_frames=self.mope_all_frames)
        projector = MoPEProjector(mope_dim=768, llm_dim=llm_dim)

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        # Move MoPE modules to match the base model device and dtype (typically cuda:N, bfloat16)
        ref_param = next(self._model.parameters())
        model_dtype = ref_param.dtype
        model_device = ref_param.device
        inner._mope_encoder.to(device=model_device, dtype=model_dtype)
        inner._mope_projector.to(device=model_device, dtype=model_dtype)

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

    def generate_until(self, requests: list) -> List[str]:
        """Process requests one-by-one, injecting MoPE frames per sample.

        Each request element has the structure produced by lmms-eval's
        Collator: (context, gen_kwargs, doc_to_visual, doc_id, task, split).
        We extract visuals by calling doc_to_visual(self.task_dict[task][split][doc_id]).
        """
        from tqdm import tqdm

        results = []

        for request in tqdm(requests, desc="[MoPE] generate_until", dynamic_ncols=True):
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


@register_model("qwen3_vl_mope_concat")
class Qwen3_VL_MoPE_Concat(Qwen3_VL_MoPE):
    """E-02b inference model: Qwen3-VL + MoPE concat fusion for lmms-eval.

    Identical to Qwen3_VL_MoPE except:
      - Uses MoPEProjectorConcat (no avg pool, per-token projection).
      - Applies _patch_model_for_mope_concat (prepend tokens to LLM sequence).

    Registers model type ``qwen3_vl_mope_concat`` via LMMS_EVAL_PLUGINS.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        **kwargs,
    ) -> None:
        # Call Qwen3_VL_MY.__init__ directly to skip the E-02a MoPE setup
        # in Qwen3_VL_MoPE.__init__.
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames

        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_encoder import MoPEEncoder
        from model.mope_projector import MoPEProjectorConcat

        encoder = MoPEEncoder(checkpoint_path=None, all_frames=self.mope_all_frames)
        projector = MoPEProjectorConcat(mope_dim=768, llm_dim=llm_dim)

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        ref_param = next(self._model.parameters())
        inner._mope_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
        inner._mope_projector.to(device=ref_param.device, dtype=ref_param.dtype)

        if success:
            _patch_model_for_mope_concat(self._model)
            print(
                f"[Qwen3_VL_MoPE_Concat] MoPE concat patch applied. "
                f"mope_all_frames={self.mope_all_frames}"
            )
        else:
            print(
                "[Qwen3_VL_MoPE_Concat] WARNING: MoPE weights could not be loaded "
                "— running as standard GUIDE model."
            )

    # ------------------------------------------------------------------
    # Override _compute_mope_frames to support PIL.Image list input
    # (VSIBench / SPAR evaluation passes 8 PIL.Image frames, not a video path).
    # ------------------------------------------------------------------

    def _compute_mope_frames(self, visuals) -> Optional[torch.Tensor]:
        """Sample T frames from visuals and return a [1, 3, T, 224, 224] float tensor.

        Supports two input types:
          1. Video file path (str with .mp4/.avi/.mov/.mkv extension) — delegates
             to parent class logic via decord (E-02a compatibility).
          2. List of PIL.Image objects — directly used as frames (SPAR / VSIBench
             evaluation format where lmms-eval passes 8 pre-loaded images).

        Args:
            visuals: list of (str path | PIL.Image) or a single such item.

        Returns:
            Tensor of shape [1, 3, T, 224, 224] on CPU, dtype float32.
            None if neither video path nor PIL.Image objects are found.
        """
        # Normalise to list
        if not isinstance(visuals, (list, tuple)):
            visuals = [visuals]

        # --- Path 1: video file present — delegate to parent ---
        for v in visuals:
            if isinstance(v, str) and v.lower().endswith(_VIDEO_EXTENSIONS):
                return Qwen3_VL_MoPE._compute_mope_frames(self, visuals)

        # --- Path 2: collect PIL.Image objects ---
        pil_images = [v for v in visuals if isinstance(v, Image.Image)]

        if not pil_images:
            print(
                "[Qwen3_VL_MoPE_Concat] WARNING: no video path or PIL.Image found "
                "in visuals — MoPE skipped for this sample."
            )
            return None

        try:
            T = self.mope_all_frames
            n = len(pil_images)

            # Uniform sampling; pad by repeating last frame if n < T
            if n >= T:
                indices = np.linspace(0, n - 1, T, dtype=int)
            else:
                indices = list(range(n)) + [n - 1] * (T - n)
                indices = np.array(indices, dtype=int)

            mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

            frame_tensors = []
            for idx in indices:
                pil_frame = pil_images[int(idx)].convert("RGB").resize(
                    (224, 224), Image.BILINEAR
                )
                arr = np.array(pil_frame, dtype=np.float32) / 255.0  # H x W x 3
                t = torch.from_numpy(arr).permute(2, 0, 1)           # 3 x 224 x 224
                t = (t - mean) / std
                frame_tensors.append(t)

            # [T, 3, 224, 224] -> [3, T, 224, 224] -> [1, 3, T, 224, 224]
            frames = torch.stack(frame_tensors, dim=0)         # [T, 3, 224, 224]
            frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, 224, 224]
            return frames

        except Exception as exc:
            print(
                f"[Qwen3_VL_MoPE_Concat] WARNING: _compute_mope_frames (PIL path) failed "
                f"({type(exc).__name__}: {exc}). MoPE will be skipped for this sample."
            )
            return None


@register_model("qwen3_vl_mope_zeroshot")
class Qwen3_VL_MoPE_ZeroShot(Qwen3_VL_MoPE_Concat):
    """E-00b inference model: GUIDE checkpoint + MoPE concat, zero-shot (no training).

    Loads a GUIDE checkpoint (no MoPE weights) and inserts a randomly-initialised
    MoPEProjectorConcat.  Unlike Qwen3_VL_MoPE_Concat, this class:
      - Never calls _load_mope_weights_from_pretrained (GUIDE checkpoint has no MoPE keys).
      - Always calls _patch_model_for_mope_concat regardless of weight-loading outcome.
      - Initialises projector.proj.weight with normal(std=0.02) instead of zeros, so
        MoPE tokens carry real (random) content and the experiment is not equivalent to
        E-00 (which has no MoPE tokens at all).

    Purpose: ablation to test whether the GUIDE LLM can zero-shot utilise randomly-
    projected MoPE features.  Expected result: E-00b ≈ E-00 or slightly worse, proving
    that training is necessary to exploit MoPE signals.

    Registers model type ``qwen3_vl_mope_zeroshot`` via LMMS_EVAL_PLUGINS.

    Interface:
        __init__(pretrained, ..., mope_all_frames=8)
        generate_until(requests) -> List[str]   [inherited from Qwen3_VL_MoPE_Concat]
        _compute_mope_frames(visuals) -> Optional[Tensor]  [inherited from Qwen3_VL_MoPE_Concat]
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        **kwargs,
    ) -> None:
        # Call Qwen3_VL_MY.__init__ directly — bypasses both Qwen3_VL_MoPE and
        # Qwen3_VL_MoPE_Concat __init__ to avoid any MoPE weight loading logic.
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames

        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_encoder import MoPEEncoder
        from model.mope_projector import MoPEProjectorConcat

        encoder = MoPEEncoder(checkpoint_path=None, all_frames=self.mope_all_frames)
        projector = MoPEProjectorConcat(mope_dim=768, llm_dim=llm_dim)

        # Override zero-init from MoPEProjectorConcat: use normal(std=0.02) for weight
        # so MoPE tokens carry real content. Bias stays zero (already set by constructor).
        # Zero-init would make all MoPE tokens identically zero, collapsing E-00b into E-00.
        nn.init.normal_(projector.proj.weight, std=0.02)

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        # Move to model device and dtype (typically cuda:N, bfloat16)
        ref_param = next(self._model.parameters())
        inner._mope_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
        inner._mope_projector.to(device=ref_param.device, dtype=ref_param.dtype)

        # Always apply the concat patch — independent of any weight loading.
        _patch_model_for_mope_concat(self._model)
        print(
            f"[Qwen3_VL_MoPE_ZeroShot] MoPE concat patch applied (random-init projector, no training). "
            f"mope_all_frames={self.mope_all_frames}, llm_dim={llm_dim}"
        )


@register_model("qwen3_vl_mope_crossattn")
class Qwen3_VL_MoPE_CrossAttn(Qwen3_VL_MoPE):
    """E-02c inference model: Qwen3-VL + MoPE cross-attention fusion for lmms-eval.

    Identical to Qwen3_VL_MoPE (E-02a) except:
      - Uses MoPEProjectorCrossAttn (per-shard cross-attention, no avg pool).
      - Applies _patch_model_for_mope_crossattn (fuses MoPE into image shards
        via residual cross-attention at the get_image_features call site).

    Sequence length is unchanged (cross-attention overwrites each image shard
    in-place rather than prepending new tokens).

    Registers model type ``qwen3_vl_mope_crossattn`` via LMMS_EVAL_PLUGINS.

    Interface:
        __init__(pretrained, ..., mope_all_frames=8)
        generate_until(requests) -> List[str]   [inherited from Qwen3_VL_MoPE]
        _compute_mope_frames(visuals) -> Optional[Tensor]  [inherited from Qwen3_VL_MoPE_Concat]
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        **kwargs,
    ) -> None:
        # Call Qwen3_VL_MY.__init__ directly to skip the E-02a MoPE setup
        # in Qwen3_VL_MoPE.__init__.
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames

        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_projector import MoPEProjectorCrossAttn

        # Encoder backbone per MOPE_ENCODER_TYPE (mope default / videomae for
        # E-03a_vmae / E-16_vmae); MUST match the checkpoint's trained encoder.
        encoder = _build_mope_encoder(self.mope_all_frames)
        # E-16t: build the temporal-PE projector when MOPE_FEED_TEMPORAL_PE=1
        # (must match the checkpoint); default OFF -> byte-for-byte E-16/E-03a.
        projector = MoPEProjectorCrossAttn(
            mope_dim=768, llm_dim=llm_dim, use_temporal_pe=_read_temporal_pe()
        )

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        ref_param = next(self._model.parameters())
        inner._mope_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
        inner._mope_projector.to(device=ref_param.device, dtype=ref_param.dtype)

        if success:
            # E-16e: the feed cross-attn also runs at inference, so an E-16e
            # checkpoint (trained with --mope_feed_causal_mask) MUST be evaluated
            # with the SAME causal-over-bins feed mask, else the model trains
            # causal but evaluates bidirectional (train/test mismatch -> invalid
            # scores). Gated by MOPE_FEED_CAUSAL_MASK=1; unset/0 -> feed_causal_mask
            # False, byte-for-byte the original bidirectional E-03a/E-16 eval.
            _feed_causal_mask = _read_feed_causal_mask()
            _patch_model_for_mope_crossattn(
                self._model, feed_causal_mask=_feed_causal_mask
            )
            print(
                f"[Qwen3_VL_MoPE_CrossAttn] MoPE cross-attn patch applied. "
                f"mope_all_frames={self.mope_all_frames}, llm_dim={llm_dim}, "
                f"feed_causal_mask={_feed_causal_mask}"
            )
        else:
            print(
                "[Qwen3_VL_MoPE_CrossAttn] WARNING: MoPE weights could not be loaded "
                "— running as standard GUIDE model."
            )

    # _compute_mope_frames: inherit from Qwen3_VL_MoPE_Concat for PIL.Image list support.
    _compute_mope_frames = Qwen3_VL_MoPE_Concat._compute_mope_frames

    # generate_until: inherit from Qwen3_VL_MoPE (sidecar _pending_mope_frames injection).
    # No override needed — Qwen3_VL_MoPE.generate_until already uses the correct pattern.


@register_model("qwen3_vl_mope_router")
class Qwen3_VL_MoPE_Router(Qwen3_VL_MoPE_CrossAttn):
    """E-10 (Router v1) inference model: Qwen3-VL + MoPE cross-attention with a
    learned content-driven scalar gate g modulating the MoPE residual.

    Identical to Qwen3_VL_MoPE_CrossAttn (E-03a) except that the projector is
    instantiated with ``use_gate=True``.  This is a SEPARATE model type rather
    than a flag on the crossattn class so that:
      - the gate weights have a matching projector to load into (avoids the
        ``load_state_dict(strict=False)`` silently dropping the gate weights and
        the model running with an untrained / wrong-shaped gate); and
      - we assert below that the gate weights were actually loaded from the
        E-10 checkpoint, so a missing-gate situation fails loudly instead of
        silently degrading to ungated routing.

    The content-driven gate is automatically active at inference: the patched
    forward sees ``input_ids`` and pools the question text to condition g
    (see mope_patch._patch_model_for_mope_crossattn / _compute_pooled_text).

    Registers model type ``qwen3_vl_mope_router`` via LMMS_EVAL_PLUGINS.

    Interface:
        __init__(pretrained, ..., mope_all_frames=8, mope_gate_mode="learned")
        generate_until(requests) -> List[str]   [inherited from Qwen3_VL_MoPE]
        _compute_mope_frames(visuals) -> Optional[Tensor]  [inherited]
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        mope_gate_mode: str = "learned",
        **kwargs,
    ) -> None:
        # Call Qwen3_VL_MY.__init__ directly to skip the ungated crossattn MoPE
        # setup in Qwen3_VL_MoPE_CrossAttn.__init__.
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames
        self.mope_gate_mode = mope_gate_mode

        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_projector import MoPEProjectorCrossAttn

        # Encoder backbone per MOPE_ENCODER_TYPE (must match the checkpoint).
        encoder = _build_mope_encoder(self.mope_all_frames)
        # E-10: gated projector — must match the checkpoint so gate weights load.
        # E-16t (CONCERN-5): temporal-PE wired here TOO so a router checkpoint
        # trained with temporal PE stays consistent; default OFF = byte-for-byte E-10.
        projector = MoPEProjectorCrossAttn(
            mope_dim=768,
            llm_dim=llm_dim,
            use_gate=True,
            gate_mode=self.mope_gate_mode,
            use_temporal_pe=_read_temporal_pe(),
        )

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        # Assert the gate weights were actually loaded — guards against the
        # strict=False silent-drop failure mode described above.
        if success:
            gate_loaded = any(
                k.startswith("_mope_projector.gate_mlp.")
                for k, _ in inner.state_dict().items()
            ) and any(
                p.abs().sum().item() != 0.0
                for n, p in inner._mope_projector.named_parameters()
                if n.startswith("gate_mlp.")
            )
            if not gate_loaded:
                raise RuntimeError(
                    "[Qwen3_VL_MoPE_Router] gate_mlp weights are absent or all-zero "
                    "after loading from "
                    f"'{pretrained}'. The model type 'qwen3_vl_mope_router' REQUIRES a "
                    "checkpoint that contains trained gate_mlp weights, i.e. an E-10 "
                    "(gated/router) checkpoint. The provided checkpoint appears to be a "
                    "non-gated (e.g. E-03a) checkpoint. Refusing to run, because the "
                    "gate would fall back to its init value (g≈sigmoid(init_bias)≈0.98) "
                    "and produce misleading routing numbers. Use an E-10 checkpoint, or "
                    "evaluate the non-gated checkpoint with its matching model type."
                )

        ref_param = next(self._model.parameters())
        inner._mope_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
        inner._mope_projector.to(device=ref_param.device, dtype=ref_param.dtype)

        if success:
            # E-16e: mirror the crossattn eval — an E-16e checkpoint whose feed
            # cross-attn was trained causal must also be evaluated causal
            # (train/test consistency). Gated by MOPE_FEED_CAUSAL_MASK=1;
            # unset/0 -> False, byte-for-byte the original bidirectional E-10 eval.
            _feed_causal_mask = _read_feed_causal_mask()
            _patch_model_for_mope_crossattn(
                self._model, feed_causal_mask=_feed_causal_mask
            )
            print(
                f"[Qwen3_VL_MoPE_Router] MoPE cross-attn + content-driven gate patch applied. "
                f"mope_all_frames={self.mope_all_frames}, gate_mode={self.mope_gate_mode}, "
                f"llm_dim={llm_dim}, feed_causal_mask={_feed_causal_mask}"
            )
        else:
            print(
                "[Qwen3_VL_MoPE_Router] WARNING: MoPE weights could not be loaded "
                "— running as standard GUIDE model."
            )

    # _compute_mope_frames inherited from Qwen3_VL_MoPE_CrossAttn.
    #
    # generate_until OVERRIDDEN below to add the two E-10 eval diagnostics:
    #   ① per-question gate-value (g) logging  — VSI + VLM4D
    #   ② VSI oracle hard-gating by static/dynamic task label  — VSI only
    # Both are pure-inference; learned mode (GATE_MODE unset/=learned) keeps the
    # inherited behaviour byte-for-byte (no oracle sidecar set; g-log is a passive
    # read-out that never alters numerics).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VSI static/dynamic task axis (D-15 oracle = task-label gating).
    # Source: lmms_eval/tasks/vsibench/utils.py — MCA_QUESTION_TYPES are the
    # dynamic subtasks (rel direction/distance, route planning, appearance
    # order), NA_QUESTION_TYPES are the static metric subtasks (abs distance,
    # counting, size, room size). The doc's question_type for rel_direction is
    # suffixed _easy/_medium/_hard, so we match by prefix for robustness.
    # ------------------------------------------------------------------
    _VSI_DYNAMIC_SUBTASKS = (
        "object_rel_direction",   # matches _easy/_medium/_hard via prefix
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    )
    _VSI_STATIC_SUBTASKS = (
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    )

    @staticmethod
    def _vsi_is_dynamic(question_type):
        """Return True if a VSI question_type is on the DYNAMIC axis, False if
        STATIC, or None if the type is unrecognised (oracle then skips it)."""
        if question_type is None:
            return None
        qt = str(question_type)
        if any(qt.startswith(d) for d in Qwen3_VL_MoPE_Router._VSI_DYNAMIC_SUBTASKS):
            return True
        if qt in Qwen3_VL_MoPE_Router._VSI_STATIC_SUBTASKS:
            return False
        return None

    @staticmethod
    def _vlm4d_source(doc):
        """Derive the VLM4D video source (ego4d / davis / youtube-vos /
        videos_synthetic) from the doc's video URL — mirrors the ``source``
        field that vlm4d_process_results computes (dir name of the local
        relative path). Returns None if the field is absent."""
        try:
            import os as _os
            video = doc.get("video")
            if video is None:
                return None
            url = str(video)
            rel = url.split("resolve/main/")[-1] if "resolve/main/" in url else _os.path.basename(url)
            return _os.path.basename(_os.path.dirname(rel))
        except Exception:
            return None

    def _gate_log_path(self):
        """Per-rank gate-log output path, or None when logging is disabled.

        Set via env ``E10_GATE_LOG`` (the eval scripts export it). Each rank
        writes its own file (``<base>.rank{r}.jsonl``) to avoid concurrent-write
        clobbering under accelerate data parallelism; merge across ranks after
        the run with a simple cat (doc_id is the alignment key)."""
        import os as _os
        base = _os.environ.get("E10_GATE_LOG")
        if not base:
            return None
        root, ext = _os.path.splitext(base)
        if not ext:
            ext = ".jsonl"
        return f"{root}.rank{self._rank}{ext}"

    def generate_until(self, requests: list) -> List[str]:
        """E-10 router eval loop: inject MoPE frames (sidecar, inherited pattern)
        + optionally force the oracle gate (② VSI only) + log per-question g (①).

        GATE_MODE env: "learned" (default) | "oracle".
          - learned: gate runs from the trained gate_mlp; no override. Identical
            numerics to the inherited crossattn/router path.
          - oracle : VSI only — g is hard-set to 1.0 for dynamic subtasks and 0.0
            for static subtasks (per the VSI static/dynamic task axis). For
            non-VSI tasks (e.g. VLM4D) or unrecognised subtasks, no override is
            applied (falls back to the learned gate) — VLM4D has no clean
            static/dynamic task axis, so ② is VSI-only by design.
        """
        import json
        import os as _os
        from tqdm import tqdm

        gate_mode = _os.environ.get("GATE_MODE", "learned").strip().lower()
        log_path = self._gate_log_path()
        if log_path is not None:
            _os.makedirs(_os.path.dirname(log_path) or ".", exist_ok=True)

        inner = self._model.model
        projector = getattr(inner, "_mope_projector", None)

        # E-10b change C: enable the per-forward residual diagnostics on the
        # projector ONLY when g-logging is on. When E10_GATE_LOG is unset the
        # projector's _eval_resid_active stays False and the default learned-eval
        # path is byte-for-byte identical to E-10 (no extra norm computation).
        if projector is not None and hasattr(projector, "_eval_resid_active"):
            projector._eval_resid_active = log_path is not None

        results = []
        log_fh = open(log_path, "a", encoding="utf-8") if log_path is not None else None
        try:
            for request in tqdm(
                requests, desc="[MoPE-Router] generate_until", dynamic_ncols=True
            ):
                context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args

                # Fetch the doc to read subtask (VSI) / source (VLM4D) + oracle label.
                doc = None
                try:
                    doc = self.task_dict[task][split][doc_id]
                except Exception as exc:
                    print(
                        f"[Qwen3_VL_MoPE_Router] WARNING: could not fetch doc "
                        f"({type(exc).__name__}: {exc}); g-log/oracle degraded for this sample."
                    )

                # Discriminate the benchmark by TASK NAME, not by field presence:
                # VLM4D docs ALSO carry a "question_type" field (constant
                # "multiple-choice", see vlm4d/utils.py FIELD_QUESTION_TYPE), so a
                # field-presence test would misclassify VLM4D as VSI and suppress
                # the per-source g breakdown. The task name is the reliable axis:
                # only the VSI-Bench task gets the subtask/oracle path; everything
                # else (VLM4D) gets the _vlm4d_source breakdown.
                is_vsi = task is not None and "vsi" in str(task).lower()
                subtask = doc.get("question_type") if (is_vsi and doc is not None) else None
                source = self._vlm4d_source(doc) if (doc is not None and not is_vsi) else None

                # ---- ② VSI oracle hard-gate: set the per-sample sidecar. ----
                oracle_g = None
                if gate_mode == "oracle" and is_vsi and projector is not None:
                    is_dyn = self._vsi_is_dynamic(subtask)
                    if is_dyn is not None:
                        oracle_g = 1.0 if is_dyn else 0.0  # dynamic -> MoPE on, static -> off
                if projector is not None:
                    projector._pending_oracle_gate = oracle_g
                    # ---- ① reset the per-question g buffer before generation. ----
                    if hasattr(projector, "_eval_gate_log"):
                        projector._eval_gate_log.clear()
                    # ---- change C: reset the per-question residual diag buffer. ----
                    if hasattr(projector, "_eval_resid_log"):
                        projector._eval_resid_log.clear()

                # ---- MoPE frames sidecar (inherited pattern). ----
                try:
                    visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                except Exception as exc:
                    print(
                        f"[Qwen3_VL_MoPE_Router] WARNING: could not extract visuals "
                        f"({type(exc).__name__}: {exc}). MoPE skipped."
                    )
                    visuals = None
                mope_frames = (
                    self._compute_mope_frames(visuals) if visuals is not None else None
                )

                try:
                    if mope_frames is not None:
                        target_device = next(inner.parameters()).device
                        target_dtype = next(inner.parameters()).dtype
                        inner._pending_mope_frames = mope_frames.to(
                            device=target_device, dtype=target_dtype
                        )
                    single_result = super(Qwen3_VL_MoPE, self).generate_until([request])
                    results.extend(single_result)
                finally:
                    inner._pending_mope_frames = None
                    if projector is not None:
                        projector._pending_oracle_gate = None

                # ---- ① drain + aggregate per-question g, then log. ----
                if log_fh is not None and projector is not None:
                    gvals = list(getattr(projector, "_eval_gate_log", []))
                    g_mean = sum(gvals) / len(gvals) if gvals else None
                    # ---- change C: drain + aggregate per-question residual diag. ----
                    rvals = list(getattr(projector, "_eval_resid_log", []))
                    if rvals:
                        mope_out_norm = sum(r[0] for r in rvals) / len(rvals)
                        image_embeds_norm = sum(r[1] for r in rvals) / len(rvals)
                        resid_ratio = sum(r[2] for r in rvals) / len(rvals)
                    else:
                        mope_out_norm = image_embeds_norm = resid_ratio = None
                    record = {
                        "sample_id": doc_id,
                        "task": task,
                        "gate_mode": gate_mode,
                        "g": g_mean,                 # per-question MEAN over forwards
                        "g_num_forwards": len(gvals),
                        "g_oracle_forced": oracle_g,  # the forced value, or null
                        # change C: per-question MEAN over forwards of each norm.
                        "mope_out_norm": mope_out_norm,            # ‖g·mope_out‖
                        "image_embeds_norm": image_embeds_norm,    # ‖image_embeds‖
                        "resid_ratio": resid_ratio,                # ‖g·mope_out‖/‖image_embeds‖
                    }
                    if is_vsi:
                        record["subtask"] = subtask
                        record["is_dynamic"] = self._vsi_is_dynamic(subtask)
                    if source is not None:
                        record["source"] = source
                    log_fh.write(json.dumps(record) + "\n")
                    log_fh.flush()
        finally:
            if log_fh is not None:
                log_fh.close()

        return results


@register_model("qwen3_vl_mope_qformer")
class Qwen3_VL_MoPE_QFormer(Qwen3_VL_MoPE_Concat):
    """E-02d inference model: Qwen3-VL + MoPE Q-Former fusion for lmms-eval.

    Identical to Qwen3_VL_MoPE_Concat (E-02b) except:
      - Uses MoPEProjectorQFormer (compresses 784 MoPE tokens to num_queries=32
        tokens via learned cross-attention queries).
      - Applies _patch_model_for_mope_qformer (prepend 32 tokens to LLM sequence,
        vs. 784 tokens in E-02b).

    Registers model type ``qwen3_vl_mope_qformer`` via LMMS_EVAL_PLUGINS.

    Interface:
        __init__(pretrained, ..., mope_all_frames=8, mope_num_queries=32)
        generate_until(requests) -> List[str]   [inherited from Qwen3_VL_MoPE_Concat]
        _compute_mope_frames(visuals) -> Optional[Tensor]  [inherited from Qwen3_VL_MoPE_Concat]
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        mope_all_frames: int = 8,
        mope_num_queries: int = 32,
        **kwargs,
    ) -> None:
        # Call Qwen3_VL_MY.__init__ directly to skip the E-02b MoPE setup
        # in Qwen3_VL_MoPE_Concat.__init__.
        Qwen3_VL_MY.__init__(self, pretrained=pretrained, **kwargs)
        self.mope_all_frames = mope_all_frames
        self.mope_num_queries = mope_num_queries

        inner = self._model.model
        llm_dim = self._model.config.text_config.hidden_size

        from model.mope_encoder import MoPEEncoder
        from model.mope_projector import MoPEProjectorQFormer

        encoder = MoPEEncoder(checkpoint_path=None, all_frames=self.mope_all_frames)
        projector = MoPEProjectorQFormer(
            mope_dim=768, llm_dim=llm_dim, num_queries=self.mope_num_queries
        )

        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)

        success = _load_mope_weights_from_pretrained(inner, pretrained)

        ref_param = next(self._model.parameters())
        inner._mope_encoder.to(device=ref_param.device, dtype=ref_param.dtype)
        inner._mope_projector.to(device=ref_param.device, dtype=ref_param.dtype)

        if success:
            _patch_model_for_mope_qformer(self._model)
            print(
                f"[Qwen3_VL_MoPE_QFormer] MoPE Q-Former patch applied. "
                f"mope_all_frames={self.mope_all_frames}, "
                f"mope_num_queries={self.mope_num_queries}, llm_dim={llm_dim}"
            )
        else:
            print(
                "[Qwen3_VL_MoPE_QFormer] WARNING: MoPE weights could not be loaded "
                "— running as standard GUIDE model."
            )
