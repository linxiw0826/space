"""mope_layer_features.py — extract per-layer MoPE encoder features (验证③ / D-13).

服务于 D-13 取层决策（MoPE 取层 + 融合算子）。
================================================================================
PENDING[D-13]: 本脚本为验证③重新设计的"运动感知特征"提供证据。D-13（MoPE 取层 +
融合算子）仍 OPEN——取哪一层的结论必须等本脚本抽出的运动感知特征跑完 probe 后才能定。
解除后，下游 MoPEEncoder 按选出的层 + 池化方式改造；本脚本无需改动（继续抽全层即可）。

背景（为什么 2026-06-07 重新设计）
----
MoPE encoder (`src/model/mope_encoder.py`) 当前只返回 transformer 末层特征
(`modeling_pretrain.py:135-141` 的 `self.norm(x_vis)`)。本脚本用 **forward hook**
挂在 `MoPEEncoder.encoder.blocks[i]`（每个 transformer block）上，收集**每一层**的
block 输出原始 token 张量 [num_patches, 768]，再按 `--pool-mode` 池化。

验证③首跑（2026-06-07）暴露的缺陷：旧逻辑对全部 784 patch（=4 时间块 × 196 空间）
做 mean-pool＝对空间+时间同时平均 → 数学上必然抹平"方向"（空间反对称 + 时变信号）→
左右 probe 全在 chance。本次新增 **运动感知池化**（lr-motion / temporal-diff），
在池化时保留时间块结构和左右空间结构，让运动方向信号可被线性探针解码。

Token 时空排布（reshape 的依据，已从 vendor 源码确认）
----
- `modeling_finetune.py:329` PatchEmbed.forward：`self.proj(x).flatten(2).transpose(1,2)`。
  Conv3d 输出 [B, C, T', H', W']（T'=T/tubelet, H'=W'=img/patch），`flatten(2)` 按
  **C-order（row-major）**展平最后三维 (T', H', W') → token 顺序为
  **idx = t*(H'*W') + h*W' + w**（时间最慢、W 最快变）。
- `modeling_pretrain.py:104-108` 进一步佐证：`num_spatial = num_patches // num_time_bins`，
  `full_time_ids = arange(num_time_bins).repeat_interleave(num_spatial)`——time index
  连续重复 num_spatial 次，即每个时间块的 num_spatial 个空间 token 连续排布。
- 因此对一层的 [num_patches, 768] token，reshape 为 **[T, H, W, 768]**（C-order，
  W 为最后/最快维），其中 left half = W∈[0, W//2)、right half = W∈[W//2, W)。

ViT-B 有 12 层 block（`encoder.get_num_layers()`），MoE 在顶部 1/3
（`moe_layer_indices = range(8, 12)`）。

GPU / CPU 哪步需要
------------------
本脚本（特征抽取）**必须 GPU**：跑 MoPE encoder forward。
配套的 `mope_probe_layers.py`（sklearn 线性探针）**纯 CPU**，登录节点直接跑。

输入数据来自哪
--------------
建议复用 VLM4D 标注（HF `shijiezhou/VLM4D`，每条有 video 路径 +
question_type 当运动类别标签：translational / rotational / action / counting /
false-positive）。也可用任意带运动类别标签的视频列表。

视频列表格式（`--video-list`，jsonl 每行一条 / 或 txt 每行一个路径）：
    jsonl 每行：{"video": "/abs/path/clip.mp4", "question_type": "translational", "id": "..."}
    txt 每行  ：/abs/path/clip.mp4         （无标签，--label-key 缺省时全部归 "unlabeled"）
`--label-key` 指定取哪个字段当标签（默认 "question_type"）。

池化模式（--pool-mode，每模式每层都产出 768 维，输出仍是 3D [N, n_layers, 768]）
----
- `spacetime-mean`（默认，向后兼容，逐字节复刻旧行为）：全部 784 patch mean → [768]。
- `lr-motion`（给左右方向）：reshape [T,H,W,768]；每时间块 left_t=W∈[0,W//2) 空间均值、
  right_t=W∈[W//2,W) 空间均值；asym_t = left_t - right_t；输出该不对称的时间变化
  asym_{T-1} - asym_0（"左右质量随时间往哪挪"＝左右运动方向）。[768]
- `temporal-diff`（给逼近/远离 looming、运动有无）：每时间块全空间均值 g_t；输出
  g_{T-1} - g_0（整体内容随时间的变化）。[768]
三模式共用同一份"hook 抽每层原始 token [num_patches,768]"逻辑，仅最后池化分叉。

输出（写到 --out-dir）
----------------------
    features.npy : float32 [N_videos, n_layers, 768]   每视频每层池化向量（按 pool-mode）
    labels.npy   : int64   [N_videos]                  标签的整数编码
    meta.json    : {n_layers, embed_dim, n_videos, video_ids, label_names,
                    label_to_id, moe_layer_indices, all_frames,
                    pool_mode, feat_dim, token_grid={T,H,W}, ...}

用法
----
    export MOPE_CKPT_PATH=/abs/path/to/mope_checkpoint.pth   # E-02a MoPE ckpt
    export PYTHONPATH=/u/lwu9/Space_sensing/projects/space:$PYTHONPATH  # so `src` is importable
    python -m src.analysis.mope_layer_features \\
        --video-list /abs/path/vlm4d.jsonl \\
        --label-key question_type \\
        --pool-mode lr-motion \\
        --out-dir /abs/path/out/mope_layer_features

约束
----
- **不修改 vendor 代码**（`src/vendor/mope/**`、`modeling_pretrain.py` 不动）；
  多层访问一律用 forward hook。
- hook 在 finally 里 remove，绝不泄漏。
- 视频读不到 → skip + warn，不崩。
- encoder 冻结 + `torch.no_grad`（MoPEEncoder.forward 已带 @torch.no_grad）。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — make the `model` package (src/model/mope_encoder.py) importable
# whether run as `python -m src.analysis.mope_layer_features` or as a file.
# __file__ = src/analysis/mope_layer_features.py ; parents[1] = src/
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing constants — copied from
#   src/eval/models/qwen3_vl_mope.py (_IMAGENET_MEAN/_STD, _VIDEO_EXTENSIONS)
# to keep the feature-extraction frame pipeline byte-identical to inference.
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# PENDING[D-13]: 当前抽 ALL 12 个 block 的输出（每层一个池化向量）以供 probe 定位
# 峰值层。D-13 取层决策解除后，下游 MoPEEncoder 只需保留所选层 + 选定的池化方式；本
# 脚本是为该决策提供证据，无需改动（继续抽全层即可对照）。
PROBE_ALL_BLOCKS = True

# PENDING[D-13]: 三种运动感知池化模式。spacetime-mean 是旧行为（默认，逐字节不变）；
# lr-motion / temporal-diff 是 2026-06-07 为验证③重新设计的运动感知特征。哪种 +
# 哪一层最能解码运动方向，是 D-13 取层结论的依据，待 probe 跑完后定。三模式每层都
# 输出 768 维（保持与 305 样本数匹配，避免维度爆炸过拟合）。
POOL_MODES = ("spacetime-mean", "lr-motion", "temporal-diff")


def _sample_mope_frames(
    video_path: str, all_frames: int = 8
) -> Optional[torch.Tensor]:
    """Sample `all_frames` frames from a video → [1, 3, T, 224, 224] float32 (CPU).

    Frame sampling + ImageNet normalisation logic is the video-path branch of
    `Qwen3_VL_MoPE._compute_mope_frames`
    (src/eval/models/qwen3_vl_mope.py:157-220), reproduced here so the analysis
    script has no hard dependency on lmms-eval / the wrapper class.

    Returns None (and warns) if the video cannot be read.
    """
    try:
        import decord  # local import: avoid hard dep at module import time

        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total == 0:
            logger.warning("Video has 0 frames, skipping: %s", video_path)
            return None

        T = all_frames
        if total >= T:
            indices = np.linspace(0, total - 1, T, dtype=int)
        else:
            # Repeat last frame to fill T slots.
            indices = list(range(total)) + [total - 1] * (T - total)
            indices = np.array(indices, dtype=int)

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

        from PIL import Image

        frame_tensors = []
        for idx in indices:
            frame_np = vr[int(idx)].asnumpy()  # H x W x 3 uint8
            pil_frame = Image.fromarray(frame_np).convert("RGB")
            pil_frame = pil_frame.resize((224, 224), Image.BILINEAR)
            arr = np.array(pil_frame, dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # 3 x 224 x 224
            t = (t - mean) / std
            frame_tensors.append(t)

        frames = torch.stack(frame_tensors, dim=0)         # [T, 3, 224, 224]
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)   # [1, 3, T, 224, 224]
        return frames

    except Exception as exc:  # noqa: BLE001 - defensive: never crash the batch
        logger.warning(
            "Frame sampling failed for %s (%s: %s) — skipping.",
            video_path, type(exc).__name__, exc,
        )
        return None


def _parse_video_list(
    list_path: str, label_key: str
) -> List[Tuple[str, str, str]]:
    """Parse the video list → list of (video_path, label, video_id).

    Supports jsonl (one JSON object per line) or plain txt (one path per line).
    For txt or when `label_key` is missing, label defaults to "unlabeled".
    Lines that fail to parse are skipped with a warning.
    """
    entries: List[Tuple[str, str, str]] = []
    p = Path(list_path)
    if not p.is_file():
        raise FileNotFoundError(f"--video-list not found: {list_path}")

    with p.open("r") as fh:
        for lineno, raw in enumerate(fh):
            line = raw.strip()
            if not line:
                continue
            video_path = None
            label = "unlabeled"
            video_id = None
            if line[0] in "{[":  # looks like JSON
                try:
                    obj = json.loads(line)
                    video_path = obj.get("video") or obj.get("video_path")
                    label = str(obj.get(label_key, "unlabeled"))
                    video_id = obj.get("id") or obj.get("video_id")
                except json.JSONDecodeError:
                    logger.warning("Bad JSON at %s:%d — skipping.", list_path, lineno + 1)
                    continue
            else:  # plain path
                video_path = line

            if not video_path:
                logger.warning(
                    "No video path at %s:%d — skipping.", list_path, lineno + 1
                )
                continue
            if video_id is None:
                video_id = Path(video_path).stem
            entries.append((video_path, label, str(video_id)))

    return entries


def load_mope_encoder(ckpt_path: Optional[str], all_frames: int, device: torch.device):
    """Build a frozen MoPEEncoder and move it to `device` (eval mode).

    Mirrors how the eval wrapper loads MoPE (qwen3_vl_mope.py:122-139):
    construct MoPEEncoder, then `.to(device)`. Checkpoint path from
    MOPE_CKPT_PATH env (or None for architecture-only smoke test).
    """
    from model.mope_encoder import MoPEEncoder

    encoder = MoPEEncoder(checkpoint_path=ckpt_path, all_frames=all_frames)
    encoder.eval()
    encoder.to(device)
    return encoder


def infer_token_grid(encoder, all_frames: int, num_patches: int) -> Tuple[int, int, int]:
    """Infer (T, H, W) of the MoPE token grid from the encoder, not hardcoded.

    Token layout confirmed from vendor source (see module docstring):
      - num_time_bins T = all_frames // tubelet_size
      - num_spatial   = num_patches // T  = H * W   (H == W for square frames)
      - flatten order = C-order over (T, H, W) → idx = t*(H*W) + h*W + w
        (modeling_finetune.py:329 flatten(2).transpose; corroborated by
         modeling_pretrain.py:104-108 time-id repeat_interleave).

    Raises a clear error if the grid cannot be inferred consistently.
    """
    vit = encoder.encoder
    patch_embed = vit.patch_embed
    tubelet = getattr(patch_embed, "tubelet_size", None)
    if tubelet is None or tubelet <= 0:
        raise RuntimeError(
            f"Cannot infer token grid: tubelet_size unavailable/invalid "
            f"(got {tubelet}). Patch embed = {type(patch_embed).__name__}."
        )
    if all_frames % tubelet != 0:
        raise RuntimeError(
            f"Cannot infer token grid: all_frames ({all_frames}) not divisible "
            f"by tubelet_size ({tubelet})."
        )
    T = all_frames // tubelet
    if T <= 0 or num_patches % T != 0:
        raise RuntimeError(
            f"Cannot infer token grid: num_patches ({num_patches}) not divisible "
            f"by num_time_bins T ({T} = all_frames {all_frames} / tubelet {tubelet})."
        )
    num_spatial = num_patches // T
    side = int(round(num_spatial ** 0.5))
    if side * side != num_spatial:
        raise RuntimeError(
            f"Cannot infer token grid: spatial patches per time bin "
            f"({num_spatial}) is not a perfect square; non-square frames "
            "unsupported. Provide square H==W input."
        )
    H = W = side
    if T * H * W != num_patches:
        raise RuntimeError(
            f"Token-grid inference inconsistent: T*H*W ({T}*{H}*{W}={T*H*W}) "
            f"!= num_patches ({num_patches})."
        )
    return T, H, W


def _pool_layer_tokens(
    tokens: torch.Tensor, pool_mode: str, grid: Tuple[int, int, int]
) -> torch.Tensor:
    """Pool one layer's raw token tensor [num_patches, C] → [C] per pool_mode.

    Token order is C-order over (T, H, W) (W fastest), confirmed from
    modeling_finetune.py:329 + modeling_pretrain.py:104-108 (see module docstring).
    All three modes return 768-d (= C).

    - spacetime-mean: mean over ALL patches  → [C]   (旧行为，逐字节不变)
    - lr-motion:      reshape [T,H,W,C]; per time-bin t:
                        left_t  = mean over W∈[0, W//2)
                        right_t = mean over W∈[W//2, W)
                        asym_t  = left_t - right_t
                      output = asym_{T-1} - asym_0        ("左右质量随时间往哪挪")
    - temporal-diff:  reshape [T,H,W,C]; g_t = mean over all spatial (H,W);
                      output = g_{T-1} - g_0              ("整体内容随时间变化")
    """
    num_patches, C = tokens.shape
    T, H, W = grid

    if pool_mode == "spacetime-mean":
        # Byte-identical to the legacy path: mean over the patch dim.
        return tokens.mean(dim=0)

    # The motion-aware modes need the spatiotemporal grid; assert it lines up
    # with the actual token count before any reshape.
    assert num_patches == T * H * W, (
        f"num_patches ({num_patches}) != T*H*W ({T}*{H}*{W}={T * H * W}); "
        "cannot reshape to the spatiotemporal grid."
    )
    grid_tok = tokens.reshape(T, H, W, C)  # C-order: W fastest, then H, then T

    if pool_mode == "temporal-diff":
        g = grid_tok.mean(dim=(1, 2))      # [T, C] — per time-bin spatial mean
        return g[T - 1] - g[0]             # [C]

    if pool_mode == "lr-motion":
        half = W // 2
        if half == 0:
            raise RuntimeError(
                f"lr-motion needs W>=2 to split left/right; got W={W}."
            )
        left = grid_tok[:, :, :half, :].mean(dim=(1, 2))     # [T, C]
        right = grid_tok[:, :, half:, :].mean(dim=(1, 2))    # [T, C]
        asym = left - right                                   # [T, C]
        return asym[T - 1] - asym[0]                          # [C]

    raise ValueError(f"Unknown pool_mode: {pool_mode!r}")


def extract_features(
    encoder,
    entries: List[Tuple[str, str, str]],
    all_frames: int,
    device: torch.device,
    pool_mode: str,
    grid: Tuple[int, int, int],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Run each video through the encoder once, collecting per-block outputs.

    Uses forward hooks on `encoder.encoder.blocks[i]` (the transformer blocks
    inside PretrainVisionTransformerEncoder) to capture every layer's RAW token
    tensor [num_patches, 768] WITHOUT modifying vendor code. Pooling is deferred
    to `_pool_layer_tokens` (per `pool_mode`) so all three modes share the same
    token-capture path. Each mode outputs 768-d per layer.

    Returns:
        features: float32 [N_ok, n_layers, 768]
        kept_labels: list[str]   labels for the successfully-processed videos
        kept_ids:    list[str]   video ids, aligned with features rows
    """
    # `encoder` is MoPEEncoder; the actual ViT lives at `encoder.encoder`
    # (PretrainVisionTransformerEncoder), whose transformer layers are
    # `encoder.encoder.blocks` (nn.ModuleList of Block, len == 12 for ViT-B).
    vit = encoder.encoder
    blocks = vit.blocks
    n_layers = vit.get_num_layers()
    embed_dim = vit.embed_dim
    logger.info(
        "MoPE encoder: n_layers=%d, embed_dim=%d, moe_layer_indices=%s",
        n_layers, embed_dim, getattr(vit, "moe_layer_indices", None),
    )

    # Per-forward capture buffer; index by layer.
    captured: List[Optional[torch.Tensor]] = [None] * n_layers

    def _make_hook(layer_idx: int):
        def _hook(_module, _inputs, output):
            # Block.forward returns the updated x_vis [B, N_vis, C].
            # Some blocks may return a tuple; take the first tensor element.
            out = output[0] if isinstance(output, (tuple, list)) else output
            # Capture the RAW token tensor [N_vis, C] (B==1 here); pooling is
            # deferred to _pool_layer_tokens so all pool-modes share this path.
            captured[layer_idx] = out.detach().float().squeeze(0).cpu()
        return _hook

    handles = []
    feats: List[np.ndarray] = []
    kept_labels: List[str] = []
    kept_ids: List[str] = []

    try:
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(_make_hook(i)))

        n_total = len(entries)
        for vi, (video_path, label, video_id) in enumerate(entries):
            frames = _sample_mope_frames(video_path, all_frames=all_frames)
            if frames is None:
                continue  # warn already emitted; skip without crashing

            for j in range(n_layers):
                captured[j] = None

            try:
                # Match encoder param dtype (may be bf16/fp16 on GPU).
                ref_dtype = next(encoder.parameters()).dtype
                frames = frames.to(device=device, dtype=ref_dtype)
                with torch.no_grad():
                    _ = encoder(frames)  # triggers hooks; return value unused
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning(
                    "OOM on video %s (idx %d). Skipping this sample.",
                    video_path, vi,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - defensive per-sample
                logger.warning(
                    "Forward failed for %s (%s: %s) — skipping.",
                    video_path, type(exc).__name__, exc,
                )
                continue

            if any(c is None for c in captured):
                missing = [k for k, c in enumerate(captured) if c is None]
                logger.warning(
                    "Hook did not fire for layers %s on %s — skipping.",
                    missing, video_path,
                )
                continue

            # Pool each layer's raw tokens [num_patches, C] → [C] per pool_mode.
            pooled = [_pool_layer_tokens(c, pool_mode, grid) for c in captured]
            layer_stack = torch.stack(pooled, dim=0).numpy()  # [n_layers, 768]
            feats.append(layer_stack.astype(np.float32))
            kept_labels.append(label)
            kept_ids.append(video_id)

            if (vi + 1) % 25 == 0 or (vi + 1) == n_total:
                logger.info("Processed %d/%d videos (%d kept).",
                            vi + 1, n_total, len(feats))

    finally:
        # Always remove hooks, even on exception.
        for h in handles:
            h.remove()

    if not feats:
        raise RuntimeError(
            "No videos were successfully processed — check --video-list paths "
            "and that decord can read the videos."
        )

    features = np.stack(feats, axis=0)  # [N_ok, n_layers, 768]
    return features, kept_labels, kept_ids


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Extract per-layer MoPE encoder features for layer probing (D-13)."
    )
    parser.add_argument(
        "--video-list", required=True,
        help="Path to a jsonl (per-line {video, <label-key>, id}) or txt (one path per line).",
    )
    parser.add_argument(
        "--label-key", default="question_type",
        help="JSON field used as the class label (default: question_type). "
             "Ignored for txt lists (label='unlabeled').",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for features.npy / labels.npy / meta.json.",
    )
    parser.add_argument(
        "--all-frames", type=int, default=8,
        help="Number of frames sampled per video (must match MoPE all_frames; default 8).",
    )
    parser.add_argument(
        "--pool-mode", default="spacetime-mean", choices=list(POOL_MODES),
        help="Per-layer pooling of the raw [num_patches,768] tokens (all → 768-d): "
             "'spacetime-mean' (default, legacy: mean over all patches), "
             "'lr-motion' (left/right asymmetry temporal change, for 左右方向), "
             "'temporal-diff' (global spatial-mean temporal change, for 逼近/远离/运动有无). "
             "# PENDING[D-13]: 运动感知模式，取层结论待 probe 后定。",
    )
    parser.add_argument(
        "--ckpt-path", default=os.environ.get("MOPE_CKPT_PATH"),
        help="MoPE checkpoint path (default: $MOPE_CKPT_PATH). "
             "If unset, architecture-only (random weights) — for pipeline smoke test only.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available). Feature extraction needs GPU.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional: process only the first N videos (debug).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        logger.warning(
            "Running on %s — MoPE forward is slow/unsupported off-GPU. "
            "This step is intended to run on GPU.", device,
        )
    if not args.ckpt_path:
        logger.warning(
            "No --ckpt-path / MOPE_CKPT_PATH set — using RANDOM encoder weights. "
            "Features are meaningless; for pipeline smoke test only."
        )

    entries = _parse_video_list(args.video_list, args.label_key)
    if args.limit is not None:
        entries = entries[: args.limit]
    logger.info("Parsed %d video entries from %s.", len(entries), args.video_list)
    if not entries:
        raise RuntimeError("Video list parsed to 0 entries.")

    encoder = load_mope_encoder(args.ckpt_path, args.all_frames, device)

    # Infer the (T, H, W) token grid from the encoder (not hardcoded).
    num_patches = int(encoder.encoder.patch_embed.num_patches)
    T, H, W = infer_token_grid(encoder, args.all_frames, num_patches)
    logger.info(
        "Token grid inferred: T=%d, H=%d, W=%d (num_patches=%d); pool_mode=%s",
        T, H, W, num_patches, args.pool_mode,
    )

    features, kept_labels, kept_ids = extract_features(
        encoder, entries, args.all_frames, device, args.pool_mode, (T, H, W)
    )

    # Encode string labels → int ids (stable, sorted for reproducibility).
    label_names = sorted(set(kept_labels))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_to_id[l] for l in kept_labels], dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "labels.npy", labels)

    vit = encoder.encoder
    _pool_notes = {
        "spacetime-mean": "features[:, i, :] = 第 i 个 block 输出对全部 784 patch 的 "
                          "mean-pool（旧行为，对空间+时间同时平均）.",
        "lr-motion": "features[:, i, :] = 第 i 个 block 的左右不对称时间变化 "
                     "asym_{T-1}-asym_0，asym_t=left_t-right_t（W 为最快变维，"
                     "left=W∈[0,W//2)、right=W∈[W//2,W) 的空间均值）；捕捉左右运动方向.",
        "temporal-diff": "features[:, i, :] = 第 i 个 block 的整体内容时间变化 "
                         "g_{T-1}-g_0，g_t=每时间块全空间均值；捕捉逼近/远离/运动有无.",
    }
    meta = {
        "n_videos": int(features.shape[0]),
        "n_layers": int(features.shape[1]),
        "embed_dim": int(features.shape[2]),
        "feat_dim": int(features.shape[2]),
        "pool_mode": args.pool_mode,
        "token_grid": {"T": int(T), "H": int(H), "W": int(W)},
        "video_ids": kept_ids,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "label_key": args.label_key,
        "all_frames": args.all_frames,
        "ckpt_path": args.ckpt_path,
        "moe_layer_indices": list(getattr(vit, "moe_layer_indices", []) or []),
        "feature_pooling": args.pool_mode,
        # PENDING[D-13]: 取层结论待 probe 后定；note 按 pool_mode 说明该模式语义。
        "note": "服务于 D-13 取层决策（运动感知特征，2026-06-07 重设计）。"
                + _pool_notes.get(args.pool_mode, ""),
    }
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved features %s + labels %s + meta.json to %s",
        features.shape, labels.shape, out_dir,
    )
    logger.info(
        "Label distribution: %s",
        {name: int((labels == i).sum()) for name, i in label_to_id.items()},
    )


if __name__ == "__main__":
    main()
