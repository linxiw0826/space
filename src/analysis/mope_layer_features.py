"""mope_layer_features.py — extract per-layer MoPE encoder features (验证③ / D-13).

服务于 D-13 取层决策（MoPE 取层 + 融合算子）。
================================================================================
背景
----
MoPE encoder (`src/model/mope_encoder.py`) 当前只返回 transformer 末层特征
(`modeling_pretrain.py:135-141` 的 `self.norm(x_vis)`)。本脚本用 **forward hook**
挂在 `MoPEEncoder.encoder.blocks[i]`（每个 transformer block）上，收集**每一层**的
block 输出，对 patch 维做 mean-pool，得到每视频每层一个 [768] 向量。配套的
`mope_probe_layers.py`（纯 CPU）再对每层训线性探针，验证 FAIR《Interpreting
Physics》(2602.07050) 结论是否在 MoPE 上成立——物理/方向信息是否在中层峰值、
向输出层退化。若是，则 MoPE 应改抽中层（D-13）。

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

输出（写到 --out-dir）
----------------------
    features.npy : float32 [N_videos, n_layers, 768]   每视频每层 mean-pool 向量
    labels.npy   : int64   [N_videos]                  标签的整数编码
    meta.json    : {n_layers, embed_dim, n_videos, video_ids, label_names,
                    label_to_id, moe_layer_indices, all_frames, ...}

用法
----
    export MOPE_CKPT_PATH=/abs/path/to/mope_checkpoint.pth   # E-02a MoPE ckpt
    export PYTHONPATH=/u/lwu9/Space_sensing/projects/space:$PYTHONPATH  # so `src` is importable
    python -m src.analysis.mope_layer_features \\
        --video-list /abs/path/vlm4d.jsonl \\
        --label-key question_type \\
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

# PENDING[D-13]: 当前抽 ALL 12 个 block 的输出（每层一个 mean-pool 向量）以供
# probe 定位峰值层。D-13 取层决策解除后，下游 MoPEEncoder 只需保留所选层；本脚本
# 是为该决策提供证据，无需改动（继续抽全层即可对照）。
PROBE_ALL_BLOCKS = True


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


def extract_features(
    encoder,
    entries: List[Tuple[str, str, str]],
    all_frames: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Run each video through the encoder once, collecting per-block outputs.

    Uses forward hooks on `encoder.encoder.blocks[i]` (the transformer blocks
    inside PretrainVisionTransformerEncoder) to capture every layer's output
    WITHOUT modifying vendor code. Each captured tensor is [B, N_vis, 768];
    we mean-pool over the patch dim → [768] per layer.

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
            # mean-pool over patch dim (dim=1) → [B, C]; B==1 here.
            captured[layer_idx] = out.detach().float().mean(dim=1).squeeze(0).cpu()
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

            layer_stack = torch.stack(captured, dim=0).numpy()  # [n_layers, 768]
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

    features, kept_labels, kept_ids = extract_features(
        encoder, entries, args.all_frames, device
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
    meta = {
        "n_videos": int(features.shape[0]),
        "n_layers": int(features.shape[1]),
        "embed_dim": int(features.shape[2]),
        "video_ids": kept_ids,
        "label_names": label_names,
        "label_to_id": label_to_id,
        "label_key": args.label_key,
        "all_frames": args.all_frames,
        "ckpt_path": args.ckpt_path,
        "moe_layer_indices": list(getattr(vit, "moe_layer_indices", []) or []),
        "feature_pooling": "mean over patch dim (per block output)",
        "note": "服务于 D-13 取层决策；features[:, i, :] 是第 i 个 transformer "
                "block 输出的 patch-mean (i=0..n_layers-1, 末层最后一个).",
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
