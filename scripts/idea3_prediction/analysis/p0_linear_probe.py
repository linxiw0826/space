#!/usr/bin/env python
# =============================================================================
# P-0 — linear-probe 机制诊断 (paper2_design §4.1, decisions.md D-20)
# =============================================================================
# 目的（"位置擦除 vs 负迁移"定案）
# --------------------------------------------------------------------------
# E-16 (feed+predict) 在 VSI-Bench 的**绝对时序顺序题**（appearance_order、route）
# 掉分，但让 VLM4D（瞬时运动）涨。有两个竞争假设：
#   ① 位置擦除：predict-next-bin 辅助头（LFP）抹掉了 LLM hidden 里的绝对时间位置。
#   ② 负迁移  ：只是普通的多任务负迁移，绝对时间位置信息并未被抹掉。
# 本脚本用 linear-probe 区分二者：测量"从 LLM 每帧 hidden 里线性还原该帧的
# 绝对时间位置（bin-index / 归一化帧位置）"的可还原度，对比不同 checkpoint。
#
# ★ 解读规则（读结果 JSON 时照此判断）★
#   干净判据 = E-16 vs E-03a（二者都 MoPE-ON，唯一变量 = 辅助头 LFP）：
#     - E-16 的 bin_acc / pos_r2 **显著 <** E-03a（且 E-16 的 bin_acc 接近
#       chance_acc=0.25）→ 支持"位置擦除" → 走 E-16c（目标端补绝对时间位置编码 /
#       bin-index 回归）。
#     - E-16 与 E-03a **相近**（辅助头没有明显压低可还原度）→ 支持"负迁移"
#       → 走调 loss 权重 / E-16d 顺序头。
#   ⚠ E-15 vs E-03a 只作 WARN-1（无泄漏臂）旁证，**不能**并入干净判据：
#     E-15 = MoPE-OFF（feed 关、只留 predict，design §4.1 WARN-1），相对 E-03a
#     同时差"辅助头"和"MoPE feed 注入"两项 confound；其低可还原度可能部分来自
#     MoPE-off 而非辅助头。故"位置擦除"的定案以 E-16 vs E-03a 为准，E-15 仅参考。
#
# --------------------------------------------------------------------------
# 方法（忠实复用训练期 LFP 头的切帧 + 分 bin 逻辑）
# --------------------------------------------------------------------------
# 这不是训练脚本：纯 forward（inference，不生成、不反传）+ sklearn 线性拟合。
#   ① 复用 eval 的加载路径实例化 model+checkpoint：
#        - qwen3_vl_mope_crossattn  → eval.models.qwen3_vl_mope.Qwen3_VL_MoPE_CrossAttn
#          （E-03a / E-16；MoPE-ON：cross-attn 残差注入 + sidecar 注入 mope_frames）
#        - qwen3_vl_my              → lmms_eval.models.simple.qwen3_vl_my.Qwen3_VL_MY
#          （E-15 / E-01；MoPE-OFF，纯 GUIDE 基座）
#     推理丢头：eval wrapper 只应用 cross-attn feed patch，不应用 LFP 辅助头
#     （与 VSI-Bench 真实评测路径 byte 对齐）。
#   ② 逐视频只前向：把 --frames 帧作为 IMAGE token 喂给 LLM（= SPAR/VSI-590K 训练
#     时的多图排布），拿 LLM 最后一层 hidden（inner model 的 outputs[0] =
#     last_hidden_state，正是训练期 LFP 头读的那批向量），用 mope_patch 的
#     `_extract_per_frame_video_hidden`（按 image/video_token_id 连续段切帧、每帧
#     mean-pool）抽出每帧 [F, D]。
#   ③ 两个标签：
#        (a) bin-index 4 类：用 mope_patch `_group_llm_frames_to_bins` 的**同一**
#            均匀分段（bin b = LLM 帧 [b*F//n_bins, (b+1)*F//n_bins)）把帧 → bin。
#        (b) 归一化帧位置 i/(F-1) 回归。
#   ④ 标准化 X；**按视频切 train/test**（同一视频的帧不跨 train/test，防泄漏）70/30。
#   ⑤ LogisticRegression(multinomial) 拟 bin-index → test accuracy + macro-F1；
#      LinearRegression 拟位置 → test R²；chance 基线 = 1/n_bins = 0.25。
#   ⑥ 输出 JSON：{checkpoint, model_type, n_videos, n_frames_total, bin_acc,
#      bin_macro_f1, pos_r2, chance_acc, frames_per_video, ...}。
#
# --------------------------------------------------------------------------
# 与训练期 LFP 头的一致性（为什么探针读的向量正确）
# --------------------------------------------------------------------------
#   - hidden 来源：调 **inner model** (`model._model.model`) 前向，取
#     `outputs.last_hidden_state`（= inner `outputs[0]`）。训练期 LFP 头在
#     mope_patch.py:1166 做的正是 `hidden_states = outputs[0]`（同一 inner 前向、
#     同一后置 norm 的末层 hidden）。
#   - 切帧 + 分 bin：直接 import mope_patch 的 `_extract_per_frame_video_hidden`
#     与 `_group_llm_frames_to_bins`（不复制、不改），保证与 LFP 头字节级同逻辑。
#   - token id：按 (image_token_id, video_token_id) 两个 id 切帧（训练多图排布用
#     image token，eval 可能是 video token），与 LFP 头一致（mope_patch:1264）。
#
# --------------------------------------------------------------------------
# GPU / 复现 / 依赖
# --------------------------------------------------------------------------
#   - 轻量 inference（batch=1、小视频子集），但仍需一张 GPU 跑 forward。用 --device
#     指定。**若训练占满 cards 0-3，请用别的空闲卡（如 CUDA_VISIBLE_DEVICES=4
#     + --device cuda:0）或等训练完再跑。**
#   - 固定 --seed；视频子集/切分确定性采样，结果可复现。
#   - 依赖只用项目已装：torch / numpy / sklearn / decord。若 sklearn 缺失，脚本
#     启动即报错退出（不自动 pip）。
#
# 用法示例：
#   python p0_linear_probe.py \
#       --checkpoint /path/to/e16_lfp_both_4b \
#       --model_type qwen3_vl_mope_crossattn \
#       --video_dir /data2/wlx/data/VSIBench --num_videos 150 \
#       --frames 8 --device cuda:0 --seed 0 \
#       --output /path/to/p0_e16.json
# =============================================================================
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("p0_linear_probe")

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# ---------------------------------------------------------------------------
# sys.path setup — make lmms_eval / qwenvl / model / eval importable exactly
# like the eval scripts (GUIDE_LMMS_EVAL + src on PYTHONPATH). We accept a
# --space_root so the script runs standalone even if PYTHONPATH is unset; the
# runner shell still exports the same PYTHONPATH as eval_e16_vsibench.sh.
# ---------------------------------------------------------------------------
def _setup_sys_path(space_root: str) -> None:
    space_root = str(Path(space_root).resolve())
    src_root = str(Path(space_root) / "src")
    lmms_eval_root = str(Path(space_root) / "src" / "vendor" / "lmms-eval")
    for p in (lmms_eval_root, src_root, space_root):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Model loading — reuse the eval wrappers verbatim (same __init__ that
# VSI-Bench eval uses). No LFP head is applied by these wrappers (training
# only), so E-16 / E-15 are evaluated exactly as at inference (head dropped).
# ---------------------------------------------------------------------------
def load_model(model_type: str, checkpoint: str, device: str,
               attn_implementation: str, max_pixels: int, min_pixels: int,
               mope_all_frames: int):
    if model_type == "qwen3_vl_mope_crossattn":
        from eval.models.qwen3_vl_mope import Qwen3_VL_MoPE_CrossAttn
        wrapper = Qwen3_VL_MoPE_CrossAttn(
            pretrained=checkpoint,
            device=device,
            device_map=device,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            mope_all_frames=mope_all_frames,
        )
        is_mope = True
    elif model_type == "qwen3_vl_my":
        from lmms_eval.models.simple.qwen3_vl_my import Qwen3_VL_MY
        wrapper = Qwen3_VL_MY(
            pretrained=checkpoint,
            device=device,
            device_map=device,
            attn_implementation=attn_implementation,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        is_mope = False
    else:
        raise ValueError(
            f"--model_type must be one of "
            f"{{qwen3_vl_mope_crossattn, qwen3_vl_my}}, got '{model_type}'"
        )
    wrapper.model.eval()
    return wrapper, is_mope


# ---------------------------------------------------------------------------
# Video list resolution (deterministic).
# ---------------------------------------------------------------------------
def resolve_video_list(args, rng: np.random.Generator) -> List[str]:
    if args.video_list:
        path = args.video_list
        if path.endswith(".json"):
            with open(path, "r") as fh:
                vids = json.load(fh)
            if isinstance(vids, dict):
                vids = list(vids.values())
        else:  # txt: one path per line
            with open(path, "r") as fh:
                vids = [ln.strip() for ln in fh if ln.strip()]
        vids = [v for v in vids if isinstance(v, str)]
        logger.info("Loaded %d videos from --video_list %s", len(vids), path)
    elif args.video_dir:
        vids = []
        for ext in _VIDEO_EXTS:
            vids.extend(glob.glob(os.path.join(args.video_dir, "**", f"*{ext}"),
                                  recursive=True))
        vids = sorted(vids)  # deterministic order before sampling
        logger.info("Found %d videos under --video_dir %s", len(vids), args.video_dir)
        if args.num_videos and len(vids) > args.num_videos:
            idx = rng.choice(len(vids), size=args.num_videos, replace=False)
            idx.sort()
            vids = [vids[i] for i in idx]
            logger.info("Deterministically sampled %d videos (seed=%d)",
                        len(vids), args.seed)
    else:
        raise ValueError("Provide either --video_list or --video_dir")
    # keep only existing files
    vids = [v for v in vids if os.path.exists(v)]
    if not vids:
        raise RuntimeError("No existing video files resolved.")
    return vids


# ---------------------------------------------------------------------------
# Frame sampling — F PIL frames, uniform over the clip (native resolution;
# the processor handles min/max-pixel resizing, mirroring generate_until).
# ---------------------------------------------------------------------------
def sample_frames(video_path: str, num_frames: int):
    import decord
    from PIL import Image

    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:  # repeat last frame to fill
        indices = list(range(total)) + [total - 1] * (num_frames - total)
        indices = np.array(indices, dtype=int)
    frames = [Image.fromarray(vr[int(i)].asnumpy()).convert("RGB") for i in indices]
    return frames


# ---------------------------------------------------------------------------
# Build the LLM forward inputs from F PIL frames + a text prompt.
# Faithfully mirrors Qwen3_VL_MY.generate_until (qwen3_vl_my.py:259-443):
# message construction -> geometry_encoder_inputs -> apply_chat_template ->
# process_vision_info -> processor -> device move -> geometry resize.
# The only intentional difference: we feed exactly `--frames` IMAGE frames
# (training-consistent) instead of the eval-time max_num_frames=32 subsample.
# ---------------------------------------------------------------------------
def build_inputs(wrapper, pil_frames, prompt: str):
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers.image_utils import SizeDict
    from torchvision import transforms as TF

    processor = wrapper.processor

    message = [{"role": "system", "content": wrapper.system_prompt}]
    processed_visuals = []
    for img in pil_frames:
        processed_visuals.append({
            "type": "image", "image": img,
            "max_pixels": wrapper.max_pixels, "min_pixels": wrapper.min_pixels,
        })
    message.append({"role": "user",
                    "content": processed_visuals + [{"type": "text", "text": prompt}]})
    batched_messages = [message]

    # --- geometry_encoder_inputs (GUIDE geometry injection) --------------
    use_geometry_inputs = getattr(wrapper.model.config, "use_geometry_encoder", False)
    use_patch_size_alin = getattr(wrapper.model.config, "use_patch_size_alin", False)
    geometry_encoder_inputs = None
    if use_geometry_inputs:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        geometry_encoder_inputs = []
        size_cfg = getattr(processor.image_processor, "size", {})
        min_pixels = size_cfg.get("shortest_edge", 56 * 56)
        max_pixels = size_cfg.get("longest_edge", 28 * 28 * 1280)
        for msg_list in batched_messages:
            per_sample = []
            for msg in msg_list:
                if msg.get("role") != "user":
                    continue
                for content in msg.get("content", []):
                    if not (isinstance(content, dict) and content.get("type") == "image"):
                        continue
                    img_obj = content.get("image")
                    if img_obj is None:
                        continue
                    pil_img = img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj
                    img_tensor = TF.ToTensor()(pil_img)
                    if (not use_patch_size_alin and isinstance(img_tensor, torch.Tensor)
                            and img_tensor.dim() == 3):
                        _, height, width = img_tensor.shape
                        rh, rw = smart_resize(height, width, factor=14,
                                              min_pixels=min_pixels, max_pixels=max_pixels)
                        if (rh, rw) != (height, width):
                            img_tensor = processor.image_processor.resize(
                                image=img_tensor, size=SizeDict(height=rh, width=rw),
                                interpolation=processor.image_processor.resample,
                            )
                    per_sample.append(img_tensor)
            if per_sample:
                shapes = [t.shape for t in per_sample if isinstance(t, torch.Tensor)]
                if shapes and all(s == shapes[0] for s in shapes):
                    per_sample = torch.stack(per_sample)
            geometry_encoder_inputs.append(per_sample)

    texts = processor.apply_chat_template(batched_messages, tokenize=False,
                                          add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(
        batched_messages, return_video_kwargs=False, image_patch_size=16,
        return_video_metadata=False,
    )
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs,
                       do_resize=False, return_tensors="pt")

    # patch-size-aligned geometry resize (qwen3_vl_my.py:388-426)
    if use_geometry_inputs and use_patch_size_alin and geometry_encoder_inputs is not None:
        grid_thw = inputs.get("image_grid_thw")
        grid_thw_list = grid_thw.tolist() if (isinstance(grid_thw, torch.Tensor)
                                              and grid_thw.dim() == 2) else None
        if grid_thw_list:
            grid_idx = 0
            resized_geometry_inputs = []
            for per_sample in geometry_encoder_inputs:
                per_sample_list = ([img for img in per_sample]
                                   if isinstance(per_sample, torch.Tensor) else per_sample)
                resized_sample = []
                for img in per_sample_list:
                    if not isinstance(img, torch.Tensor) or img.dim() != 3:
                        resized_sample.append(img)
                        grid_idx += 1
                        continue
                    if grid_idx >= len(grid_thw_list):
                        resized_sample.append(img)
                        continue
                    _, grid_h, grid_w = grid_thw_list[grid_idx]
                    target_h, target_w = int(grid_h) * 14, int(grid_w) * 14
                    _, height, width = img.shape
                    if (target_h, target_w) != (height, width):
                        img = processor.image_processor.resize(
                            image=img, size=SizeDict(height=target_h, width=target_w),
                            interpolation=processor.image_processor.resample,
                        )
                    resized_sample.append(img)
                    grid_idx += 1
                if isinstance(per_sample, torch.Tensor):
                    resized_sample = torch.stack(resized_sample) if resized_sample else per_sample
                resized_geometry_inputs.append(resized_sample)
            geometry_encoder_inputs = resized_geometry_inputs

    device = wrapper.device
    inputs = inputs.to(device)
    if use_geometry_inputs and geometry_encoder_inputs is not None:
        moved = []
        for ge in geometry_encoder_inputs:
            if isinstance(ge, torch.Tensor):
                moved.append(ge.to(device))
            elif isinstance(ge, list):
                moved.append([g.to(device) if isinstance(g, torch.Tensor) else g for g in ge])
            else:
                moved.append(ge)
        inputs["geometry_encoder_inputs"] = moved
    return inputs


# ---------------------------------------------------------------------------
# Faithful bin labels: same uniform segmentation as
# mope_patch._group_llm_frames_to_bins (frame f -> bin b iff
# b*F//n_bins <= f < (b+1)*F//n_bins). Returns an int array length F.
# ---------------------------------------------------------------------------
def frame_bin_labels(num_frames: int, n_bins: int) -> np.ndarray:
    labels = np.full(num_frames, -1, dtype=int)
    for b in range(n_bins):
        s = (b * num_frames) // n_bins
        e = ((b + 1) * num_frames) // n_bins
        if e <= s:
            e = s + 1
        e = min(e, num_frames)
        labels[s:e] = b
    # defensive: any frame left unassigned (F < n_bins edge) -> last bin
    labels[labels < 0] = n_bins - 1
    return labels


# ---------------------------------------------------------------------------
# Per-video forward -> per-frame hidden [Fp, D] (float32) + labels.
# ---------------------------------------------------------------------------
def forward_video(wrapper, is_mope, video_path, frames, prompt,
                  extract_fn, image_token_id, video_token_id, n_bins):
    import torch

    pil_frames = sample_frames(video_path, frames)
    inputs = build_inputs(wrapper, pil_frames, prompt)

    inner = wrapper.model.model  # Qwen3VLModel (crossattn-patched for MoPE models)

    # MoPE sidecar: inject mope_frames exactly as generate_until does, so the
    # cross-attn residual fusion runs and the hidden reflects the MoPE-ON path.
    if is_mope:
        mope_frames = wrapper._compute_mope_frames([video_path])
        if mope_frames is not None:
            tdev = next(inner.parameters()).device
            tdt = next(inner.parameters()).dtype
            inner._pending_mope_frames = mope_frames.to(device=tdev, dtype=tdt)

    try:
        call_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            use_cache=False,
        )
        if "geometry_encoder_inputs" in inputs:
            call_kwargs["geometry_encoder_inputs"] = inputs["geometry_encoder_inputs"]
        with torch.inference_mode():
            outputs = inner(**call_kwargs)
        # inner returns Qwen3VLModelOutputWithPast; [0] == last_hidden_state,
        # identical to what the LFP head reads (mope_patch.py:1166).
        hidden_states = outputs[0]  # [1, L, D]
        per_frame = extract_fn(hidden_states, inputs["input_ids"],
                               (image_token_id, video_token_id))
    finally:
        if is_mope:
            inner._pending_mope_frames = None

    if per_frame is None:
        return None, None, None
    per_frame = per_frame[0].float().cpu().numpy()  # [Fp, D]
    Fp = per_frame.shape[0]
    bin_lab = frame_bin_labels(Fp, n_bins)
    pos_lab = (np.arange(Fp, dtype=np.float64) / max(Fp - 1, 1))
    return per_frame, bin_lab, pos_lab


# ---------------------------------------------------------------------------
# Linear probes: group (video)-wise ~70/30 split -> LogisticRegression(bin) +
# LinearRegression(pos). StandardScaler fit on TRAIN only (no leakage).
#
# W1 fix: each video's train/test side is a PURE FUNCTION of (seed, its stable
# group key) — a per-group hash threshold, NOT a permutation over the observed
# set. So a given video lands on the SAME side regardless of which OTHER videos
# a particular checkpoint happened to skip; two checkpoints therefore agree on
# every shared video's side (fair "same test videos" comparison), and it stays
# deterministic + seed-controllable. ~30% go to test via modulo-1000 threshold.
# ---------------------------------------------------------------------------
def _video_is_test(group_key: int, seed: int, test_frac: float = 0.30) -> bool:
    h = int(hashlib.md5(f"{seed}:{int(group_key)}".encode("utf-8")).hexdigest()[:12], 16)
    return (h % 1000) < int(round(test_frac * 1000))


def run_probes(X, y_bin, y_pos, groups, n_bins, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, r2_score

    uniq = np.unique(groups)
    test_vids = {int(g) for g in uniq if _video_is_test(int(g), seed)}
    test_mask = np.array([int(g) in test_vids for g in groups])
    train_mask = ~test_mask
    n_train_vids = len(uniq) - len(test_vids)
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        raise RuntimeError(
            "Degenerate train/test split (all videos fell on one side). "
            "Need >=2 videos; try a different --seed or more videos."
        )

    scaler = StandardScaler().fit(X[train_mask])
    Xtr, Xte = scaler.transform(X[train_mask]), scaler.transform(X[test_mask])

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, y_bin[train_mask])
    pred = clf.predict(Xte)
    bin_acc = float(accuracy_score(y_bin[test_mask], pred))
    bin_macro_f1 = float(f1_score(y_bin[test_mask], pred, average="macro"))

    reg = LinearRegression()
    reg.fit(Xtr, y_pos[train_mask])
    pos_r2 = float(r2_score(y_pos[test_mask], reg.predict(Xte)))

    return {
        "bin_acc": bin_acc,
        "bin_macro_f1": bin_macro_f1,
        "pos_r2": pos_r2,
        "chance_acc": 1.0 / n_bins,
        "n_train_videos": int(n_train_vids),
        "n_test_videos": int(len(test_vids)),
        "n_train_frames": int(train_mask.sum()),
        "n_test_frames": int(test_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="P-0 linear-probe: absolute time-bin recoverability from LLM per-frame hidden.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir (pretrained=)")
    ap.add_argument("--model_type", required=True,
                    choices=["qwen3_vl_mope_crossattn", "qwen3_vl_my"],
                    help="qwen3_vl_mope_crossattn (E-03a/E-16) | qwen3_vl_my (E-15/E-01)")
    ap.add_argument("--video_list", default=None, help="json (list/dict) or txt (one path/line) of video paths")
    ap.add_argument("--video_dir", default=None, help="dir to glob **/*.mp4 (with --num_videos)")
    ap.add_argument("--num_videos", type=int, default=150, help="deterministic sample size from --video_dir")
    ap.add_argument("--frames", type=int, default=8, help="LLM frames per video (train-consistent; default 8)")
    ap.add_argument("--mope_all_frames", type=int, default=8,
                    help="MoPE encoder frames (frozen ckpt = 8; keep 8 to match 4 time-bins)")
    ap.add_argument("--n_bins", type=int, default=4, help="time bins (MoPE tubelet-2 / 8 frames = 4)")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--device", default="cuda:0", help="e.g. cuda:0 (see GPU note in docstring)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--space_root", default=None,
                    help="project root (space/); default = infer from this file's location")
    ap.add_argument("--attn_implementation", default="flash_attention_2",
                    help="match eval (flash_attention_2); use sdpa if flash-attn unavailable")
    ap.add_argument("--max_pixels", type=int, default=268324, help="match eval scripts")
    ap.add_argument("--min_pixels", type=int, default=8192, help="match eval scripts")
    ap.add_argument("--prompt", default="These are frames of a video. Describe the spatial layout and the order of events in the video.",
                    help="fixed neutral prompt (deterministic); probe is robust to prompt")
    args = ap.parse_args()

    # sklearn availability check (do not auto-pip)
    try:
        import sklearn  # noqa: F401
    except ImportError:
        logger.error("scikit-learn is NOT installed. Install it (project dep) and re-run; "
                     "this script will not pip anything automatically.")
        sys.exit(2)

    space_root = args.space_root or str(Path(__file__).resolve().parents[3])
    _setup_sys_path(space_root)

    rng = np.random.default_rng(args.seed)
    videos = resolve_video_list(args, rng)

    logger.info("Loading model_type=%s checkpoint=%s device=%s",
                args.model_type, args.checkpoint, args.device)
    wrapper, is_mope = load_model(
        args.model_type, args.checkpoint, args.device,
        args.attn_implementation, args.max_pixels, args.min_pixels, args.mope_all_frames,
    )

    from model.mope_patch import _extract_per_frame_video_hidden  # noqa: E402
    cfg = wrapper.model.config
    image_token_id = getattr(cfg, "image_token_id", None)
    video_token_id = getattr(cfg, "video_token_id", None)
    logger.info("image_token_id=%s video_token_id=%s", image_token_id, video_token_id)

    # W1 fix: the group key MUST be a STABLE per-video identity independent of
    # success/failure order, so that any two checkpoints run on the same video
    # set get the SAME 70/30 video split ("same test videos" fair comparison).
    # Using a running success counter (n_ok) would shift group ids whenever a
    # checkpoint skips a different video. We instead derive a deterministic
    # int key from a stable md5 hash of the video path.
    videos = sorted(videos)  # stable canonical order

    def _stable_group_key(path: str) -> int:
        return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:12], 16)

    X_list, bin_list, pos_list, group_list = [], [], [], []
    n_ok = 0
    for vi, vpath in enumerate(videos):
        try:
            per_frame, bin_lab, pos_lab = forward_video(
                wrapper, is_mope, vpath, args.frames, args.prompt,
                _extract_per_frame_video_hidden, image_token_id, video_token_id, args.n_bins,
            )
            if per_frame is None:
                logger.warning("[%d/%d] no per-frame hidden extracted, skipping: %s",
                               vi + 1, len(videos), vpath)
                continue
            X_list.append(per_frame)
            bin_list.append(bin_lab)
            pos_list.append(pos_lab)
            gkey = _stable_group_key(vpath)
            group_list.append(np.full(per_frame.shape[0], gkey, dtype=np.int64))
            n_ok += 1
            if (vi + 1) % 20 == 0:
                logger.info("[%d/%d] processed (%d ok, %d frames so far)",
                            vi + 1, len(videos), n_ok, sum(x.shape[0] for x in X_list))
        except Exception as exc:  # skip corrupt/failed samples, keep going
            logger.warning("[%d/%d] forward failed (%s: %s), skipping: %s",
                           vi + 1, len(videos), type(exc).__name__, exc, vpath)
            continue

    if n_ok < 2:
        logger.error("Only %d videos succeeded (<2). Cannot run a grouped split.", n_ok)
        sys.exit(3)

    X = np.vstack(X_list)
    y_bin = np.concatenate(bin_list)
    y_pos = np.concatenate(pos_list)
    groups = np.concatenate(group_list)
    logger.info("Aggregated X=%s (n_videos=%d, n_frames=%d, D=%d)",
                X.shape, n_ok, X.shape[0], X.shape[1])

    probe = run_probes(X, y_bin, y_pos, groups, args.n_bins, args.seed)

    result = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "n_videos": int(n_ok),
        "n_frames_total": int(X.shape[0]),
        "hidden_dim": int(X.shape[1]),
        "frames_per_video": int(args.frames),
        "n_bins": int(args.n_bins),
        "seed": int(args.seed),
        "is_mope_on": bool(is_mope),
        **probe,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Wrote %s", args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
