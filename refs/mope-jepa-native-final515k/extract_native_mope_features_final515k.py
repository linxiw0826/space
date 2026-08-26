#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.native_mope import native_mope_jepa_base_patch16_224

DEFAULT_CKPT = "/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-50.pth"


def sample_indices_4x4(total_frames, groups=4, frames_per_group=4):
    if total_frames <= 0:
        raise ValueError("video has no decodable frames")
    indices = []
    for group_id in range(groups):
        start = int(np.floor(group_id * total_frames / groups))
        end = int(np.floor((group_id + 1) * total_frames / groups))
        end = max(end, start + 1)
        local = np.linspace(start, end - 1, frames_per_group)
        indices.extend(np.rint(local).astype(np.int64).tolist())
    indices = [int(np.clip(i, 0, total_frames - 1)) for i in indices]
    target = groups * frames_per_group
    while len(indices) < target:
        indices.append(indices[-1])
    return indices[:target]


def read_video_opencv(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def load_4x4_frames(video_path, input_size=224, groups=4, frames_per_group=4):
    frames = read_video_opencv(video_path)
    indices = sample_indices_4x4(len(frames), groups=groups, frames_per_group=frames_per_group)
    selected = [frames[i] for i in indices]
    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    tensor = torch.stack([transform(frame) for frame in selected], dim=1).unsqueeze(0)
    return tensor, indices, len(frames)


def build_model():
    model = native_mope_jepa_base_patch16_224(
        pretrained=False,
        all_frames=16,
        tubelet_size=2,
        encoder_depth=12,
        dense_layers=8,
        num_experts=8,
        top_k=1,
        num_shared_experts=1,
        router_score_func="sigmoid",
        router_bias_update_speed=0.0,
        num_anchors=1,
        pos_embed_type="3d_sincos",
        predictor_pos_embed_type="3d_sincos",
    )
    return model


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("[load] missing sample:", missing[:8], flush=True)
    if unexpected:
        print("[load] unexpected sample:", unexpected[:8], flush=True)


def iter_videos(args):
    if args.video:
        yield Path(args.video)
    if args.video_dir:
        root = Path(args.video_dir)
        for suffix in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            yield from sorted(root.rglob(suffix))


def save_feature(out_path, feature, meta, save_npy=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"feature": feature.cpu(), "meta": meta}
    torch.save(payload, out_path)
    if save_npy:
        np.save(out_path.with_suffix(".npy"), feature.float().cpu().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--video")
    parser.add_argument("--video-dir")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pool", choices=["temporal", "mean", "none"], default="temporal")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--frames-per-group", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    args = parser.parse_args()

    if not args.video and not args.video_dir:
        raise SystemExit("please provide --video or --video-dir")
    if args.groups * args.frames_per_group != 16:
        raise SystemExit("final515k MoPE expects exactly 16 frames; use --groups 4 --frames-per-group 4")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model()
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()

    save_dir = Path(args.save_dir)
    videos = list(iter_videos(args))
    print(f"[run] videos={len(videos)} pool={args.pool} ckpt={args.ckpt}", flush=True)

    with torch.no_grad():
        for idx, video_path in enumerate(videos, 1):
            out_path = save_dir / f"{video_path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                continue
            video, indices, decoded_frames = load_4x4_frames(
                video_path,
                input_size=args.input_size,
                groups=args.groups,
                frames_per_group=args.frames_per_group,
            )
            raw = model.encode(video.to(device, non_blocking=True), token_mask=None, record_stats=False).detach().float()
            if args.pool == "mean":
                feature = raw.mean(dim=1, keepdim=True).squeeze(0)
            elif args.pool == "temporal":
                bsz, tokens, dim = raw.shape
                t_bins = 16 // 2
                spatial = tokens // t_bins
                feature = raw.view(bsz, t_bins, spatial, dim).mean(dim=2).squeeze(0)
            else:
                feature = raw.squeeze(0)
            meta = {
                "video": str(video_path),
                "ckpt": args.ckpt,
                "sampling": "4x4_uniform_segments",
                "num_input_frames": 16,
                "decoded_frames": decoded_frames,
                "indices": indices,
                "input_size": args.input_size,
                "pool": args.pool,
                "feature_shape": list(feature.shape),
            }
            save_feature(out_path, feature, meta, save_npy=args.save_npy)
            print(f"[{idx}/{len(videos)}] {video_path.name} -> {out_path} shape={tuple(feature.shape)}", flush=True)

    summary = {
        "model_code": "/data2/mope-jepa-native-final515k",
        "ckpt": args.ckpt,
        "sampling": "16 frames = 4 temporal groups x 4 frames/group",
        "default_output": "pool=temporal gives 8 x 768 MoPE tokens",
    }
    (save_dir / "_README_feature_extraction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
