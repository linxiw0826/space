"""Extract Native-MoPE8 3DPos video representations from this native repo.

Example:
CUDA_VISIBLE_DEVICES=0 python extract_native_mope_features.py \
  --video /path/to/video.mp4 \
  --save-dir /tmp/mope_features \
  --pool temporal
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.loader import get_video_loader
from models.native_mope import native_mope_jepa_base_patch16_224


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def default_assets_root():
    env = os.environ.get("MOPE_ASSETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if str(REPO_ROOT).startswith("/data2/"):
        return Path("/data2/mope-jepa-assets")
    if str(REPO_ROOT).startswith("/data2/"):
        return Path("/data2/mope-jepa-assets")
    return (REPO_ROOT.parent / "mope-jepa-assets").resolve()


def parse_args():
    assets_root = default_assets_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=(
        assets_root
        / "jepa_checkpoints"
        / "native_mope8_dense4_moe4_top2_wisa_dyn_3dpos"
        / "checkpoint-50.pth"
    ))
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--pool", choices=["none", "temporal", "mean"], default="temporal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sampling-rate", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-npy", action="store_true")
    return parser.parse_args()


def build_inference_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_video_frames(video_path, num_frames, sampling_rate):
    vr = get_video_loader()(str(video_path))
    total = len(vr)
    skip = num_frames * sampling_rate
    if total <= skip:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        center = total // 2
        start = max(0, center - skip // 2)
        start = min(start, total - skip)
        indices = np.arange(start, start + skip, sampling_rate)[:num_frames]
    indices = np.clip(indices, 0, total - 1).tolist()
    data = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(data[i]).convert("RGB") for i in range(len(indices))]


def frames_to_tensor(images, transform, num_frames):
    frames = [transform(img) for img in images]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames[:num_frames], dim=1).unsqueeze(0)


def iter_videos(args):
    if args.video is not None:
        yield args.video
    if args.video_dir is not None:
        for path in sorted(args.video_dir.rglob("*")):
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def load_native_model(args):
    model = native_mope_jepa_base_patch16_224(
        pretrained=False,
        all_frames=args.num_frames,
        tubelet_size=2,
        encoder_depth=8,
        dense_layers=4,
        num_experts=4,
        top_k=2,
        num_shared_experts=1,
        router_score_func="sigmoid",
        router_bias_update_speed=0.0,
        num_anchors=1,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = None
    for key in ("model", "module", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        state = ckpt
    state = {str(k).replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[load] missing sample={msg.missing_keys[:5]}")
    model.eval().to(args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def encode_video(model, video_path, args, transform):
    frames = load_video_frames(video_path, args.num_frames, args.sampling_rate)
    video = frames_to_tensor(frames, transform, args.num_frames).to(args.device)
    feat = model.encode(video, token_mask=None, record_stats=False).detach().float().cpu()
    if args.pool == "mean":
        feat = feat.mean(dim=1, keepdim=True)
    elif args.pool == "temporal":
        bsz, tokens, dim = feat.shape
        t_bins = args.num_frames // 2
        spatial = tokens // t_bins
        feat = feat.view(bsz, t_bins, spatial, dim).mean(dim=2)
    return feat.squeeze(0)


@torch.no_grad()
def main():
    args = parse_args()
    videos = list(iter_videos(args))
    if not videos:
        raise SystemExit("No video found. Provide --video or --video-dir.")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    transform = build_inference_transform(args.input_size)
    model = load_native_model(args)

    for index, video in enumerate(videos, 1):
        out_path = args.save_dir / f"{video.stem}.pt"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue
        feat = encode_video(model, video, args, transform)
        payload = {
            "video": str(video),
            "checkpoint": str(args.ckpt),
            "pool": args.pool,
            "features": feat,
            "shape": list(feat.shape),
        }
        torch.save(payload, out_path)
        if args.save_npy:
            np.save(args.save_dir / f"{video.stem}.npy", feat.numpy())
        print(json.dumps({
            "index": index,
            "total": len(videos),
            "video": str(video),
            "out": str(out_path),
            "shape": list(feat.shape),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
