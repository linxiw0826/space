"""
v0b_layer_probe.py  —  Frozen-probing experiment: MoPE layer separability
==========================================================================

Purpose
-------
Extract intermediate-layer features from a frozen MoPEEncoder and train a
linear probe (LogisticRegression + 5-fold StratifiedKFold) per layer to
measure how well each layer separates "dynamic" from "static" videos.

This answers two questions:
  (a) Does the MoPE encoder capture dynamic/physical information?
  (b) Which layer is best for downstream spatial/temporal reasoning?

Manifest format (--manifest)
-----------------------------
A CSV file with a header row and two columns:

    video_path,label
    /path/to/video1.mp4,1
    /path/to/video2.mp4,0
    ...

  video_path : absolute (or relative to CWD) path to a video file.
               Supported formats: .mp4 .avi .mov .mkv .webm .flv
  label      : integer class label
                 0 = static  (no significant motion / physics event)
                 1 = dynamic (motion, collision, deformation, etc.)

Fill in paths that are accessible on the remote server where the script runs.

Output (--output)
-----------------
A JSON file with the structure:

    {
      "layer_7":       {"accuracy_mean": 0.82, "accuracy_std": 0.03,
                        "auc_mean": 0.89,      "auc_std": 0.02,
                        "n_samples": 120},
      "layer_11":      {...},
      "layer_postnorm": {...},    # final encoder output after self.norm
      "meta": {
          "n_total": 130, "n_skipped": 10,
          "layers_probed": [7, 11, "postnorm"],
          "num_frames": 16, "n_folds": 5
      }
    }

Usage
-----
    python v0b_layer_probe.py \\
        --manifest /path/to/manifest.csv \\
        --mope_ckpt /path/to/checkpoint.pth \\
        --layers 7,11 \\
        --output /path/to/report.json \\
        --num_frames 16 \\
        --device cuda

The script does NOT modify any MoPE model files.
All intermediate features are collected via PyTorch forward hooks (zero-invasive).
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# ---------------------------------------------------------------------------
# Project sys.path setup
# ---------------------------------------------------------------------------
# This script lives at:  <SPACE_ROOT>/scripts/probe/v0b_layer_probe.py
# We need:               <SPACE_ROOT>  (for "from src.model.mope_encoder import …")
#                        <SPACE_ROOT>/src/vendor/mope  (for MoPE's own "models" package)
_SCRIPT_DIR = Path(__file__).resolve().parent
_SPACE_ROOT = _SCRIPT_DIR.parent.parent          # .../projects/space
_SRC_ROOT   = _SPACE_ROOT / "src"
_MOPE_ROOT  = _SRC_ROOT   / "vendor" / "mope"

for _p in [str(_SPACE_ROOT), str(_SRC_ROOT), str(_MOPE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Now we can import the MoPE video-loader and our encoder wrapper.
from dataset.loader import get_video_loader          # noqa: E402  (from vendor/mope)
from src.model.mope_encoder import MoPEEncoder        # noqa: E402

# ---------------------------------------------------------------------------
# Video preprocessing  (mirrors infer_mope.py exactly)
# ---------------------------------------------------------------------------

def build_inference_transform(input_size: int = 224) -> transforms.Compose:
    """ImageNet-normalised crop/resize transform — must match MoPE training."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(input_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_video_frames(video_path: str, num_frames: int = 16,
                      sampling_rate: int = 4):
    """Load `num_frames` PIL images from a video file.

    Sampling strategy (identical to infer_mope.py):
      - If the video is shorter than num_frames*sampling_rate frames, use
        linspace indices to sample uniformly across the full clip.
      - Otherwise, sample from the centre of the clip.
    """
    from PIL import Image  # lazy import — PIL is not always installed globally

    video_loader = get_video_loader()
    vr = video_loader(video_path)
    total = len(vr)
    skip_length = num_frames * sampling_rate

    if total <= skip_length:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        center = total // 2
        start  = max(0, center - skip_length // 2)
        start  = min(start, total - skip_length)
        indices = np.arange(start, start + skip_length, sampling_rate)[:num_frames]

    indices = np.clip(indices, 0, total - 1).tolist()
    video_data = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(video_data[i]).convert("RGB")
            for i in range(len(indices))]


def frames_to_tensor(images, transform: transforms.Compose,
                     num_frames: int = 16) -> torch.Tensor:
    """Stack PIL frames into a [1, C, T, H, W] float tensor."""
    frames = [transform(img) for img in images]
    # Pad by repeating last frame if clip was shorter than num_frames.
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]
    return torch.stack(frames, dim=1).unsqueeze(0)  # [1, C, T, H, W]


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(csv_path: str) -> List[Tuple[str, int]]:
    """Parse manifest CSV → list of (video_path, label) tuples.

    Validates that:
      - The file has at least one data row.
      - Labels are integers.
      - At least two classes are present.
    """
    rows: List[Tuple[str, int]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "video_path" not in (reader.fieldnames or []):
            raise ValueError(
                f"Manifest CSV must have a 'video_path' column. "
                f"Found: {reader.fieldnames}"
            )
        if "label" not in (reader.fieldnames or []):
            raise ValueError(
                f"Manifest CSV must have a 'label' column. "
                f"Found: {reader.fieldnames}"
            )
        for i, row in enumerate(reader, start=2):   # start=2 → row number in file
            try:
                label = int(row["label"])
            except (ValueError, KeyError):
                raise ValueError(
                    f"Manifest row {i}: cannot parse label={row.get('label')!r} as int."
                )
            rows.append((row["video_path"].strip(), label))

    if not rows:
        raise ValueError(f"Manifest '{csv_path}' is empty — no data rows found.")

    classes = set(lbl for _, lbl in rows)
    if len(classes) < 2:
        raise ValueError(
            f"Manifest contains only one class ({classes}). "
            "Need at least two classes (0=static, 1=dynamic) for a probe experiment."
        )

    return rows


# ---------------------------------------------------------------------------
# Feature extraction with layer hooks
# ---------------------------------------------------------------------------

class LayerHookCollector:
    """
    Attach forward hooks to specified blocks of the MoPE encoder and
    collect their outputs in a dict keyed by layer index.

    Usage:
        with LayerHookCollector(encoder, layers=[7, 11]) as collector:
            _ = mope_encoder(frames)   # triggers hooks
        feats = collector.features     # {7: Tensor[B, N_vis, 768], ...}
    """

    def __init__(self, encoder: nn.Module, layers: List[int]):
        self.encoder = encoder
        self.layers  = layers
        self.features: Dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        for layer_idx in self.layers:
            handle = self.encoder.blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # `output` for a transformer block is typically a Tensor [B, N, D]
            # or a tuple whose first element is the hidden state.
            if isinstance(output, (tuple, list)):
                feat = output[0]
            else:
                feat = output
            # Detach and move to CPU immediately to save GPU memory.
            self.features[layer_idx] = feat.detach().cpu()
        return hook

    def reset(self):
        self.features.clear()


def extract_all_features(
    mope_encoder: MoPEEncoder,
    manifest: List[Tuple[str, int]],
    layers: List[int],
    num_frames: int,
    device: torch.device,
    sampling_rate: int = 4,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, int]:
    """
    Run inference on every video in the manifest, collecting per-layer
    mean-pooled features.

    Returns
    -------
    features_by_layer : dict mapping layer key → float32 array [N, 768]
                        Keys: "layer_<i>" for i in `layers`, plus "layer_postnorm".
    labels            : int array [N]
    n_skipped         : number of videos that failed to load / process
    """
    transform = build_inference_transform(input_size=224)

    # Determine the underlying ViT encoder (encoder attribute on MoPEEncoder).
    inner_encoder = mope_encoder.encoder

    # Pre-allocate lists.
    layer_keys     = [f"layer_{i}" for i in layers] + ["layer_postnorm"]
    features_accum: Dict[str, List[np.ndarray]] = {k: [] for k in layer_keys}
    labels_accum: List[int] = []
    n_skipped = 0

    total = len(manifest)
    print(f"\n[extract] Processing {total} videos ...")

    for idx, (video_path, label) in enumerate(manifest):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}]  {Path(video_path).name}")

        try:
            images = load_video_frames(video_path,
                                       num_frames=num_frames,
                                       sampling_rate=sampling_rate)
            frames_tensor = frames_to_tensor(images, transform,
                                             num_frames=num_frames)
            frames_tensor = frames_tensor.to(device)

            hook_collector = LayerHookCollector(inner_encoder, layers)

            with hook_collector:
                # MoPEEncoder.forward is already @torch.no_grad.
                postnorm_out = mope_encoder(frames_tensor)  # [1, N_vis, 768]

            # Mean-pool over the token dimension → [768] per layer.
            for layer_idx in layers:
                feat = hook_collector.features[layer_idx]  # [1, N_vis, 768]
                pooled = feat[0].mean(dim=0).numpy()        # [768]
                features_accum[f"layer_{layer_idx}"].append(pooled)

            # Post-norm (final encoder output, already mean-pooled over tokens).
            postnorm_pooled = postnorm_out[0].mean(dim=0).cpu().numpy()  # [768]
            features_accum["layer_postnorm"].append(postnorm_pooled)

            labels_accum.append(label)

        except Exception as exc:
            n_skipped += 1
            print(f"  [SKIP] {video_path}: {exc}")
            continue

    if not labels_accum:
        raise RuntimeError(
            "All videos failed to load. Check video paths and codec support."
        )

    features_by_layer = {
        k: np.stack(v, axis=0).astype(np.float32)
        for k, v in features_accum.items()
    }
    labels = np.array(labels_accum, dtype=np.int64)

    print(f"\n[extract] Done. Loaded {len(labels_accum)} / {total} videos "
          f"({n_skipped} skipped).")
    return features_by_layer, labels, n_skipped


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    layer_name: str = "?",
) -> Dict:
    """
    Train a linear LogisticRegression probe on features X with labels y
    using StratifiedKFold cross-validation.

    Features are StandardScaled within each fold (fit on train, applied to val)
    to prevent data leakage.

    Returns a dict with accuracy_mean, accuracy_std, auc_mean, auc_std,
    n_samples, n_folds_used.
    """
    from sklearn.linear_model    import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import accuracy_score, roc_auc_score

    n_samples = len(y)
    classes   = np.unique(y)

    # Auto-reduce folds if we have very few samples.
    min_class_count = min(np.bincount(y)[c] for c in classes)
    safe_folds      = min(n_folds, min_class_count)
    if safe_folds < 2:
        print(f"  [probe:{layer_name}] Not enough samples per class "
              f"(min={min_class_count}) for cross-validation. Skipping.")
        return {
            "accuracy_mean": None, "accuracy_std": None,
            "auc_mean": None,      "auc_std": None,
            "n_samples": n_samples, "n_folds_used": 0,
            "error": f"Insufficient samples per class (min={min_class_count})"
        }
    if safe_folds < n_folds:
        print(f"  [probe:{layer_name}] Reduced folds {n_folds}→{safe_folds} "
              f"(min class count={min_class_count}).")

    skf         = StratifiedKFold(n_splits=safe_folds, shuffle=True,
                                  random_state=42)
    accuracies  = []
    aucs        = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features (no leakage: fit only on train).
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress ConvergenceWarning
            clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_val)
        y_prob  = clf.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        # For binary, use the positive class probability; for multi-class,
        # use the OvR macro AUC.
        if len(classes) == 2:
            auc = roc_auc_score(y_val, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_val, y_prob, multi_class="ovr",
                                average="macro")

        accuracies.append(acc)
        aucs.append(auc)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std":  float(np.std(accuracies)),
        "auc_mean":      float(np.mean(aucs)),
        "auc_std":       float(np.std(aucs)),
        "n_samples":     n_samples,
        "n_folds_used":  safe_folds,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report_table(results: Dict, layers: List[int]):
    """Pretty-print a sorted ranking table to stdout."""
    layer_keys = [f"layer_{i}" for i in layers] + ["layer_postnorm"]
    rows = []
    for k in layer_keys:
        r = results.get(k, {})
        acc  = r.get("accuracy_mean")
        auc  = r.get("auc_mean")
        if acc is None:
            rows.append((k, -1.0, -1.0, r.get("n_samples", 0), r.get("error", "")))
        else:
            rows.append((k, acc, auc, r.get("n_samples", 0), ""))

    # Sort by AUC descending (use acc as tiebreak).
    rows.sort(key=lambda x: (x[2], x[1]), reverse=True)

    header = f"\n{'Layer':<20} {'Acc (mean±std)':<22} {'AUC (mean±std)':<22} {'N':>6}  Note"
    sep    = "-" * 80
    print(header)
    print(sep)
    for (key, acc, auc, n, note) in rows:
        r = results.get(key, {})
        if acc < 0:
            print(f"  {key:<18}  {'SKIPPED':^20}  {'SKIPPED':^20}  {n:>6}  {note}")
        else:
            acc_str = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
            auc_str = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
            print(f"  {key:<18}  {acc_str:<22}  {auc_str:<22}  {n:>6}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=("Frozen-probing: evaluate MoPE layer separability for "
                     "dynamic vs static video classification."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest", required=True,
        help=("Path to the manifest CSV file.  "
              "Columns: video_path,label  (label: 0=static, 1=dynamic)."),
    )
    p.add_argument(
        "--mope_ckpt", required=True,
        help="Path to the MoPE checkpoint .pth file.",
    )
    p.add_argument(
        "--layers", default="7,11",
        help=("Comma-separated block indices to probe (0-based, depth=12).  "
              "Additionally, 'layer_postnorm' (final encoder output after "
              "self.norm) is always included as a baseline.  Default: '7,11'."),
    )
    p.add_argument(
        "--output", required=True,
        help="Path for the JSON report output file.",
    )
    p.add_argument(
        "--num_frames", type=int, default=16,
        help=("Number of video frames to sample (default: 16).  "
              "Note: MoPEEncoder was built with all_frames=8; this controls "
              "how many frames are read from disk before subsampling by the "
              "encoder's tubelet stride."),
    )
    p.add_argument(
        "--sampling_rate", type=int, default=4,
        help="Temporal sampling rate for frame extraction (default: 4).",
    )
    p.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of StratifiedKFold cross-validation folds (default: 5).",
    )
    p.add_argument(
        "--device", default="cuda",
        help="PyTorch device string, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda).",
    )
    p.add_argument(
        "--all_frames", type=int, default=8,
        help=("all_frames param for MoPEEncoder (tubelet/temporal stride context, "
              "default: 8).  Must match the checkpoint's training config."),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Device ---
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[main] Using device: {device}")

    # --- Parse layer list ---
    try:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    except ValueError:
        raise ValueError(
            f"--layers must be a comma-separated list of integers, got: {args.layers!r}"
        )
    if not layers:
        raise ValueError("--layers produced an empty list.")
    for li in layers:
        if not (0 <= li <= 11):
            raise ValueError(
                f"Layer index {li} is out of range. MoPE ViT-B has 12 blocks (0-11)."
            )
    print(f"[main] Probing layers: {layers}  + postnorm baseline")

    # --- Load manifest ---
    print(f"[main] Loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)
    print(f"[main] Manifest: {len(manifest)} entries, "
          f"classes: {sorted(set(lbl for _, lbl in manifest))}")

    # --- Load frozen MoPE encoder ---
    print(f"[main] Loading MoPEEncoder from: {args.mope_ckpt}")
    mope_encoder = MoPEEncoder(
        checkpoint_path=args.mope_ckpt,
        all_frames=args.all_frames,
    )
    mope_encoder.to(device)
    mope_encoder.eval()

    # Sanity-check: the encoder must have a .blocks ModuleList of length 12.
    n_blocks = len(mope_encoder.encoder.blocks)
    print(f"[main] Encoder has {n_blocks} blocks.")
    if n_blocks != 12:
        print(f"[WARNING] Expected 12 blocks, found {n_blocks}.  "
              "Proceeding, but check layer indices.")

    # --- Feature extraction ---
    features_by_layer, labels, n_skipped = extract_all_features(
        mope_encoder=mope_encoder,
        manifest=manifest,
        layers=layers,
        num_frames=args.num_frames,
        device=device,
        sampling_rate=args.sampling_rate,
    )

    # --- Per-layer linear probe ---
    print(f"\n[probe] Training linear probes ({args.n_folds}-fold StratifiedKFold)...")
    results: Dict = {}
    layer_keys = [f"layer_{i}" for i in layers] + ["layer_postnorm"]

    for key in layer_keys:
        X = features_by_layer[key]   # [N, 768]
        y = labels                   # [N]
        print(f"\n  Layer '{key}': X.shape={X.shape}, "
              f"class distribution={dict(zip(*np.unique(y, return_counts=True)))}")
        probe_result = train_probe(X, y,
                                   n_folds=args.n_folds,
                                   layer_name=key)
        results[key] = probe_result
        if probe_result.get("accuracy_mean") is not None:
            print(f"    accuracy = {probe_result['accuracy_mean']:.4f} "
                  f"± {probe_result['accuracy_std']:.4f}  |  "
                  f"AUC = {probe_result['auc_mean']:.4f} "
                  f"± {probe_result['auc_std']:.4f}")

    # --- Print ranking table ---
    print_report_table(results, layers)

    # --- Save JSON report ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = dict(results)
    report["meta"] = {
        "n_total":       len(manifest),
        "n_skipped":     n_skipped,
        "n_loaded":      int(len(labels)),
        "layers_probed": layers + ["postnorm"],
        "num_frames":    args.num_frames,
        "sampling_rate": args.sampling_rate,
        "all_frames":    args.all_frames,
        "n_folds":       args.n_folds,
        "manifest":      args.manifest,
        "mope_ckpt":     args.mope_ckpt,
        "device":        str(device),
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"[main] JSON report saved to: {output_path}")
    print("[main] Done.")


if __name__ == "__main__":
    main()
