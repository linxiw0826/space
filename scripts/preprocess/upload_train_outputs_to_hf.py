#!/usr/bin/env python3
"""Upload every trained model under /data2/wlx/output/train to a private HF
repo named after the experiment directory (space_paper1/<name>).

For each experiment directory, only the *final* weights are uploaded: the
directory root itself if it's a complete HF save (has config.json), otherwise
the highest-numbered checkpoint-N subdirectory that has one. Training-state
files (optimizer.pt, scheduler.pt, rng_state*, training_args.bin) are
excluded — they are typically larger than the model weights and are not
needed to load/run the model.

Usage:
    huggingface-cli login   # once, with a token that has write access
    python upload_train_outputs_to_hf.py [--root /data2/wlx/output/train] \
        [--namespace space_paper1] [--dry-run] [--only NAME [NAME ...]]
"""
import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

EXCLUDE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "training_args.bin",
    "*.work.*",
    "*.staged.*",
    "*.backup.*",
    # DeepSpeed ZeRO native checkpoint shards — raw optimizer/model state used
    # only to resume training, duplicates of the HF-format safetensors weights
    # and typically far larger (fp32 optimizer moments, per-rank shards).
    "*optim_states.pt",
    "*model_states.pt",
    "global_step*/**",
    "latest",
    "zero_to_fp32.py",
]


def find_final_weights_dir(exp_dir: Path) -> Path | None:
    """Return the directory holding the final complete HF checkpoint, or None."""
    if (exp_dir / "config.json").is_file():
        return exp_dir
    candidates = []
    for ckpt in exp_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", ckpt.name)
        if m and (ckpt / "config.json").is_file():
            candidates.append((int(m.group(1)), ckpt))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def discover_experiments(root: Path) -> dict[str, Path]:
    """Map repo-name -> final-weights-dir for every model found under root.

    Handles one level of nesting (e.g. guide_reproduced/4b, guide_reproduced/8b)
    in addition to the flat case (e.g. e05_quarter_epoch3).
    """
    experiments: dict[str, Path] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        weights_dir = find_final_weights_dir(entry)
        if weights_dir is not None:
            experiments[entry.name] = weights_dir
            continue
        # one level of nesting, e.g. guide_reproduced/{4b,8b}
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir():
                continue
            sub_weights_dir = find_final_weights_dir(sub)
            if sub_weights_dir is not None:
                experiments[f"{entry.name}_{sub.name}"] = sub_weights_dir
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data2/wlx/output/train")
    parser.add_argument("--namespace", default="space_paper1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="only upload these experiment names")
    args = parser.parse_args()

    root = Path(args.root)
    experiments = discover_experiments(root)

    if args.only:
        experiments = {k: v for k, v in experiments.items() if k in args.only}

    if not experiments:
        print("No experiments with a complete HF checkpoint found.")
        return

    print(f"Found {len(experiments)} experiment(s) to upload:")
    for name, weights_dir in experiments.items():
        print(f"  {name:60s} <- {weights_dir}")

    if args.dry_run:
        print("\n--dry-run set, not uploading.")
        return

    api = HfApi()
    for name, weights_dir in experiments.items():
        repo_id = f"{args.namespace}/{name}"
        print(f"\n=== {repo_id} ===")
        api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(weights_dir),
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=EXCLUDE_PATTERNS,
        )
        print(f"Uploaded {weights_dir} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
