"""Add validated full-video sidecars to an existing VSI-590K SPAR manifest.

GUIDE keeps consuming the existing eight ``image`` entries.  The added
``mope_video`` entry points at the source video so MoPE-final515k can perform
its owner-defined 4 groups x 4 frames sampling independently.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _metadata(row: dict) -> tuple[str, str]:
    info = row.get("spar_info", {})
    if isinstance(info, str):
        info = json.loads(info)
    dataset = info.get("dataset") or row.get("metadata", {}).get("dataset")
    scene = info.get("scene_name") or row.get("metadata", {}).get("scene_name")
    if not dataset or not scene:
        raise ValueError(f"missing dataset/scene metadata for id={row.get('id')}")
    return str(dataset), str(scene)


def _resolve_video(video_root: Path, dataset: str, scene: str) -> Path:
    flat = video_root / dataset / f"{scene}.mp4"
    if flat.is_file():
        return flat.resolve()
    scene_dir = video_root / dataset / scene
    preferred = scene_dir / "raw_navigation_camera__0.mp4"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(scene_dir.glob("*.mp4")) if scene_dir.is_dir() else []
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(
        f"no source video for dataset={dataset} scene={scene}; "
        f"checked {flat} and {scene_dir}"
    )


def migrate(rows: list[dict], video_root: Path) -> list[dict]:
    migrated = []
    for row in rows:
        dataset, scene = _metadata(row)
        video = _resolve_video(video_root, dataset, scene)
        updated = dict(row)
        updated["mope_video"] = str(video)
        migrated.append(updated)
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must differ from input; preserve the GUIDE manifest")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"expected a non-empty JSON list: {args.input}")
    migrated = migrate(rows, args.video_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(migrated, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"PASS rows={len(migrated)} output={args.output}")


if __name__ == "__main__":
    main()
