#!/usr/bin/env python3
"""Extract a tiny, exact-frame VSI ScanNet++ pilot bundle for projection QC."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "scannetppv2_pilot_frame_bundle_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_indices(frame_count: int, count: int = 5) -> list[int]:
    if frame_count < count or count < 2:
        raise ValueError("video must contain at least count>=2 frames")
    return [
        (position * (frame_count - 1)) // (count - 1)
        for position in range(count)
    ]


def media_record(report: dict[str, Any], scene_id: str) -> dict[str, Any]:
    expected = f"scannetppv2/{scene_id}.mp4"
    matches = [row for row in report.get("videos", ()) if row.get("media") == expected]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one video metadata row for {expected}")
    row = matches[0]
    if row.get("status") != "ok":
        raise ValueError(f"video metadata is not usable: {row.get('status')}")
    return row


def resolve_video(media_root: Path, media: str) -> Path:
    path = PurePosixPath(media)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "scannetppv2"
        or path.suffix.lower() != ".mp4"
    ):
        raise ValueError(f"invalid ScanNet++ media path: {media!r}")
    root = media_root.resolve()
    resolved = root.joinpath(*path.parts).resolve()
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-metadata", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-output", required=True, type=Path)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    if shutil.which(args.ffmpeg) is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {args.ffmpeg}")
    report = json.loads(args.video_metadata.read_text(encoding="utf-8"))
    row = media_record(report, args.scene_id)
    frame_count = int(row["frame_count"])
    indices = selected_indices(frame_count, args.frame_count)
    video = resolve_video(args.media_root, row["media"])

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.archive_output.exists():
        raise FileExistsError(args.archive_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for index in indices:
        output = args.output_dir / f"frame_{index:06d}.png"
        command = [
            args.ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{index})",
            # ``-fps_mode`` was added after the ffmpeg release installed on
            # the execution server.  ``-vsync 0`` is the legacy-compatible
            # equivalent here and prevents timestamp-driven frame duplication.
            "-vsync",
            "0",
            "-frames:v",
            "1",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"ffmpeg failed for frame {index}: {completed.stderr.strip()}"
            )
        frames.append(
            {
                "vsi_frame_index": index,
                "candidate_official_iphone_frame_index": index,
                "path": output.name,
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": args.scene_id,
        "source_video": {
            key: row.get(key)
            for key in (
                "media",
                "path",
                "size_bytes",
                "codec_name",
                "width",
                "height",
                "frame_count",
                "duration_seconds",
                "avg_fps",
                "r_fps",
                "avg_frame_rate_raw",
                "r_frame_rate_raw",
            )
        },
        "video_metadata_report": {
            "path": str(args.video_metadata.resolve()),
            "sha256": sha256_file(args.video_metadata),
        },
        "candidate_frame_mapping": "vsi_mp4_frame_i_to_iphone_frame_i",
        "frames": frames,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(args.archive_output, "w:gz") as archive:
        archive.add(args.output_dir, arcname=args.output_dir.name)
    result = {
        **manifest,
        "archive": {
            "path": str(args.archive_output.resolve()),
            "size_bytes": args.archive_output.stat().st_size,
            "sha256": sha256_file(args.archive_output),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
