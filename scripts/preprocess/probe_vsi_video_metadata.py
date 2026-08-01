#!/usr/bin/env python3
"""Probe exact source video metadata without copying or decoding the videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "vsi_video_metadata_probe_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_media(jsonl: Path, source: str) -> list[str]:
    values: set[str] = set()
    prefix = f"{source}/"
    with jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{jsonl}:{line_number}: {error}") from error
            media = row.get("video")
            if isinstance(media, str) and media.startswith(prefix):
                values.add(media)
    if not values:
        raise ValueError(f"no video rows found for source={source!r}")
    return sorted(values)


def resolve_media(root: Path, media: str, source: str) -> Path:
    posix = PurePosixPath(media)
    if (
        posix.is_absolute()
        or len(posix.parts) != 2
        or posix.parts[0] != source
        or posix.suffix.lower() != ".mp4"
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"invalid {source} media path: {media!r}")
    root = root.resolve()
    path = root.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"media escapes root: {media!r}") from error
    return path


def parse_rate(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    rate = Fraction(str(value))
    return float(rate) if rate > 0 else None


def probe_one(media: str, path: Path, ffprobe: str) -> dict[str, Any]:
    if not path.is_file():
        return {"media": media, "path": str(path), "status": "missing"}
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "media": media,
            "path": str(path),
            "status": "ffprobe_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        return {
            "media": media,
            "path": str(path),
            "status": "video_stream_count_invalid",
            "video_stream_count": len(streams),
        }
    stream = streams[0]
    frame_count = stream.get("nb_frames")
    duration = stream.get("duration", payload.get("format", {}).get("duration"))
    try:
        frame_count_int = int(frame_count)
    except (TypeError, ValueError):
        frame_count_int = None
    try:
        duration_float = float(duration)
    except (TypeError, ValueError):
        duration_float = None
    avg_fps = parse_rate(stream.get("avg_frame_rate"))
    r_fps = parse_rate(stream.get("r_frame_rate"))
    status = "ok"
    if frame_count_int is None or frame_count_int <= 0:
        status = "frame_count_unavailable"
    elif avg_fps is None or duration_float is None or duration_float <= 0:
        status = "timing_unavailable"
    return {
        "media": media,
        "scene_id": PurePosixPath(media).stem,
        "path": str(path),
        "status": status,
        "size_bytes": path.stat().st_size,
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "frame_count": frame_count_int,
        "duration_seconds": duration_float,
        "avg_fps": avg_fps,
        "r_fps": r_fps,
        "avg_frame_rate_raw": stream.get("avg_frame_rate"),
        "r_frame_rate_raw": stream.get("r_frame_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--source", default="scannetppv2")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    if not args.media_root.is_dir():
        raise FileNotFoundError(args.media_root)

    media_values = source_media(args.jsonl, args.source)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                probe_one,
                media,
                resolve_media(args.media_root, media, args.source),
                args.ffprobe,
            ): media
            for media in media_values
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: row["media"])
    status_counts: dict[str, int] = {}
    for row in records:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if status_counts == {"ok": len(records)} else "failed",
        "source": args.source,
        "jsonl": str(args.jsonl.resolve()),
        "jsonl_sha256": sha256_file(args.jsonl),
        "media_root": str(args.media_root.resolve()),
        "video_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "videos": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "videos"}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
