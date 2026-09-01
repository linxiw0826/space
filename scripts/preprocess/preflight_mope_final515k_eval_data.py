#!/usr/bin/env python3
"""Fail-closed data preflight for final515k E-02c evaluation.

The default metadata pass validates every annotation and every referenced file.
Use ``--decode sample`` for a bounded dual-backend decode check or
``--decode full`` to decode every unique video.  A smoke annotation can be
written without changing the source annotation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


VSI_QUESTION_TYPES = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
VLM4D_SOURCES = ("davis", "ego4d", "youtube-vos")


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"annotation does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"annotation is empty: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    else:
        records = value if isinstance(value, list) else [value]
    if not records:
        raise ValueError("annotation contains no records")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("every annotation record must be a JSON object")
    return records


def _required_text(record: dict, field: str, row: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row}: {field!r} must be a non-empty string")
    return value.strip()


def _required_scalar(record: dict, field: str, row: int):
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"row {row}: {field!r} must be a non-empty scalar")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"row {row}: {field!r} must be a non-empty scalar")
    return value


def _safe_path(root: Path, relative: Path, row: int) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"row {row}: unsafe video relative path: {relative}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"row {row}: video path escapes root: {relative}")
    return path


def resolve_vsibench(record: dict, video_root: Path, row: int) -> Path:
    dataset = _required_text(record, "dataset", row)
    scene = _required_text(record, "scene_name", row)
    _required_text(record, "question", row)
    _required_scalar(record, "ground_truth", row)
    question_type = _required_text(record, "question_type", row)
    if question_type not in VSI_QUESTION_TYPES:
        raise ValueError(f"row {row}: unsupported VSI question_type={question_type!r}")
    if question_type in VSI_QUESTION_TYPES[:6]:
        options = record.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"row {row}: MCA record requires at least two options")
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise ValueError(f"row {row}: MCA options must be non-empty strings")
    return _safe_path(video_root, Path(dataset) / f"{scene}.mp4", row)


def vlm4d_relative_path(value: str, row: int) -> Path:
    if "resolve/main/" in value:
        relative = value.split("resolve/main/", 1)[1]
    else:
        parsed = urlparse(value)
        relative = Path(unquote(parsed.path)).name if parsed.scheme else Path(value).name
    if not relative:
        raise ValueError(f"row {row}: VLM4D video has no usable path")
    return Path(unquote(relative))


def resolve_vlm4d(record: dict, video_root: Path, row: int) -> Path:
    video = _required_text(record, "video", row)
    _required_text(record, "question", row)
    answer = _required_scalar(record, "answer", row)
    question_type = _required_text(record, "question_type", row)
    if question_type != "multiple-choice":
        raise ValueError(f"row {row}: VLM4D question_type must be 'multiple-choice'")
    choices = record.get("choices")
    if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
        raise ValueError(f"row {row}: VLM4D choices must have exactly A/B/C/D")
    choice_values = []
    for value in choices.values():
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"row {row}: VLM4D choice values must be non-empty scalars")
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"row {row}: VLM4D choice values must be non-empty scalars")
        choice_values.append(normalized)
    if str(answer).strip() not in choice_values:
        raise ValueError(f"row {row}: VLM4D answer does not match any choice")
    return _safe_path(video_root, vlm4d_relative_path(video, row), row)


def final515k_indices(frame_count: int) -> list[int]:
    """Owner-script 4 segments x 4 linspace/rint samples."""
    import numpy as np

    if frame_count < 1:
        raise ValueError(f"video has no frames (frame_count={frame_count})")
    indices = []
    for group in range(4):
        start = int(np.floor(group * frame_count / 4))
        end = int(np.floor((group + 1) * frame_count / 4)) - 1
        end = max(end, start)
        indices.extend(np.rint(np.linspace(start, end, 4)).astype(np.int64).tolist())
    return indices


def assert_rgb_frame_parity(decord_frame, opencv_bgr_frame, frame_index: int) -> None:
    """Require the eval decoder to match the owner extractor pixel-for-pixel."""
    import numpy as np

    opencv_rgb = opencv_bgr_frame[..., ::-1]
    if decord_frame.shape != opencv_rgb.shape:
        raise ValueError(
            f"selected frame {frame_index} shape mismatch: "
            f"Decord={decord_frame.shape}, OpenCV-RGB={opencv_rgb.shape}"
        )
    if not np.array_equal(decord_frame, opencv_rgb):
        difference = np.abs(
            decord_frame.astype(np.int16) - opencv_rgb.astype(np.int16)
        )
        raise ValueError(
            f"selected frame {frame_index} RGB mismatch: "
            f"max_abs_diff={int(difference.max())}, "
            f"different_values={int(np.count_nonzero(difference))}"
        )


def decode_video(path: Path) -> tuple[int, list[int]]:
    try:
        import cv2
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ValueError("decode mode requires both opencv-python and decord") from exc

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    decord_count = len(reader)
    indices = final515k_indices(decord_count)
    frames = reader.get_batch(indices).asnumpy()
    if frames.shape[0] != 16 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Decord did not return 16 RGB frames: shape={frames.shape}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open video")
    try:
        cv_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if cv_count != decord_count:
            raise ValueError(f"frame-count mismatch: OpenCV={cv_count}, Decord={decord_count}")
        for position, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"OpenCV could not read selected frame {index}")
            assert_rgb_frame_parity(frames[position], frame, index)
    finally:
        capture.release()
    return decord_count, indices


def select_smoke(dataset: str, records: list[dict]) -> list[dict]:
    if dataset == "vsibench":
        first = {}
        for record in records:
            first.setdefault(record["question_type"], record)
        return [first[name] for name in VSI_QUESTION_TYPES]

    # One item per real-video source keeps the smoke bounded while covering all
    # path families (normally davis, ego4d and youtube-vos).
    first = {}
    for row, record in enumerate(records, 1):
        relative = vlm4d_relative_path(record["video"], row)
        source = relative.parent.name
        first.setdefault(source, record)
    return [first[source] for source in VLM4D_SOURCES]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict) -> None:
    """Atomically replace the latest audit report, including failed runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("vsibench", "vlm4d"))
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--decode", choices=("none", "sample", "full"), default="none")
    parser.add_argument("--decode-limit", type=int, default=16,
                        help="unique videos decoded in sample mode (default: 16)")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.video_root.is_dir():
            raise ValueError(f"video root does not exist: {args.video_root}")
        if args.decode_limit < 1:
            raise ValueError("--decode-limit must be positive")
        records = load_records(args.annotation)
        if args.expected_rows is not None and len(records) != args.expected_rows:
            raise ValueError(
                f"annotation row count mismatch: expected={args.expected_rows}, actual={len(records)}"
            )
        resolver = resolve_vsibench if args.dataset == "vsibench" else resolve_vlm4d
        paths: list[Path] = []
        path_by_record_id: dict[int, Path] = {}
        for row, record in enumerate(records, 1):
            path = resolver(record, args.video_root, row)
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"row {row}: unsupported video extension: {path}")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"row {row}: video is missing or empty: {path}")
            paths.append(path)
            path_by_record_id[id(record)] = path

        type_counts = Counter(record.get("question_type") for record in records)
        if args.dataset == "vsibench":
            missing = [name for name in VSI_QUESTION_TYPES if type_counts[name] == 0]
            if missing:
                raise ValueError(f"VSI annotation lacks required question types: {missing}")
            source_counts = None
        else:
            source_counts = Counter(
                vlm4d_relative_path(record["video"], row).parent.name
                for row, record in enumerate(records, 1)
            )
            actual_sources = set(source_counts)
            expected_sources = set(VLM4D_SOURCES)
            if actual_sources != expected_sources:
                raise ValueError(
                    "VLM4D video sources mismatch: "
                    f"expected={sorted(expected_sources)}, actual={sorted(actual_sources)}"
                )

        unique_paths = list(dict.fromkeys(paths))
        if args.expected_videos is not None and len(unique_paths) != args.expected_videos:
            raise ValueError(
                f"unique video count mismatch: expected={args.expected_videos}, "
                f"actual={len(unique_paths)}"
            )
        smoke = select_smoke(args.dataset, records)
        smoke_paths = list(
            dict.fromkeys(path_by_record_id[id(record)] for record in smoke)
        )
        decode_paths = []
        if args.decode == "sample":
            count = min(args.decode_limit, len(unique_paths))
            # Always cover the exact smoke records first (all VSI task types or
            # all VLM4D source families), then fill remaining slots uniformly.
            decode_paths = smoke_paths[:count]
            selected_paths = set(decode_paths)
            remaining = [path for path in unique_paths if path not in selected_paths]
            slots = count - len(decode_paths)
            if slots == 1:
                decode_paths.extend(remaining[:1])
            elif slots > 1:
                positions = [
                    round(i * (len(remaining) - 1) / (slots - 1))
                    for i in range(slots)
                ]
                decode_paths.extend(remaining[position] for position in positions)
        elif args.decode == "full":
            decode_paths = unique_paths
        decoded = []
        for index, path in enumerate(decode_paths, 1):
            frame_count, indices = decode_video(path)
            decoded.append({"video": str(path), "frame_count": frame_count, "indices": indices})
            print(f"[{index}/{len(decode_paths)}] decode=PASS video={path}", flush=True)

        if args.smoke_output:
            write_jsonl(args.smoke_output, smoke)
        report = {
            "schema_version": "mope_final515k_eval_data_preflight_v1",
            "status": "complete_passed",
            "dataset": args.dataset,
            "annotation": str(args.annotation),
            "video_root": str(args.video_root),
            "annotation_rows": len(records),
            "unique_videos": len(unique_paths),
            "question_type_counts": dict(sorted(type_counts.items())),
            "video_source_counts": (
                dict(sorted(source_counts.items())) if source_counts is not None else None
            ),
            "metadata_videos_checked": len(unique_paths),
            "decode_mode": args.decode,
            "decoded_videos": len(decoded),
            "decoded": decoded,
            "smoke_rows": len(smoke),
            "smoke_output": str(args.smoke_output) if args.smoke_output else None,
        }
        if args.report:
            write_report(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        if args.report:
            try:
                write_report(args.report, {
                    "schema_version": "mope_final515k_eval_data_preflight_v1",
                    "status": "failed",
                    "dataset": args.dataset,
                    "annotation": str(args.annotation),
                    "video_root": str(args.video_root),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception as report_exc:
                print(
                    f"WARNING: could not persist failed preflight report: {report_exc}",
                    file=sys.stderr,
                )
        print(f"STATUS=FAILED error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
