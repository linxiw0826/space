#!/usr/bin/env python3
"""Read-only coverage audit for ADT GT-supported GUIDE clips.

This script never modifies source archives, videos, or training tables.  It
finds raw MP4 frames that pass the frozen trajectory/calibration timestamp
thresholds, selects the longest contiguous run (earliest run breaks ties), and
applies GUIDE's deterministic dynamic 16--32 frame sampler inside that run.

ADT object geometry is scene-level direct GT.  RGB boxes and dynamic object
poses are observation-level evidence: a frame with no RGB boxes can be a valid
empty observation, so box presence is reported but is not misused as a
per-frame coverage requirement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT))

from scripts.preprocess.finalize_adt_parta_training_data import (
    RGB_STREAM_ID,
    csv_rows,
    load_calibrations,
    load_qa,
    load_trajectory,
    mp4_timestamps,
    mp4_video_metadata,
    nearest_index,
)
from src.parta_data_contract import T0_FIXTURES, guide_frame_indices


SCHEMA_VERSION = "adt_gt_supported_clip_coverage_v1"
SAMPLING_POLICY = "guide_exact_over_gt_supported_clip_v1"
REQUIRED_GT_MEMBERS = {
    "aria_trajectory.csv",
    "instances.json",
    "3d_bounding_box.csv",
    "scene_objects.csv",
    "2d_bounding_box.csv",
}
REJECTION_CODES = {
    "missing_video",
    "groundtruth_file_count",
    "calibration_file_count",
    "video_metadata_invalid",
    "video_timestamp_count_mismatch",
    "trajectory_empty",
    "calibration_empty",
    "direct_gt_member_missing",
    "direct_gt_parse_error",
    "direct_gt_object_join_empty",
    "artifact_hash_error",
    "no_gt_supported_run",
    "gt_supported_run_too_short",
    "scene_processing_error",
}


class SceneAuditError(ValueError):
    """Scene-local, machine-classified audit failure."""

    def __init__(self, code: str, detail: str):
        if code not in REJECTION_CODES:
            raise ValueError(f"Unknown rejection code: {code}")
        super().__init__(detail)
        self.code = code
        self.detail = detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--groundtruth-root", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--base-interval", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument(
        "--max-trajectory-error-ns", type=int, default=5_000_000
    )
    parser.add_argument(
        "--max-calibration-error-ns", type=int, default=50_000_000
    )
    parser.add_argument("--sequence-limit", type=int)
    parser.add_argument("--require-t0-fixtures", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def artifact_record(path: Path) -> dict[str, Any]:
    """Return provenance even when a candidate is missing or cannot be hashed."""
    resolved = path.resolve()
    if not path.is_file():
        return {
            "status": "missing",
            "path": str(resolved),
            "size_bytes": None,
            "sha256": None,
        }
    try:
        return {"status": "ok", **file_artifact(path)}
    except Exception as error:
        return {
            "status": "hash_error",
            "path": str(resolved),
            "size_bytes": (
                path.stat().st_size if path.exists() else None
            ),
            "sha256": None,
            "error": f"{type(error).__name__}: {error}",
        }


def candidate_artifact(
    candidates: Sequence[Path],
    *,
    expected_path: Path,
) -> dict[str, Any]:
    if not candidates:
        return artifact_record(expected_path)
    records = [artifact_record(path) for path in candidates]
    if len(records) == 1:
        return records[0]
    return {
        "status": "multiple",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "candidates": records,
    }


def artifact_paths(record: dict[str, Any]) -> list[Path]:
    if record["status"] == "multiple":
        return [
            Path(candidate["path"]) for candidate in record["candidates"]
        ]
    return [Path(record["path"])] if record.get("path") else []


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def temporal_support(
    frame_timestamps: Sequence[int],
    trajectory_timestamps: Sequence[int],
    calibration_timestamps: Sequence[int],
    *,
    max_trajectory_error_ns: int,
    max_calibration_error_ns: int,
) -> tuple[list[bool], list[dict[str, Any]]]:
    """Return per-raw-frame hard support and diagnostics."""
    if not trajectory_timestamps:
        raise SceneAuditError("trajectory_empty", "Trajectory contains no timestamps")
    if not calibration_timestamps:
        raise SceneAuditError(
            "calibration_empty", "Calibration contains no timestamps"
        )
    valid = []
    diagnostics = []
    for frame_index, timestamp in enumerate(frame_timestamps):
        trajectory_error = None
        calibration_error = None
        reasons = []
        if not trajectory_timestamps[0] <= timestamp <= trajectory_timestamps[-1]:
            reasons.append("outside_trajectory_span")
        else:
            trajectory_index = nearest_index(trajectory_timestamps, timestamp)
            trajectory_error = abs(
                trajectory_timestamps[trajectory_index] - timestamp
            )
            if trajectory_error > max_trajectory_error_ns:
                reasons.append("trajectory_time_gap")
        calibration_index = nearest_index(calibration_timestamps, timestamp)
        calibration_error = abs(
            calibration_timestamps[calibration_index] - timestamp
        )
        if calibration_error > max_calibration_error_ns:
            reasons.append("calibration_time_gap")
        valid.append(not reasons)
        diagnostics.append({
            "frame_index": frame_index,
            "device_timestamp_ns": int(timestamp),
            "trajectory_timestamp_error_ns": trajectory_error,
            "calibration_timestamp_error_ns": calibration_error,
            "valid": not reasons,
            "reasons": reasons,
        })
    return valid, diagnostics


def contiguous_true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Return inclusive contiguous true runs in raw-frame coordinates."""
    runs = []
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def select_maximal_run(runs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Choose longest run; the earliest start is the deterministic tie-break."""
    if not runs:
        raise SceneAuditError(
            "no_gt_supported_run", "No GT-supported raw-frame run"
        )
    return min(runs, key=lambda run: (-(run[1] - run[0] + 1), run[0]))


def sample_run(
    run: tuple[int, int],
    fps: float,
    *,
    base_interval: float,
    min_frames: int,
    max_frames: int,
) -> list[int]:
    start, end = run
    try:
        local = guide_frame_indices(
            end - start + 1,
            fps,
            base_interval=base_interval,
            min_frames=min_frames,
            max_frames=max_frames,
        )
    except Exception as error:
        raise SceneAuditError(
            "gt_supported_run_too_short", str(error)
        ) from error
    selected = [start + value for value in local]
    if len(selected) != len(set(selected)):
        raise AssertionError("GUIDE selected duplicate raw-frame IDs")
    if any(index < start or index > end for index in selected):
        raise AssertionError("GUIDE selected an ID outside the chosen run")
    return selected


def analyze_arrays(
    *,
    scene_id: str,
    frame_timestamps: Sequence[int],
    fps: float,
    trajectory_timestamps: Sequence[int],
    calibration_timestamps: Sequence[int],
    qa_count: int,
    base_interval: float = 1.0,
    min_frames: int = 16,
    max_frames: int = 32,
    max_trajectory_error_ns: int = 5_000_000,
    max_calibration_error_ns: int = 50_000_000,
) -> dict[str, Any]:
    """Pure core used by the CLI and synthetic tests."""
    total_frames = len(frame_timestamps)
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise SceneAuditError(
            "video_metadata_invalid", "Invalid MP4 frame metadata"
        )
    mask, diagnostics = temporal_support(
        frame_timestamps,
        trajectory_timestamps,
        calibration_timestamps,
        max_trajectory_error_ns=max_trajectory_error_ns,
        max_calibration_error_ns=max_calibration_error_ns,
    )
    runs = contiguous_true_runs(mask)
    chosen = select_maximal_run(runs)
    selected = sample_run(
        chosen,
        fps,
        base_interval=base_interval,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    selected_diagnostics = [diagnostics[index] for index in selected]
    invalid_selected = [
        item for item in selected_diagnostics if not item["valid"]
    ]
    if invalid_selected:
        raise AssertionError(
            "Selected raw IDs are not fully GT-supported: "
            f"{invalid_selected}"
        )
    start, end = chosen
    run_lengths = [run_end - run_start + 1 for run_start, run_end in runs]
    whole_duration = (
        (frame_timestamps[-1] - frame_timestamps[0]) / 1e9
        if total_frames > 1
        else 0.0
    )
    clip_duration = (
        (frame_timestamps[end] - frame_timestamps[start]) / 1e9
        if end > start
        else 0.0
    )
    duration_coverage = (
        clip_duration / whole_duration
        if whole_duration > 0
        else (1.0 if start == 0 and end == total_frames - 1 else 0.0)
    )
    return {
        "scene_id": scene_id,
        "video_total_frames": total_frames,
        "video_fps": float(fps),
        "whole_video_duration_s": float(whole_duration),
        "clip_start_raw_frame": start,
        "clip_end_raw_frame": end,
        "clip_start_device_timestamp_ns": int(frame_timestamps[start]),
        "clip_end_device_timestamp_ns": int(frame_timestamps[end]),
        "clip_duration_s": float(clip_duration),
        "clip_frame_count": end - start + 1,
        "frame_coverage_ratio": (end - start + 1) / total_frames,
        "duration_coverage_ratio": float(duration_coverage),
        "all_run_count": len(runs),
        "all_runs": [
            {"start_raw_frame": a, "end_raw_frame": b, "length": b - a + 1}
            for a, b in runs
        ],
        "all_run_lengths": run_lengths,
        "run_selection_rule": "max_length_then_earliest_start",
        "selected_raw_frame_ids": selected,
        "selected_frame_count": len(selected),
        "selected_device_timestamps_ns": [
            int(frame_timestamps[index]) for index in selected
        ],
        "trajectory_failures_within_selected": sum(
            "outside_trajectory_span" in item["reasons"]
            or "trajectory_time_gap" in item["reasons"]
            for item in selected_diagnostics
        ),
        "calibration_failures_within_selected": sum(
            "calibration_time_gap" in item["reasons"]
            for item in selected_diagnostics
        ),
        "support_failure_counts_all_frames": dict(Counter(
            reason
            for item in diagnostics
            for reason in item["reasons"]
        )),
        "qa_count": int(qa_count),
        "usable": True,
        "rejection_code": None,
        "rejection_reason": None,
    }


def inspect_direct_gt(archive: zipfile.ZipFile) -> dict[str, Any]:
    """Report direct-GT capabilities without treating empty views as missing.

    The 2D-box table has rows for visible instances rather than an explicit
    row for every annotated empty image.  Therefore, absence of a box row is
    not evidence that the raw frame lacks GT support.
    """
    names = set(archive.namelist())
    missing = sorted(REQUIRED_GT_MEMBERS - names)
    if missing:
        raise SceneAuditError(
            "direct_gt_member_missing",
            f"Missing required direct-GT members: {missing}",
        )
    try:
        instances = json.loads(archive.read("instances.json"))
        if not isinstance(instances, dict):
            raise TypeError(
                f"instances.json must be an object, got {type(instances).__name__}"
            )
        instance_ids = {str(value) for value in instances}
        bbox_ids = {
            str(row["object_uid"])
            for row in csv_rows(archive, "3d_bounding_box.csv")
            if row.get("object_uid")
        }
        pose_ids = {
            str(row["object_uid"])
            for row in csv_rows(archive, "scene_objects.csv")
            if row.get("object_uid")
        }
    except SceneAuditError:
        raise
    except Exception as error:
        raise SceneAuditError(
            "direct_gt_parse_error",
            f"Could not parse direct object identity/geometry tables: {error}",
        ) from error
    joined_ids = instance_ids & bbox_ids & pose_ids
    if not joined_ids:
        raise SceneAuditError(
            "direct_gt_object_join_empty",
            "No object ID joins across instances.json, 3d_bounding_box.csv, "
            "and scene_objects.csv",
        )
    rgb_timestamps = set()
    visible_rows = 0
    for row in csv_rows(archive, "2d_bounding_box.csv"):
        if row.get("stream_id") != RGB_STREAM_ID:
            continue
        rgb_timestamps.add(int(row["timestamp[ns]"]))
        visible_rows += 1
    object_pose_timestamps = set()
    dynamic_pose_rows = 0
    for row in csv_rows(archive, "scene_objects.csv"):
        timestamp = int(row["timestamp[ns]"])
        if timestamp != -1:
            object_pose_timestamps.add(timestamp)
            dynamic_pose_rows += 1
    return {
        "scene_geometry_direct_gt": True,
        "object_identity_direct_gt": True,
        "instance_object_count": len(instance_ids),
        "bbox_object_count": len(bbox_ids),
        "pose_object_count": len(pose_ids),
        "joined_direct_object_count": len(joined_ids),
        "instance_without_bbox_count": len(instance_ids - bbox_ids),
        "instance_without_pose_count": len(instance_ids - pose_ids),
        "direct_object_parse_errors": 0,
        "rgb_box_stream_id": RGB_STREAM_ID,
        "rgb_box_annotation_timestamps": len(rgb_timestamps),
        "rgb_box_rows": visible_rows,
        "dynamic_object_pose_timestamps": len(object_pose_timestamps),
        "dynamic_object_pose_rows": dynamic_pose_rows,
        "per_frame_hard_support_fields": ["trajectory", "calibration"],
        "observation_support_fields": ["rgb_boxes", "object_pose"],
        "empty_rgb_box_policy": (
            "valid_empty_observation_not_missing_frame_annotation"
        ),
    }


def rejection_for(error: BaseException) -> tuple[str, str]:
    if isinstance(error, SceneAuditError):
        return error.code, error.detail
    if isinstance(error, FileNotFoundError):
        return "missing_video", str(error)
    return "scene_processing_error", f"{type(error).__name__}: {error}"


def failed_row(scene_id: str, qa_count: int, error: BaseException) -> dict[str, Any]:
    code, detail = rejection_for(error)
    return {
        "scene_id": scene_id,
        "qa_count": int(qa_count),
        "usable": False,
        "rejection_code": code,
        "rejection_reason": detail,
    }


def percentile(values: Sequence[float], quantile: float) -> float | None:
    return float(np.percentile(values, quantile)) if values else None


def build_summary(
    rows: Sequence[dict[str, Any]],
    requested: int,
    fixture_ids: Sequence[str],
) -> dict[str, Any]:
    usable = [row for row in rows if row["usable"]]
    total_qa = sum(row["qa_count"] for row in rows)
    usable_qa = sum(row["qa_count"] for row in usable)
    frame_ratios = [row["frame_coverage_ratio"] for row in usable]
    duration_ratios = [row["duration_coverage_ratio"] for row in usable]
    available = {row["scene_id"] for row in usable}
    return {
        "requested_sequences": requested,
        "completed_sequences": len(rows),
        "usable_sequences": len(usable),
        "scene_retention_rate": len(usable) / requested if requested else 0.0,
        "requested_qa_rows": total_qa,
        "usable_qa_rows": usable_qa,
        "qa_retention_rate": usable_qa / total_qa if total_qa else 0.0,
        "frame_coverage_ratio_percentiles": {
            "p0": percentile(frame_ratios, 0),
            "p10": percentile(frame_ratios, 10),
            "p25": percentile(frame_ratios, 25),
            "p50": percentile(frame_ratios, 50),
            "p75": percentile(frame_ratios, 75),
            "p90": percentile(frame_ratios, 90),
            "p100": percentile(frame_ratios, 100),
        },
        "duration_coverage_ratio_percentiles": {
            "p0": percentile(duration_ratios, 0),
            "p10": percentile(duration_ratios, 10),
            "p25": percentile(duration_ratios, 25),
            "p50": percentile(duration_ratios, 50),
            "p75": percentile(duration_ratios, 75),
            "p90": percentile(duration_ratios, 90),
            "p100": percentile(duration_ratios, 100),
        },
        "selected_frame_count_distribution": dict(sorted(Counter(
            str(row["selected_frame_count"]) for row in usable
        ).items())),
        "failure_codes": dict(sorted(Counter(
            row["rejection_code"] for row in rows if not row["usable"]
        ).items())),
        "failure_details": dict(Counter(
            row["rejection_reason"] for row in rows if not row["usable"]
        )),
        "source_artifact_status_counts": {
            kind: dict(sorted(Counter(
                row.get("input_artifacts", {})
                .get(kind, {"status": "unresolved"})["status"]
                for row in rows
            ).items()))
            for kind in ("video", "groundtruth", "calibration")
        },
        "scene_input_artifacts": {
            row["scene_id"]: row.get("input_artifacts", {
                kind: {
                    "status": "unresolved",
                    "path": None,
                    "size_bytes": None,
                    "sha256": None,
                }
                for kind in ("video", "groundtruth", "calibration")
            })
            for row in rows
        },
        "t0_fixtures": {
            fixture: {
                "present_in_requested": any(
                    row["scene_id"] == fixture for row in rows
                ),
                "usable": fixture in available,
            }
            for fixture in fixture_ids
        },
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_value(row.get(key)) for key in fields})
    atomic_write_text(path, buffer.getvalue())


def matched_file_count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern))


def is_within(path: Path, root: Path) -> bool:
    """Python-3.10-compatible resolved containment check."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_output_collisions(
    outputs: Sequence[Path],
    protected_files: Sequence[Path],
    protected_roots: Sequence[Path],
) -> None:
    resolved_outputs = [path.resolve() for path in outputs]
    protected = {path.resolve() for path in protected_files}
    for output in resolved_outputs:
        if output in protected:
            raise ValueError(f"Output path collides with source file: {output}")
        for root in protected_roots:
            if is_within(output, root):
                raise ValueError(
                    f"Output path is inside protected source root {root}: {output}"
                )


def validate_global_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.base_interval) or args.base_interval <= 0:
        raise ValueError("--base-interval must be finite and > 0")
    if not (1 <= args.min_frames <= args.max_frames):
        raise ValueError("Expected 1 <= --min-frames <= --max-frames")
    if args.max_trajectory_error_ns < 0:
        raise ValueError("--max-trajectory-error-ns must be >= 0")
    if args.max_calibration_error_ns < 0:
        raise ValueError("--max-calibration-error-ns must be >= 0")
    if args.sequence_limit is not None and args.sequence_limit <= 0:
        raise ValueError("--sequence-limit must be positive")
    if args.output_json.resolve() == args.output_csv.resolve():
        raise ValueError("--output-json and --output-csv must differ")
    validate_output_collisions(
        (args.output_json, args.output_csv),
        (args.jsonl, args.sequences),
        (
            args.video_root,
            args.groundtruth_root,
            args.calibration_root,
        ),
    )
    for name in ("jsonl", "sequences"):
        path = getattr(args, name)
        if not path.is_file() or not os.access(path, os.R_OK):
            raise FileNotFoundError(
                f"Missing or unreadable global input --{name}: {path}"
            )
    for name in ("groundtruth_root", "calibration_root", "video_root"):
        path = getattr(args, name)
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            raise FileNotFoundError(
                f"Missing or unreadable global input --{name}: {path}"
            )
    for output in (args.output_json, args.output_csv):
        output.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(output.parent, os.W_OK | os.X_OK):
            raise PermissionError(f"Output parent is not writable: {output.parent}")
        if output.exists() and (
            not output.is_file() or not os.access(output, os.W_OK)
        ):
            raise PermissionError(f"Output is not writable: {output}")


def main() -> None:
    args = parse_args()
    validate_global_args(args)
    sequences = [
        value.strip()
        for value in args.sequences.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Sequence list contains duplicate IDs")
    if args.sequence_limit is not None:
        sequences = sequences[:args.sequence_limit]
    qa_by_scene, source_counts = load_qa(args.jsonl)
    fixture_ids = list(T0_FIXTURES["adt"])
    if args.require_t0_fixtures:
        missing = sorted(set(fixture_ids) - set(sequences))
        if missing:
            raise ValueError(f"Missing fixed T0 fixtures from request: {missing}")

    rows = []
    for index, sequence in enumerate(sequences, 1):
        qa_count = len(qa_by_scene.get(sequence, []))
        video_path = args.video_root / f"ADT_{sequence}_preview_rgb.mp4"
        gt_directory = args.groundtruth_root / sequence
        calibration_directory = args.calibration_root / sequence
        gt_files = sorted(gt_directory.glob("*_main_groundtruth.zip"))
        calibration_files = sorted(
            calibration_directory.glob("*_mps_slam_calibration.zip")
        )
        input_artifacts = {
            "video": candidate_artifact(
                [video_path] if video_path.is_file() else [],
                expected_path=video_path,
            ),
            "groundtruth": candidate_artifact(
                gt_files,
                expected_path=(
                    gt_directory
                    / f"ADT_{sequence}_main_groundtruth.zip"
                ),
            ),
            "calibration": candidate_artifact(
                calibration_files,
                expected_path=(
                    calibration_directory
                    / f"ADT_{sequence}_mps_slam_calibration.zip"
                ),
            ),
        }
        # A resolved symlink could place a candidate outside the declared root.
        # Such an output/source collision is a global invocation error, not a
        # scene-quality rejection, and therefore must not be swallowed below.
        validate_output_collisions(
            (args.output_json, args.output_csv),
            [
                path
                for record in input_artifacts.values()
                for path in artifact_paths(record)
            ],
            (),
        )
        try:
            hash_errors = [
                f"{kind}:{candidate.get('path')}:{candidate.get('error')}"
                for kind, record in input_artifacts.items()
                for candidate in (
                    record.get("candidates", [])
                    if record["status"] == "multiple"
                    else [record]
                )
                if candidate["status"] == "hash_error"
            ]
            if hash_errors:
                raise SceneAuditError(
                    "artifact_hash_error", "; ".join(hash_errors)
                )
            if input_artifacts["video"]["status"] == "missing":
                raise SceneAuditError(
                    "missing_video", f"Missing video: {video_path}"
                )
            if len(gt_files) != 1:
                raise SceneAuditError(
                    "groundtruth_file_count",
                    f"Expected exactly one groundtruth ZIP, found {len(gt_files)}"
                )
            if len(calibration_files) != 1:
                raise SceneAuditError(
                    "calibration_file_count",
                    "Expected exactly one calibration ZIP, "
                    f"found {len(calibration_files)}",
                )
            try:
                frame_timestamps = mp4_timestamps(video_path)
                total_frames, fps = mp4_video_metadata(video_path)
            except Exception as error:
                raise SceneAuditError(
                    "video_metadata_invalid",
                    f"Could not parse MP4 timestamps/metadata: {error}",
                ) from error
            if total_frames != len(frame_timestamps):
                raise SceneAuditError(
                    "video_timestamp_count_mismatch",
                    "MP4 frame/timestamp count mismatch: "
                    f"{total_frames} != {len(frame_timestamps)}",
                )
            calibrations = load_calibrations(calibration_files[0])
            with zipfile.ZipFile(gt_files[0]) as archive:
                direct_gt = inspect_direct_gt(archive)
                trajectory = load_trajectory(archive)
            row = analyze_arrays(
                scene_id=sequence,
                frame_timestamps=frame_timestamps,
                fps=fps,
                trajectory_timestamps=[
                    item["timestamp_ns"] for item in trajectory
                ],
                calibration_timestamps=[
                    item["timestamp_ns"] for item in calibrations
                ],
                qa_count=qa_count,
                base_interval=args.base_interval,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
                max_trajectory_error_ns=args.max_trajectory_error_ns,
                max_calibration_error_ns=args.max_calibration_error_ns,
            )
            row.update({
                "vsi_media": f"adt/ADT_{sequence}_preview_rgb.mp4",
                "video_path": str(video_path.resolve()),
                "groundtruth_path": str(gt_files[0].resolve()),
                "calibration_path": str(calibration_files[0].resolve()),
                "direct_gt_capabilities": direct_gt,
            })
        except Exception as error:
            row = failed_row(sequence, qa_count, error)
        row["input_artifacts"] = input_artifacts
        rows.append(row)
        print(f"[{index}/{len(sequences)}] {sequence}: "
              f"{'usable' if row['usable'] else row['rejection_reason']}",
              flush=True)

    summary = build_summary(rows, len(sequences), fixture_ids)
    if args.require_t0_fixtures:
        unusable_fixtures = [
            fixture for fixture, status in summary["t0_fixtures"].items()
            if not status["usable"]
        ]
        if unusable_fixtures:
            # Reports are still written below; exit is made non-zero afterward.
            fixture_error = (
                f"Fixed T0 fixtures are not usable: {unusable_fixtures}"
            )
        else:
            fixture_error = None
    else:
        fixture_error = None

    report = {
        "schema_version": SCHEMA_VERSION,
        "sampling_policy": SAMPLING_POLICY,
        "parameters": {
            "base_interval": args.base_interval,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "max_trajectory_error_ns": args.max_trajectory_error_ns,
            "max_calibration_error_ns": args.max_calibration_error_ns,
            "sequence_limit": args.sequence_limit,
            "require_t0_fixtures": args.require_t0_fixtures,
            "direct_gt_support_contract": (
                "trajectory+calibration are per-frame hard support; "
                "scene geometry/identity are scene-level; RGB boxes and "
                "object poses are observation-level and reported separately"
            ),
        },
        "provenance": {
            "inputs": {
                "jsonl": {
                    "path": str(args.jsonl.resolve()),
                    "sha256": file_sha256(args.jsonl),
                },
                "sequences": {
                    "path": str(args.sequences.resolve()),
                    "sha256": file_sha256(args.sequences),
                },
                "groundtruth_root": {
                    "path": str(args.groundtruth_root.resolve()),
                    "matched_file_count": matched_file_count(
                        args.groundtruth_root, "*/*_main_groundtruth.zip"
                    ),
                },
                "calibration_root": {
                    "path": str(args.calibration_root.resolve()),
                    "matched_file_count": matched_file_count(
                        args.calibration_root,
                        "*/*_mps_slam_calibration.zip",
                    ),
                },
                "video_root": {
                    "path": str(args.video_root.resolve()),
                    "matched_file_count": matched_file_count(
                        args.video_root, "ADT_*_preview_rgb.mp4"
                    ),
                },
            },
            "vsi_source_counts": dict(source_counts),
            "scene_input_artifacts": {
                row["scene_id"]: row["input_artifacts"]
                for row in rows
            },
        },
        "summary": summary,
        "scenes": rows,
    }
    atomic_write_text(
        args.output_json,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    write_csv(args.output_csv, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"JSON report: {args.output_json}")
    print(f"CSV report: {args.output_csv}")
    if fixture_error:
        raise SystemExit(fixture_error)


if __name__ == "__main__":
    main()
