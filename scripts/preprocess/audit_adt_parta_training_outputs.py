#!/usr/bin/env python3
"""Audit referential, temporal, and geometric integrity of ADT Part-A outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-scenes", type=int)
    parser.add_argument("--expected-candidate-frames", type=int, default=32)
    parser.add_argument(
        "--max-trajectory-error-ns", type=int, default=5_000_000
    )
    parser.add_argument(
        "--max-calibration-error-ns", type=int, default=50_000_000
    )
    return parser.parse_args()


def rows(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    args = parse_args()
    scene_path = args.input_dir / "adt_scene_states.jsonl"
    frame_path = args.input_dir / "adt_frame_states.jsonl"
    qa_path = args.input_dir / "adt_qa_train.jsonl"
    errors = []
    stats = Counter()
    node_ids_by_scene = {}
    frame_keys = set()
    frames_per_scene = Counter()
    trajectory_errors = []
    calibration_errors = []
    bbox_errors = []
    object_pose_errors = []
    determinants = []

    for line_number, row in rows(scene_path):
        scene = row["scene_id"]
        node_ids = [node["object_id"] for node in row["nodes"]]
        if scene in node_ids_by_scene:
            errors.append(f"duplicate scene: {scene}")
        if len(node_ids) != len(set(node_ids)):
            errors.append(f"duplicate node ID: {scene}")
        node_ids_by_scene[scene] = set(node_ids)
        stats["scenes"] += 1
        stats["nodes"] += len(node_ids)
        stats["dynamic_nodes"] += sum(
            node.get("motion_type") != "static" for node in row["nodes"]
        )

    for line_number, row in rows(frame_path):
        scene = row["scene_id"]
        frame_key = row["frame_key"]
        if frame_key in frame_keys:
            errors.append(f"duplicate frame key: {frame_key}")
        frame_keys.add(frame_key)
        frames_per_scene[scene] += 1
        if scene not in node_ids_by_scene:
            errors.append(f"frame references unknown scene: {frame_key}")
            known_nodes = set()
        else:
            known_nodes = node_ids_by_scene[scene]
        rotation = np.asarray(
            row["rotation_world_from_camera"], dtype=np.float64
        )
        determinant = float(np.linalg.det(rotation))
        determinants.append(determinant)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or abs(determinant - 1.0) > 1e-5
            or np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-5
        ):
            errors.append(f"invalid camera rotation: {frame_key}")
        translation = np.asarray(
            row["translation_world_from_camera_m"], dtype=np.float64
        )
        if translation.shape != (3,) or not np.isfinite(translation).all():
            errors.append(f"invalid camera translation: {frame_key}")
        trajectory_error = row["trajectory_timestamp_error_ns"]
        calibration_error = row["calibration_timestamp_error_ns"]
        trajectory_errors.append(trajectory_error)
        calibration_errors.append(calibration_error)
        if trajectory_error > args.max_trajectory_error_ns:
            errors.append(
                f"trajectory timestamp error: {frame_key}={trajectory_error}"
            )
        if calibration_error > args.max_calibration_error_ns:
            errors.append(
                f"calibration timestamp error: "
                f"{frame_key}={calibration_error}"
            )
        for visible in row["visible_nodes"]:
            stats["visible_node_observations"] += 1
            if visible["object_id"] not in known_nodes:
                errors.append(
                    f"unknown visible node: {frame_key}/"
                    f"{visible['object_id']}"
                )
            bbox = visible["bbox_xyxy"]
            if not (
                len(bbox) == 4
                and bbox[0] <= bbox[2]
                and bbox[1] <= bbox[3]
                and all(math.isfinite(value) for value in bbox)
            ):
                errors.append(f"invalid bbox: {frame_key}")
            center = np.asarray(
                visible["center_camera_m"], dtype=np.float64
            )
            if center.shape != (3,) or not np.isfinite(center).all():
                errors.append(f"invalid camera-space center: {frame_key}")
            elif center[2] > 0:
                stats["visible_center_in_front"] += 1
            bbox_errors.append(visible["frame_timestamp_error_ns"])
            object_pose_errors.append(
                visible["object_pose_timestamp_error_ns"]
            )
        stats["frames"] += 1

    for scene, count in frames_per_scene.items():
        if count != args.expected_candidate_frames:
            errors.append(
                f"candidate frame count: {scene}={count}, "
                f"expected={args.expected_candidate_frames}"
            )

    for line_number, row in rows(qa_path):
        scene = row["scene_id"]
        if scene not in node_ids_by_scene:
            errors.append(
                f"QA references unknown scene: line {line_number}/{scene}"
            )
        candidate_keys = row["candidate_frame_keys"]
        if len(candidate_keys) != args.expected_candidate_frames:
            errors.append(
                f"QA candidate count: line {line_number}/"
                f"{len(candidate_keys)}"
            )
        unknown = set(candidate_keys) - frame_keys
        if unknown:
            errors.append(
                f"QA references unknown frames: line {line_number}/"
                f"{sorted(unknown)[:3]}"
            )
        stats["qa_rows"] += 1

    if args.expected_scenes is not None:
        if stats["scenes"] != args.expected_scenes:
            errors.append(
                f"scene count={stats['scenes']}, "
                f"expected={args.expected_scenes}"
            )
        expected_frames = (
            args.expected_scenes * args.expected_candidate_frames
        )
        if stats["frames"] != expected_frames:
            errors.append(
                f"frame count={stats['frames']}, "
                f"expected={expected_frames}"
            )

    visible = max(stats["visible_node_observations"], 1)
    report = {
        "schema_version": "adt_training_output_audit_v1",
        **dict(stats),
        "frames_per_scene_distribution": dict(
            Counter(frames_per_scene.values())
        ),
        "visible_center_in_front_rate": (
            stats["visible_center_in_front"] / visible
        ),
        "camera_rotation_determinant": {
            "min": min(determinants) if determinants else None,
            "median": percentile(determinants, 50),
            "max": max(determinants) if determinants else None,
        },
        "timestamp_error_ns": {
            "trajectory_median": percentile(trajectory_errors, 50),
            "trajectory_p99": percentile(trajectory_errors, 99),
            "trajectory_max": max(trajectory_errors, default=None),
            "calibration_median": percentile(calibration_errors, 50),
            "calibration_p99": percentile(calibration_errors, 99),
            "calibration_max": max(calibration_errors, default=None),
            "bbox_median": percentile(bbox_errors, 50),
            "bbox_p99": percentile(bbox_errors, 99),
            "bbox_max": max(bbox_errors, default=None),
            "object_pose_median": percentile(object_pose_errors, 50),
            "object_pose_p99": percentile(object_pose_errors, 99),
            "object_pose_max": max(object_pose_errors, default=None),
        },
        "timestamp_thresholds_ns": {
            "trajectory": args.max_trajectory_error_ns,
            "calibration": args.max_calibration_error_ns,
        },
        "errors": errors[:1000],
        "error_count": len(errors),
        "all_valid": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(f"ADT output audit failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
