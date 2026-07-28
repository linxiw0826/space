#!/usr/bin/env python3
"""Audit VSI ADT preview-video frames against source-native ADT GT timestamps."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", required=True, type=Path)
    p.add_argument("--groundtruth-root", required=True, type=Path)
    p.add_argument("--video-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def csv_rows(archive, name):
    with archive.open(name) as source:
        text = io.TextIOWrapper(source, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def nearest_differences(reference, query):
    reference = np.asarray(sorted(reference), dtype=np.int64)
    query = np.asarray(sorted(query), dtype=np.int64)
    if not len(reference) or not len(query):
        return np.asarray([], dtype=np.int64)
    indices = np.searchsorted(reference, query)
    right = np.clip(indices, 0, len(reference) - 1)
    left = np.clip(indices - 1, 0, len(reference) - 1)
    return np.minimum(
        np.abs(reference[right] - query),
        np.abs(reference[left] - query),
    )


def timestamp_summary(values):
    values = np.asarray(sorted(values), dtype=np.int64)
    if not len(values):
        return {
            "count": 0,
            "min_ns": None,
            "max_ns": None,
            "span_seconds": None,
            "median_interval_ns": None,
            "estimated_hz": None,
        }
    differences = np.diff(values)
    positive = differences[differences > 0]
    median_interval = (
        float(np.median(positive)) if len(positive) else None
    )
    return {
        "count": int(len(values)),
        "min_ns": int(values[0]),
        "max_ns": int(values[-1]),
        "span_seconds": float((values[-1] - values[0]) / 1e9),
        "median_interval_ns": median_interval,
        "estimated_hz": (
            float(1e9 / median_interval) if median_interval else None
        ),
    }


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    result = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    result["duration_seconds"] = (
        result["frame_count"] / result["fps"]
        if result["fps"] > 0
        else None
    )
    return result


def main():
    args = parse_args()
    sequences = [
        line.strip()
        for line in args.sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {}
    status = Counter()
    errors = {}
    nearest_all = []
    for index, sequence in enumerate(sequences, 1):
        try:
            sequence_dir = args.groundtruth_root / sequence
            candidates = sorted(sequence_dir.glob("*_main_groundtruth.zip"))
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one main_groundtruth ZIP, found {len(candidates)}"
                )
            video = (
                args.video_root / f"ADT_{sequence}_preview_rgb.mp4"
            )
            if not video.is_file():
                raise FileNotFoundError(f"Missing video: {video}")
            info = video_info(video)
            with zipfile.ZipFile(candidates[0]) as archive:
                trajectory = csv_rows(archive, "aria_trajectory.csv")
                boxes_2d = csv_rows(archive, "2d_bounding_box.csv")
                objects = json.loads(
                    archive.read("instances.json").decode("utf-8")
                )
                boxes_3d = csv_rows(archive, "3d_bounding_box.csv")
                object_poses = csv_rows(archive, "scene_objects.csv")
            trajectory_timestamps = {
                int(row["tracking_timestamp_us"]) * 1000
                for row in trajectory
            }
            bbox_timestamps = {
                int(row["timestamp[ns]"]) for row in boxes_2d
            }
            trajectory_time = timestamp_summary(trajectory_timestamps)
            bbox_time = timestamp_summary(bbox_timestamps)
            bbox_uids = {row["object_uid"] for row in boxes_2d}
            object_ids = set(objects)
            bbox3d_uids = {row["object_uid"] for row in boxes_3d}
            pose_uids = {row["object_uid"] for row in object_poses}
            differences = nearest_differences(
                trajectory_timestamps, bbox_timestamps
            )
            nearest_all.extend(differences.tolist())
            frame_count = info["frame_count"]
            trajectory_count = len(trajectory_timestamps)
            bbox_time_count = len(bbox_timestamps)
            if frame_count == trajectory_count:
                alignment = "video_equals_trajectory"
            elif frame_count == bbox_time_count:
                alignment = "video_equals_bbox_timestamps"
            elif abs(frame_count - trajectory_count) <= 1:
                alignment = "video_trajectory_off_by_one"
            else:
                alignment = "count_mismatch"
            status[alignment] += 1
            results[sequence] = {
                **info,
                "trajectory_timestamps": trajectory_count,
                "bbox_timestamps": bbox_time_count,
                "trajectory_time": trajectory_time,
                "bbox_time": bbox_time,
                "video_trajectory_duration_ratio": (
                    info["duration_seconds"]
                    / trajectory_time["span_seconds"]
                    if trajectory_time["span_seconds"]
                    else None
                ),
                "video_bbox_duration_ratio": (
                    info["duration_seconds"] / bbox_time["span_seconds"]
                    if bbox_time["span_seconds"]
                    else None
                ),
                "instances": len(object_ids),
                "bbox3d_uids": len(bbox3d_uids),
                "object_pose_uids": len(pose_uids),
                "visible_uids": len(bbox_uids),
                "unknown_bbox_uids": len(bbox_uids - object_ids),
                "missing_bbox3d_uids": len(object_ids - bbox3d_uids),
                "missing_pose_uids": len(object_ids - pose_uids),
                "nearest_bbox_trajectory_ns_median": (
                    float(np.median(differences))
                    if len(differences)
                    else None
                ),
                "nearest_bbox_trajectory_ns_max": (
                    int(differences.max()) if len(differences) else None
                ),
                "alignment": alignment,
            }
        except Exception as error:
            status["error"] += 1
            errors[sequence] = f"{type(error).__name__}: {error}"
        print(f"[{index}/{len(sequences)}] {sequence}", flush=True)

    nearest_all = np.asarray(nearest_all, dtype=np.int64)
    duration_ratios = np.asarray([
        item["video_trajectory_duration_ratio"]
        for item in results.values()
        if item["video_trajectory_duration_ratio"] is not None
    ])
    report = {
        "schema_version": "adt_video_groundtruth_alignment_audit_v1",
        "requested_sequences": len(sequences),
        "audited_sequences": len(results),
        "status": dict(status),
        "nearest_bbox_trajectory_ns": {
            "count": int(len(nearest_all)),
            "median": (
                float(np.median(nearest_all)) if len(nearest_all) else None
            ),
            "p99": (
                float(np.percentile(nearest_all, 99))
                if len(nearest_all)
                else None
            ),
            "max": int(nearest_all.max()) if len(nearest_all) else None,
        },
        "video_trajectory_duration_ratio": {
            "count": int(len(duration_ratios)),
            "median": (
                float(np.median(duration_ratios))
                if len(duration_ratios)
                else None
            ),
            "p01": (
                float(np.percentile(duration_ratios, 1))
                if len(duration_ratios)
                else None
            ),
            "p99": (
                float(np.percentile(duration_ratios, 99))
                if len(duration_ratios)
                else None
            ),
        },
        "errors": errors,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in (
        "requested_sequences",
        "audited_sequences",
        "status",
        "nearest_bbox_trajectory_ns",
        "video_trajectory_duration_ratio",
        "errors",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
