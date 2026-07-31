#!/usr/bin/env python3
"""Build directly aligned ADT Part-A scene, frame, and QA training data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (
    GUIDE_EXACT_SAMPLING_POLICY,
    guide_sampling_binding_sha256,
)
from src.adt_gt_supported_clip import (
    GT_SUPPORTED_CLIP_POLICY,
    LEGACY_POLICY,
    WHOLE_MP4_POLICY,
    build_support_certificate,
    supported_clip_sampling,
)


RGB_STREAM_ID = "214-1"
GUIDE_EXACT_POLICY = GT_SUPPORTED_CLIP_POLICY
FROZEN_MAX_OBJECT_POSE_ERROR_NS = 5_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--groundtruth-root", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-output", required=True, type=Path)
    parser.add_argument(
        "--sampling-policy",
        choices=(GUIDE_EXACT_POLICY, WHOLE_MP4_POLICY, LEGACY_POLICY),
        default=GUIDE_EXACT_POLICY,
        help=(
            "Part A defaults to D-59 clip-local exact GUIDE raw MP4 IDs. "
            "Whole-MP4 and legacy policies are explicit historical modes and "
            "are rejected by the formal canonical/exact contract."
        ),
    )
    parser.add_argument("--candidate-frames", type=int, default=32)
    parser.add_argument("--base-interval", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument(
        "--max-trajectory-error-ns", type=int, default=5_000_000
    )
    parser.add_argument(
        "--max-calibration-error-ns", type=int, default=50_000_000
    )
    parser.add_argument(
        "--max-object-pose-error-ns", type=int, default=5_000_000
    )
    parser.add_argument("--sequence-limit", type=int)
    return parser.parse_args()


def csv_rows(archive, name):
    with archive.open(name) as source:
        text = io.TextIOWrapper(source, encoding="utf-8", newline="")
        yield from csv.DictReader(text)


def quaternion_matrix(w, x, y, z):
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = float(np.dot(quaternion, quaternion))
    if norm < 1e-16:
        raise ValueError("Degenerate quaternion")
    quaternion *= np.sqrt(2.0 / norm)
    outer = np.outer(quaternion, quaternion)
    return np.asarray([
        [
            1.0 - outer[2, 2] - outer[3, 3],
            outer[1, 2] - outer[3, 0],
            outer[1, 3] + outer[2, 0],
        ],
        [
            outer[1, 2] + outer[3, 0],
            1.0 - outer[1, 1] - outer[3, 3],
            outer[2, 3] - outer[1, 0],
        ],
        [
            outer[1, 3] - outer[2, 0],
            outer[2, 3] + outer[1, 0],
            1.0 - outer[1, 1] - outer[2, 2],
        ],
    ])


def nearest_index(sorted_values, query):
    index = bisect_left(sorted_values, query)
    candidates = []
    if index < len(sorted_values):
        candidates.append(index)
    if index:
        candidates.append(index - 1)
    return min(candidates, key=lambda i: abs(sorted_values[i] - query))


def mp4_timestamps(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=description",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    description = (
        json.loads(probe.stdout)
        .get("format", {})
        .get("tags", {})
        .get("description")
    )
    values = json.loads(description) if description else []
    if not isinstance(values, list) or not values:
        raise ValueError(f"No MP4 device timestamps: {path}")
    return [int(value) for value in values]


def mp4_video_metadata(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if denominator == 0:
        raise ValueError(f"Invalid MP4 frame rate: {path}")
    return int(stream["nb_frames"]), numerator / denominator


def select_candidate_indices(
    timestamps,
    trajectory_timestamps,
    calibration_timestamps,
    count,
    max_trajectory_error_ns,
    max_calibration_error_ns,
):
    valid = []
    rejected = Counter()
    for index, timestamp in enumerate(timestamps):
        if not (
            trajectory_timestamps[0]
            <= timestamp
            <= trajectory_timestamps[-1]
        ):
            rejected["outside_trajectory_span"] += 1
            continue
        trajectory_index = nearest_index(
            trajectory_timestamps, timestamp
        )
        trajectory_error = abs(
            trajectory_timestamps[trajectory_index] - timestamp
        )
        if trajectory_error > max_trajectory_error_ns:
            rejected["trajectory_time_gap"] += 1
            continue
        calibration_index = nearest_index(
            calibration_timestamps, timestamp
        )
        calibration_error = abs(
            calibration_timestamps[calibration_index] - timestamp
        )
        if calibration_error > max_calibration_error_ns:
            rejected["calibration_time_gap"] += 1
            continue
        valid.append(index)
    if not valid:
        raise ValueError("No video frames pass temporal alignment thresholds")
    positions = np.linspace(0, len(valid) - 1, min(count, len(valid)))
    return (
        [valid[int(round(position))] for position in positions],
        rejected,
    )


def validate_exact_indices(
    frame_timestamps,
    selected_indices,
    trajectory_timestamps,
    calibration_timestamps,
    max_trajectory_error_ns,
    max_calibration_error_ns,
):
    """Validate preselected raw IDs without replacing or resampling any ID."""
    failures = []
    diagnostics = []
    for frame_index in selected_indices:
        if frame_index < 0 or frame_index >= len(frame_timestamps):
            failures.append({
                "frame_index": frame_index,
                "reason": "raw_frame_out_of_range",
            })
            continue
        timestamp = frame_timestamps[frame_index]
        if not (
            trajectory_timestamps[0]
            <= timestamp
            <= trajectory_timestamps[-1]
        ):
            failures.append({
                "frame_index": frame_index,
                "device_timestamp_ns": timestamp,
                "reason": "outside_trajectory_span",
            })
            continue
        trajectory_index = nearest_index(trajectory_timestamps, timestamp)
        calibration_index = nearest_index(calibration_timestamps, timestamp)
        trajectory_error = abs(
            trajectory_timestamps[trajectory_index] - timestamp
        )
        calibration_error = abs(
            calibration_timestamps[calibration_index] - timestamp
        )
        reasons = []
        if trajectory_error > max_trajectory_error_ns:
            reasons.append("trajectory_time_gap")
        if calibration_error > max_calibration_error_ns:
            reasons.append("calibration_time_gap")
        if reasons:
            failures.append({
                "frame_index": frame_index,
                "device_timestamp_ns": timestamp,
                "reason": "+".join(reasons),
                "trajectory_timestamp_error_ns": trajectory_error,
                "calibration_timestamp_error_ns": calibration_error,
            })
            continue
        diagnostics.append({
            "frame_index": frame_index,
            "device_timestamp_ns": timestamp,
            "trajectory_timestamp_error_ns": trajectory_error,
            "calibration_timestamp_error_ns": calibration_error,
        })
    if failures:
        raise ValueError(
            "Exact GUIDE raw frames failed temporal alignment; "
            f"failures={json.dumps(failures, ensure_ascii=False)}"
        )
    if [item["frame_index"] for item in diagnostics] != list(selected_indices):
        raise AssertionError("Exact raw frame validation changed selected IDs")
    return diagnostics


def load_qa(path):
    result = defaultdict(list)
    source_counts = Counter()
    with path.open(encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            row = json.loads(line)
            media = row.get("video") or ""
            source_name = media.split("/", 1)[0] if "/" in media else ""
            source_counts[source_name] += 1
            if not media.startswith("adt/"):
                continue
            name = Path(media).stem
            if name.startswith("ADT_"):
                name = name[4:]
            if name.endswith("_preview_rgb"):
                name = name[: -len("_preview_rgb")]
            result[name].append({
                "vsi_row_index": row_index,
                "vsi_media": media,
                "question_type": row.get("question_type"),
                "conversations": row["conversations"],
            })
    return result, source_counts


def load_calibrations(path):
    records = []
    with zipfile.ZipFile(path) as archive:
        with archive.open("online_calibration.jsonl") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                camera = next(
                    item
                    for item in row["CameraCalibrations"]
                    if item["Label"] == "camera-rgb"
                )
                quaternion = camera["T_Device_Camera"]["UnitQuaternion"]
                records.append({
                    "timestamp_ns": int(row["tracking_timestamp_us"]) * 1000,
                    "translation_device_from_camera_m": np.asarray(
                        camera["T_Device_Camera"]["Translation"],
                        dtype=np.float64,
                    ),
                    "rotation_device_from_camera": quaternion_matrix(
                        float(quaternion[0]),
                        *map(float, quaternion[1]),
                    ),
                    "projection": camera["Projection"],
                    "calibrated": bool(camera["Calibrated"]),
                })
    records.sort(key=lambda item: item["timestamp_ns"])
    return records


def load_trajectory(archive):
    records = []
    for row in csv_rows(archive, "aria_trajectory.csv"):
        records.append({
            "timestamp_ns": int(row["tracking_timestamp_us"]) * 1000,
            "translation_world_from_device_m": np.asarray([
                float(row["tx_world_device"]),
                float(row["ty_world_device"]),
                float(row["tz_world_device"]),
            ]),
            "rotation_world_from_device": quaternion_matrix(
                float(row["qw_world_device"]),
                float(row["qx_world_device"]),
                float(row["qy_world_device"]),
                float(row["qz_world_device"]),
            ),
            "linear_velocity_device_mps": [
                float(row["device_linear_velocity_x_device"]),
                float(row["device_linear_velocity_y_device"]),
                float(row["device_linear_velocity_z_device"]),
            ],
            "angular_velocity_device_rps": [
                float(row["angular_velocity_x_device"]),
                float(row["angular_velocity_y_device"]),
                float(row["angular_velocity_z_device"]),
            ],
            "quality_score": float(row["quality_score"]),
        })
    records.sort(key=lambda item: item["timestamp_ns"])
    return records


def load_object_geometry(archive):
    instances = json.loads(archive.read("instances.json"))
    boxes = {}
    for row in csv_rows(archive, "3d_bounding_box.csv"):
        object_id = row["object_uid"]
        minimum = np.asarray([
            float(row["p_local_obj_xmin[m]"]),
            float(row["p_local_obj_ymin[m]"]),
            float(row["p_local_obj_zmin[m]"]),
        ])
        maximum = np.asarray([
            float(row["p_local_obj_xmax[m]"]),
            float(row["p_local_obj_ymax[m]"]),
            float(row["p_local_obj_zmax[m]"]),
        ])
        boxes[object_id] = {
            "center_local_m": 0.5 * (minimum + maximum),
            "extent_m": maximum - minimum,
        }
    poses = defaultdict(list)
    for row in csv_rows(archive, "scene_objects.csv"):
        object_id = row["object_uid"]
        poses[object_id].append({
            "timestamp_ns": int(row["timestamp[ns]"]),
            "translation_world_from_object_m": np.asarray([
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ]),
            "rotation_world_from_object": quaternion_matrix(
                float(row["q_wo_w"]),
                float(row["q_wo_x"]),
                float(row["q_wo_y"]),
                float(row["q_wo_z"]),
            ),
        })
    for values in poses.values():
        values.sort(key=lambda item: item["timestamp_ns"])
    usable = sorted(set(instances) & set(boxes) & set(poses), key=int)
    return instances, boxes, poses, usable


def object_pose_at(poses, timestamp):
    if len(poses) == 1 and poses[0]["timestamp_ns"] == -1:
        return poses[0], 0
    timestamps = [item["timestamp_ns"] for item in poses]
    index = nearest_index(timestamps, timestamp)
    return poses[index], abs(timestamps[index] - timestamp)


def node_state(object_id, box, pose):
    rotation = pose["rotation_world_from_object"]
    translation = pose["translation_world_from_object_m"]
    center = translation + rotation @ box["center_local_m"]
    return center, rotation


def compact_node(object_id, metadata, box, pose, pose_error_ns, max_error_ns):
    geometry_valid = object_geometry_is_valid(
        metadata, pose_error_ns, max_error_ns
    )
    center, rotation = (
        node_state(object_id, box, pose)
        if geometry_valid else (None, None)
    )
    return {
        "object_id": object_id,
        "instance_name": metadata.get("instance_name"),
        "prototype_name": metadata.get("prototype_name"),
        "category": metadata.get("category"),
        "category_uid": metadata.get("category_uid"),
        "motion_type": metadata.get("motion_type"),
        "rigidity": metadata.get("rigidity"),
        "center_world_m": center.tolist() if center is not None else None,
        "extent_m": (
            box["extent_m"].tolist() if geometry_valid else None
        ),
        "rotation_world_from_object": (
            rotation.tolist() if rotation is not None else None
        ),
        "geometry_valid": geometry_valid,
        "reference_pose_timestamp_error_ns": int(pose_error_ns),
        "identity_source": "adt_object_uid",
    }


def build_edges(nodes, knn):
    geometry_nodes = [
        node for node in nodes if node.get("geometry_valid", True)
    ]
    if len(geometry_nodes) < 2:
        return []
    centers = np.asarray([node["center_world_m"] for node in geometry_nodes])
    edges = []
    for index, source in enumerate(geometry_nodes):
        delta = centers - centers[index]
        distance = np.linalg.norm(delta, axis=1)
        neighbors = np.argsort(distance)[
            1 : 1 + min(knn, len(geometry_nodes) - 1)
        ]
        for target in neighbors:
            edges.append({
                "source_id": source["object_id"],
                "target_id": geometry_nodes[target]["object_id"],
                "delta_world_m": delta[target].tolist(),
                "distance_m": float(distance[target]),
            })
    return edges


def load_rgb_boxes(archive, candidate_timestamps):
    boxes = defaultdict(list)
    timestamp_mapping = {}
    for row in csv_rows(archive, "2d_bounding_box.csv"):
        if row["stream_id"] != RGB_STREAM_ID:
            continue
        timestamp = int(row["timestamp[ns]"])
        if timestamp not in timestamp_mapping:
            index = nearest_index(candidate_timestamps, timestamp)
            nearest = candidate_timestamps[index]
            timestamp_mapping[timestamp] = (
                nearest if abs(nearest - timestamp) <= 1_000_000 else None
            )
        target = timestamp_mapping[timestamp]
        if target is None:
            continue
        boxes[target].append({
            "object_id": row["object_uid"],
            "bbox_xyxy": [
                int(row["x_min[pixel]"]),
                int(row["y_min[pixel]"]),
                int(row["x_max[pixel]"]),
                int(row["y_max[pixel]"]),
            ],
            "visibility_ratio": float(row["visibility_ratio[%]"]),
            "source_timestamp_ns": timestamp,
            "frame_timestamp_error_ns": abs(target - timestamp),
        })
    return boxes


def write_archive_atomic(archive_output, paths):
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{archive_output.name}.",
        suffix=".tmp",
        dir=archive_output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=path.name)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, archive_output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        archive_output.unlink(missing_ok=True)
        raise


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_geometry_is_valid(instance, pose_error_ns, max_error_ns):
    """Static geometry is timeless; dynamic geometry requires a close pose."""
    return (
        instance.get("motion_type") == "static"
        or int(pose_error_ns) <= int(max_error_ns)
    )


def main():
    args = parse_args()
    if args.max_object_pose_error_ns != FROZEN_MAX_OBJECT_POSE_ERROR_NS:
        raise ValueError(
            "Formal ADT object-pose threshold is frozen at 5000000 ns"
        )
    # A previous successful archive must never survive a fresh failed run.
    args.archive_output.unlink(missing_ok=True)
    sequences = [
        line.strip()
        for line in args.sequences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.sequence_limit is not None:
        sequences = sequences[: args.sequence_limit]
    qa_by_scene, source_counts = load_qa(args.jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / "adt_scene_states.jsonl"
    frame_path = args.output_dir / "adt_frame_states.jsonl"
    qa_path = args.output_dir / "adt_qa_train.jsonl"
    certificate_path = args.output_dir / "adt_support_certificates.jsonl"
    report_path = args.output_dir / "adt_alignment_report.json"
    stats = Counter()
    errors = []
    clip_coverage = []

    with (
        scene_path.open("w", encoding="utf-8") as scene_out,
        frame_path.open("w", encoding="utf-8") as frame_out,
        qa_path.open("w", encoding="utf-8") as qa_out,
        certificate_path.open("w", encoding="utf-8") as certificate_out,
    ):
        for sequence_index, sequence in enumerate(sequences, 1):
            try:
                sequence_stats = Counter()
                scene_buffer = io.StringIO()
                frame_buffer = io.StringIO()
                qa_buffer = io.StringIO()
                certificate_buffer = io.StringIO()
                video_path = (
                    args.video_root / f"ADT_{sequence}_preview_rgb.mp4"
                )
                groundtruth_files = sorted(
                    (args.groundtruth_root / sequence)
                    .glob("*_main_groundtruth.zip")
                )
                calibration_files = sorted(
                    (args.calibration_root / sequence)
                    .glob("*_mps_slam_calibration.zip")
                )
                if len(groundtruth_files) != 1:
                    raise ValueError("Expected one main_groundtruth ZIP")
                if len(calibration_files) != 1:
                    raise ValueError("Expected one calibration ZIP")
                frame_timestamps = mp4_timestamps(video_path)
                total_frames, fps = mp4_video_metadata(video_path)
                if len(frame_timestamps) != total_frames:
                    raise ValueError(
                        "MP4 device timestamp count does not match raw frame "
                        f"count: timestamps={len(frame_timestamps)}, "
                        f"frames={total_frames}"
                    )
                calibrations = load_calibrations(calibration_files[0])
                calibration_timestamps = [
                    item["timestamp_ns"] for item in calibrations
                ]
                with zipfile.ZipFile(groundtruth_files[0]) as archive:
                    trajectory = load_trajectory(archive)
                    trajectory_timestamps = [
                        item["timestamp_ns"] for item in trajectory
                    ]
                    clip_contract = None
                    support_certificate = None
                    if args.sampling_policy == GUIDE_EXACT_POLICY:
                        clip_contract = supported_clip_sampling(
                            frame_timestamps,
                            trajectory_timestamps,
                            calibration_timestamps,
                            fps,
                            base_interval=args.base_interval,
                            min_frames=args.min_frames,
                            max_frames=args.max_frames,
                            max_trajectory_error_ns=(
                                args.max_trajectory_error_ns
                            ),
                            max_calibration_error_ns=(
                                args.max_calibration_error_ns
                            ),
                        )
                        candidate_indices = clip_contract[
                            "raw_frame_indices"
                        ]
                        support_certificate = build_support_certificate(
                            scene_id=sequence,
                            vsi_media=(
                                f"adt/ADT_{sequence}_preview_rgb.mp4"
                            ),
                            frame_timestamps=frame_timestamps,
                            fps=fps,
                            support_mask=clip_contract["support_mask"],
                            max_trajectory_error_ns=(
                                args.max_trajectory_error_ns
                            ),
                            max_calibration_error_ns=(
                                args.max_calibration_error_ns
                            ),
                        )
                        validate_exact_indices(
                            frame_timestamps,
                            candidate_indices,
                            trajectory_timestamps,
                            calibration_timestamps,
                            args.max_trajectory_error_ns,
                            args.max_calibration_error_ns,
                        )
                    elif args.sampling_policy == WHOLE_MP4_POLICY:
                        from src.parta_data_contract import guide_frame_indices
                        candidate_indices = guide_frame_indices(
                            total_frames,
                            fps,
                            base_interval=args.base_interval,
                            min_frames=args.min_frames,
                            max_frames=args.max_frames,
                        )
                        validate_exact_indices(
                            frame_timestamps,
                            candidate_indices,
                            trajectory_timestamps,
                            calibration_timestamps,
                            args.max_trajectory_error_ns,
                            args.max_calibration_error_ns,
                        )
                    else:
                        (
                            candidate_indices,
                            temporal_rejections,
                        ) = select_candidate_indices(
                            frame_timestamps,
                            trajectory_timestamps,
                            calibration_timestamps,
                            args.candidate_frames,
                            args.max_trajectory_error_ns,
                            args.max_calibration_error_ns,
                        )
                        sequence_stats.update(temporal_rejections)
                    candidate_timestamps = [
                        frame_timestamps[index] for index in candidate_indices
                    ]
                    instances, boxes, poses, usable = load_object_geometry(
                        archive
                    )
                    rgb_boxes = load_rgb_boxes(
                        archive, candidate_timestamps
                    )

                nodes = []
                for object_id in usable:
                    reference_pose, reference_pose_error = object_pose_at(
                        poses[object_id], candidate_timestamps[0]
                    )
                    nodes.append(compact_node(
                        object_id,
                        instances[object_id],
                        boxes[object_id],
                        reference_pose,
                        reference_pose_error,
                        args.max_object_pose_error_ns,
                    ))
                node_ids = {node["object_id"] for node in nodes}
                scene_buffer.write(json.dumps({
                    "schema_version": "adt_scene_state_v1",
                    "scene_id": sequence,
                    "vsi_media": (
                        f"adt/ADT_{sequence}_preview_rgb.mp4"
                    ),
                    "nodes": nodes,
                    "edges_knn": build_edges(nodes, args.knn),
                    "coordinate_frame": "adt_world_meters",
                    "supervision_tier": "gold",
                    "identity_source": "adt_object_uid_direct_join",
                }, ensure_ascii=False) + "\n")
                sequence_stats["scenes"] += 1
                sequence_stats["nodes"] += len(nodes)
                sequence_stats["dynamic_nodes"] += sum(
                    node["motion_type"] != "static" for node in nodes
                )
                frame_keys = []
                sampling_provenance = {
                    "sampling_policy": args.sampling_policy,
                    "total_frames": total_frames,
                    "fps": fps,
                    "base_interval": (
                        args.base_interval
                        if args.sampling_policy == GUIDE_EXACT_POLICY
                        else None
                    ),
                    "min_frames": (
                        args.min_frames
                        if args.sampling_policy == GUIDE_EXACT_POLICY
                        else None
                    ),
                    "max_frames": (
                        args.max_frames
                        if args.sampling_policy == GUIDE_EXACT_POLICY
                        else None
                    ),
                    "raw_frame_indices": candidate_indices,
                }
                if clip_contract is not None:
                    sampling_provenance["clip_provenance"] = {
                        "whole_video_total_frames": total_frames,
                        "whole_video_fps_hex": float(fps).hex(),
                        "whole_video_start_device_timestamp_ns": int(
                            frame_timestamps[0]
                        ),
                        "whole_video_end_device_timestamp_ns": int(
                            frame_timestamps[-1]
                        ),
                        "clip_start_raw_frame": clip_contract[
                            "clip_start_raw_frame"
                        ],
                        "clip_end_raw_frame": clip_contract[
                            "clip_end_raw_frame"
                        ],
                        "clip_start_device_timestamp_ns": clip_contract[
                            "clip_start_device_timestamp_ns"
                        ],
                        "clip_end_device_timestamp_ns": clip_contract[
                            "clip_end_device_timestamp_ns"
                        ],
                        "clip_frame_count": clip_contract[
                            "clip_frame_count"
                        ],
                        "hard_support_fields": clip_contract[
                            "hard_support_fields"
                        ],
                        "support_runs": clip_contract["support_runs"],
                        "tie_policy": clip_contract["tie_policy"],
                        "max_trajectory_error_ns": (
                            args.max_trajectory_error_ns
                        ),
                        "max_calibration_error_ns": (
                            args.max_calibration_error_ns
                        ),
                        "local_frame_indices": clip_contract[
                            "local_frame_indices"
                        ],
                        "selected_device_timestamps_ns": clip_contract[
                            "selected_device_timestamps_ns"
                        ],
                        "support_certificate": support_certificate,
                    }

                for candidate_rank, (frame_index, timestamp) in enumerate(
                    zip(candidate_indices, candidate_timestamps)
                ):
                    trajectory_index = nearest_index(
                        trajectory_timestamps, timestamp
                    )
                    calibration_index = nearest_index(
                        calibration_timestamps, timestamp
                    )
                    trajectory_record = trajectory[trajectory_index]
                    calibration = calibrations[calibration_index]
                    rotation_world_from_device = trajectory_record[
                        "rotation_world_from_device"
                    ]
                    translation_world_from_device = trajectory_record[
                        "translation_world_from_device_m"
                    ]
                    rotation_device_from_camera = calibration[
                        "rotation_device_from_camera"
                    ]
                    translation_device_from_camera = calibration[
                        "translation_device_from_camera_m"
                    ]
                    rotation_world_from_camera = (
                        rotation_world_from_device
                        @ rotation_device_from_camera
                    )
                    translation_world_from_camera = (
                        translation_world_from_device
                        + rotation_world_from_device
                        @ translation_device_from_camera
                    )
                    visible_nodes = []
                    for visible in rgb_boxes.get(timestamp, []):
                        object_id = visible["object_id"]
                        if object_id not in node_ids:
                            sequence_stats["visible_unknown_node"] += 1
                            continue
                        pose, pose_error = object_pose_at(
                            poses[object_id], timestamp
                        )
                        is_static = (
                            instances[object_id].get("motion_type") == "static"
                        )
                        geometry_valid = object_geometry_is_valid(
                            instances[object_id],
                            pose_error,
                            args.max_object_pose_error_ns,
                        )
                        if geometry_valid:
                            center_world, rotation_world_from_object = (
                                node_state(object_id, boxes[object_id], pose)
                            )
                            center_camera = (
                                rotation_world_from_camera.T
                                @ (
                                    center_world
                                    - translation_world_from_camera
                                )
                            )
                        else:
                            center_world = None
                            rotation_world_from_object = None
                            center_camera = None
                        visible_nodes.append({
                            **visible,
                            "center_world_m": (
                                center_world.tolist()
                                if center_world is not None else None
                            ),
                            "rotation_world_from_object": (
                                rotation_world_from_object.tolist()
                                if rotation_world_from_object is not None
                                else None
                            ),
                            "center_camera_m": (
                                center_camera.tolist()
                                if center_camera is not None else None
                            ),
                            "camera_distance_m": (
                                float(np.linalg.norm(center_camera))
                                if center_camera is not None else None
                            ),
                            "object_pose_timestamp_error_ns": pose_error,
                            "object_geometry_valid": geometry_valid,
                            "object_pose_policy": (
                                "static_scene_pose"
                                if is_static
                                else "nearest_with_5ms_hard_mask"
                            ),
                            "motion_type": instances[object_id].get(
                                "motion_type"
                            ),
                        })
                    frame_key = f"{sequence}/{frame_index:06d}"
                    frame_keys.append(frame_key)
                    frame_buffer.write(json.dumps({
                        "schema_version": "adt_frame_state_v1",
                        "frame_key": frame_key,
                        "scene_id": sequence,
                        "vsi_media": (
                            f"adt/ADT_{sequence}_preview_rgb.mp4"
                        ),
                        "frame_index": frame_index,
                        "candidate_rank": candidate_rank,
                        "device_timestamp_ns": timestamp,
                        "rotation_world_from_camera": (
                            rotation_world_from_camera.tolist()
                        ),
                        "translation_world_from_camera_m": (
                            translation_world_from_camera.tolist()
                        ),
                        "rotation_camera_from_world": (
                            rotation_world_from_camera.T.tolist()
                        ),
                        "translation_camera_from_world_m": (
                            -rotation_world_from_camera.T
                            @ translation_world_from_camera
                        ).tolist(),
                        "rotation_world_from_device": (
                            rotation_world_from_device.tolist()
                        ),
                        "translation_world_from_device_m": (
                            translation_world_from_device.tolist()
                        ),
                        "linear_velocity_device_mps": trajectory_record[
                            "linear_velocity_device_mps"
                        ],
                        "angular_velocity_device_rps": trajectory_record[
                            "angular_velocity_device_rps"
                        ],
                        "trajectory_quality_score": trajectory_record[
                            "quality_score"
                        ],
                        "trajectory_timestamp_error_ns": abs(
                            trajectory_record["timestamp_ns"] - timestamp
                        ),
                        "calibration_timestamp_error_ns": abs(
                            calibration["timestamp_ns"] - timestamp
                        ),
                        "camera_projection": calibration["projection"],
                        "visible_nodes": visible_nodes,
                        "supervision_tier": "gold",
                    }, ensure_ascii=False) + "\n")
                    sequence_stats["frames"] += 1
                    sequence_stats["visible_node_observations"] += len(
                        visible_nodes
                    )

                sampling_provenance["sampling_binding_sha256"] = (
                    guide_sampling_binding_sha256(
                        source_dataset="adt",
                        scene_id=sequence,
                        vsi_media=(
                            f"adt/ADT_{sequence}_preview_rgb.mp4"
                        ),
                        frame_keys=frame_keys,
                        frame_indices=candidate_indices,
                        total_frames=total_frames,
                        fps=fps,
                        base_interval=args.base_interval,
                        min_frames=args.min_frames,
                        max_frames=args.max_frames,
                        sampling_policy=args.sampling_policy,
                        clip_provenance=sampling_provenance.get(
                            "clip_provenance"
                        ),
                    )
                    if args.sampling_policy == GUIDE_EXACT_POLICY
                    else None
                )
                if support_certificate is not None:
                    sampling_provenance[
                        "support_certificate_sha256"
                    ] = support_certificate["certificate_sha256"]
                    certificate_buffer.write(
                        json.dumps(
                            support_certificate, ensure_ascii=False
                        ) + "\n"
                    )
                for qa in qa_by_scene.get(sequence, []):
                    qa_buffer.write(json.dumps({
                        "schema_version": "adt_qa_train_v1",
                        "vsi_row_index": qa["vsi_row_index"],
                        "scene_id": sequence,
                        "candidate_frame_keys": frame_keys,
                        "candidate_frame_indices": candidate_indices,
                        "vsi_media": qa["vsi_media"],
                        "question_type": qa["question_type"],
                        "conversations": qa["conversations"],
                        "loss_masks": {
                            "qa": True,
                            "node_identity": True,
                            "scene_geometry": True,
                            "camera_geometry": True,
                            "node_dynamics": any(
                                node["motion_type"] != "static"
                                for node in nodes
                            ),
                        },
                        **sampling_provenance,
                    }, ensure_ascii=False) + "\n")
                    sequence_stats["qa_rows"] += 1
                if sequence not in qa_by_scene:
                    sequence_stats["scenes_without_qa"] += 1
                if clip_contract is not None:
                    clip_coverage.append({
                        "scene_id": sequence,
                        "whole_video_total_frames": total_frames,
                        "clip_start_raw_frame": clip_contract[
                            "clip_start_raw_frame"
                        ],
                        "clip_end_raw_frame": clip_contract[
                            "clip_end_raw_frame"
                        ],
                        "clip_frame_count": clip_contract[
                            "clip_frame_count"
                        ],
                        "frame_coverage_ratio": (
                            clip_contract["clip_frame_count"] / total_frames
                        ),
                        "selected_frame_count": len(candidate_indices),
                    })
                scene_out.write(scene_buffer.getvalue())
                frame_out.write(frame_buffer.getvalue())
                qa_out.write(qa_buffer.getvalue())
                certificate_out.write(certificate_buffer.getvalue())
                stats.update(sequence_stats)
            except Exception as error:
                stats["scene_errors"] += 1
                errors.append({
                    "scene": sequence,
                    "error": f"{type(error).__name__}: {error}",
                })
            print(
                f"[{sequence_index}/{len(sequences)}] {sequence}",
                flush=True,
            )

    report = {
        "schema_version": "adt_alignment_report_v1",
        **dict(stats),
        "requested_sequences": len(sequences),
        "completed_sequences": stats["scenes"],
        "adt_qa_rows_in_source": sum(
            len(rows) for rows in qa_by_scene.values()
        ),
        "source_counts": dict(source_counts),
        "errors": errors,
        "clip_coverage": clip_coverage,
        "support_certificate_registry": {
            "path": certificate_path.name,
            "sha256": file_sha256(certificate_path),
            "trust_stage": "finalizer_output_v1",
        },
        "coordinate_conventions": {
            "trajectory_transform": "world_from_device",
            "calibration_transform": "device_from_rgb_camera",
            "camera_transform": (
                "world_from_device @ device_from_rgb_camera"
            ),
            "object_transform": "world_from_object",
            "units": "meters",
            "timestamps": "device_time_nanoseconds",
        },
        "selection": {
            "policy": args.sampling_policy,
            "candidate_frames_per_scene": (
                args.candidate_frames
                if args.sampling_policy == LEGACY_POLICY
                else None
            ),
            "base_interval": args.base_interval,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
            "rgb_stream_id": RGB_STREAM_ID,
            "out_of_video_gt_policy": "discard_without_extrapolation",
            "max_trajectory_error_ns": args.max_trajectory_error_ns,
            "max_calibration_error_ns": args.max_calibration_error_ns,
            "max_object_pose_error_ns": args.max_object_pose_error_ns,
        },
        "files": [
            scene_path.name,
            frame_path.name,
            qa_path.name,
            certificate_path.name,
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.sampling_policy in (GUIDE_EXACT_POLICY, WHOLE_MP4_POLICY) and errors:
        raise SystemExit(
            f"{len(errors)} scenes failed exact raw-frame alignment; "
            f"no archive was created; see {report_path}"
        )
    write_archive_atomic(
        args.archive_output,
        (scene_path, frame_path, qa_path, certificate_path, report_path),
    )
    print(f"Archive: {args.archive_output}")


if __name__ == "__main__":
    main()
