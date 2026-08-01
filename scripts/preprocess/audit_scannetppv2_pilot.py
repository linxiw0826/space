#!/usr/bin/env python3
"""Fail-closed ScanNet++ V2 pilot audit for VSI instance/camera contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "scannetppv2_pilot_qc_v1"
FRAME_KEY_RE = re.compile(r"^frame_(\d{6})$")
LABEL_ALIASES = {
    "ceiling lamp": "ceiling light",
    "office chair": "chair",
    "trash bin": "trash can",
    "mouse": "computer mouse",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ply_vertex_count(path: Path) -> tuple[int, str]:
    vertex_count = None
    format_name = None
    with path.open("rb") as handle:
        for _ in range(256):
            line = handle.readline()
            if not line:
                break
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError("PLY header is not ASCII") from error
            if text.startswith("format "):
                format_name = text.split()[1]
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[2])
            if text == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds 256 lines")
    if vertex_count is None or format_name is None:
        raise ValueError("PLY header lacks format or vertex count")
    return vertex_count, format_name


def normalized_label(value: Any) -> str:
    label = " ".join(str(value).strip().lower().split())
    return LABEL_ALIASES.get(label, label)


def finite_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite shape {shape}, got {array.shape}")
    return array


def projection_metrics(
    *,
    frames: list[dict[str, Any]],
    poses: dict[str, dict[str, Any]],
    centers: dict[int, np.ndarray],
    source_index,
    width: int,
    height: int,
    sample_count: int = 101,
) -> dict[str, Any]:
    sample_indices = np.linspace(
        0, len(frames) - 1, min(sample_count, len(frames)), dtype=np.int64
    )
    total = 0
    in_front = 0
    center_in_image = 0
    for vsi_index in sample_indices:
        pose = poses[f"frame_{source_index(int(vsi_index)):06d}"]
        world_from_camera = finite_array(pose["aligned_pose"], (4, 4))
        camera_from_world = np.linalg.inv(world_from_camera)
        intrinsic = finite_array(pose["intrinsic"], (3, 3))
        instance_ids = {
            int(value)
            for record in frames[int(vsi_index)].values()
            for value in record["inst_ids"]
        }
        for instance_id in instance_ids:
            camera = camera_from_world @ np.r_[centers[instance_id], 1.0]
            total += 1
            if camera[2] <= 0:
                continue
            in_front += 1
            pixel = intrinsic @ camera[:3]
            u, v = pixel[:2] / pixel[2]
            center_in_image += int(0 <= u < width and 0 <= v < height)
    if total == 0:
        raise ValueError("projection audit has no visible observations")
    return {
        "sampled_vsi_frames": len(sample_indices),
        "visible_object_centers": total,
        "in_front_count": in_front,
        "in_front_rate": in_front / total,
        "center_in_image_count": center_in_image,
        "center_in_image_rate": center_in_image / total,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scene_root = args.data_root / "data" / args.scene_id
    paths = {
        "mesh": scene_root / "scans/mesh_aligned_0.05.ply",
        "segments": scene_root / "scans/segments.json",
        "annotation": scene_root / "scans/segments_anno.json",
        "iphone_pose": scene_root / "iphone/pose_intrinsic_imu.json",
        "iphone_exif": scene_root / "iphone/exif.json",
        "frame_metainfo": args.frame_metainfo,
        "video_metadata": args.video_metadata,
    }
    artifacts = {name: artifact(path) for name, path in paths.items()}

    annotation = load_json(paths["annotation"])
    groups = annotation.get("segGroups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("segments_anno.json lacks nonempty segGroups")
    required_group_keys = {"index", "id", "objectId", "label", "segments", "obb"}
    if any(not required_group_keys <= set(group) for group in groups):
        raise ValueError("annotation group lacks required fields")
    indices = [int(group["index"]) for group in groups]
    object_ids = [int(group["objectId"]) for group in groups]
    if indices != list(range(len(groups))):
        raise ValueError("segGroups.index must equal zero-based list position")
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("duplicate annotation objectId")
    if any(int(group["id"]) != int(group["objectId"]) for group in groups):
        raise ValueError("annotation id/objectId mismatch")

    obb_determinants = []
    centers = {}
    for group in groups:
        obb = group["obb"]
        centers[int(group["index"])] = finite_array(obb["centroid"], (3,))
        lengths = finite_array(obb["axesLengths"], (3,))
        axes = finite_array(obb["normalizedAxes"], (9,)).reshape(3, 3)
        if (lengths <= 0).any():
            raise ValueError("OBB axesLengths must be positive")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5, rtol=1e-5):
            raise ValueError("OBB normalizedAxes are not orthonormal")
        determinant = float(np.linalg.det(axes))
        if not math.isclose(abs(determinant), 1.0, abs_tol=1e-5):
            raise ValueError("OBB axes determinant is not unit magnitude")
        obb_determinants.append(determinant)

    segments = load_json(paths["segments"]).get("segIndices")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments.json lacks nonempty segIndices")
    vertex_count, ply_format = ply_vertex_count(paths["mesh"])
    if vertex_count != len(segments):
        raise ValueError("mesh vertex count differs from segIndices count")
    available_segments = set(map(int, segments))
    annotated_owner: dict[int, int] = {}
    duplicate_segment_owners = []
    missing_segment_refs = []
    for group in groups:
        for segment in map(int, group["segments"]):
            if segment not in available_segments:
                missing_segment_refs.append(segment)
            previous = annotated_owner.setdefault(segment, int(group["index"]))
            if previous != int(group["index"]):
                duplicate_segment_owners.append(segment)
    if missing_segment_refs or duplicate_segment_owners:
        raise ValueError("annotation segment references are invalid or multiply owned")

    if not args.allow_pickled_npy:
        raise ValueError("trusted pickled NPY requires --allow-pickled-npy")
    raw = np.load(args.frame_metainfo, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise ValueError("frame metainfo must be scalar object NPY")
    payload = raw.item()
    frames = payload.get(args.scene_id)
    if not isinstance(frames, list) or not frames:
        raise ValueError("pilot scene lacks frame metainfo")
    vsi_labels: dict[int, set[str]] = defaultdict(set)
    observation_count = 0
    for frame in frames:
        for category, record in frame.items():
            for instance_id in record["inst_ids"]:
                vsi_labels[int(instance_id)].add(str(category))
                observation_count += 1
    if not vsi_labels:
        raise ValueError("pilot contains no VSI instance observations")
    missing_annotation_indices = sorted(set(vsi_labels) - set(indices))
    multi_label_indices = {
        str(key): sorted(values)
        for key, values in vsi_labels.items()
        if len(values) != 1
    }
    label_mismatches = []
    group_by_index = {int(group["index"]): group for group in groups}
    for index, labels in sorted(vsi_labels.items()):
        if index not in group_by_index or len(labels) != 1:
            continue
        vsi_label = next(iter(labels))
        official_label = str(group_by_index[index]["label"])
        if normalized_label(vsi_label) != normalized_label(official_label):
            label_mismatches.append(
                {
                    "index": index,
                    "vsi_label": vsi_label,
                    "official_label": official_label,
                    "objectId": int(group_by_index[index]["objectId"]),
                }
            )
    if missing_annotation_indices or multi_label_indices or label_mismatches:
        raise ValueError("VSI instance-index/category join failed")

    poses = load_json(paths["iphone_pose"])
    if not isinstance(poses, dict) or not poses:
        raise ValueError("iPhone pose metadata must be a nonempty mapping")
    pose_keys = list(poses)
    pose_indices = []
    timestamps = []
    for key in pose_keys:
        match = FRAME_KEY_RE.fullmatch(key)
        if match is None:
            raise ValueError(f"invalid iPhone frame key: {key}")
        pose_indices.append(int(match.group(1)))
        record = poses[key]
        timestamps.append(float(record["timestamp"]))
        finite_array(record["intrinsic"], (3, 3))
        finite_array(record["pose"], (4, 4))
        finite_array(record["aligned_pose"], (4, 4))
    if pose_indices != list(range(len(poses))):
        raise ValueError("iPhone pose frame keys are not contiguous")
    timestamp_steps = np.diff(np.asarray(timestamps, dtype=np.float64))
    if not (timestamp_steps > 0).all():
        raise ValueError("iPhone timestamps are not strictly increasing")

    exif = load_json(paths["iphone_exif"])
    if not isinstance(exif, dict) or len(exif) != len(poses):
        raise ValueError("EXIF record count differs from pose count")
    exif_timestamps = {float(value) for value in exif}
    if exif_timestamps != set(timestamps):
        raise ValueError("EXIF timestamp keys differ from pose timestamps")

    image_widths = {int(record["PixelXDimension"]) for record in exif.values()}
    image_heights = {int(record["PixelYDimension"]) for record in exif.values()}
    if len(image_widths) != 1 or len(image_heights) != 1:
        raise ValueError("iPhone EXIF dimensions are not constant")
    image_width = next(iter(image_widths))
    image_height = next(iter(image_heights))

    candidate_stride = 2
    expected_metainfo_frames = (len(poses) + candidate_stride - 1) // candidate_stride
    if expected_metainfo_frames != len(frames):
        raise ValueError("VSI metainfo count is not ceil(iPhone pose count / 2)")

    video_report = load_json(args.video_metadata)
    expected_media = f"scannetppv2/{args.scene_id}.mp4"
    video_rows = [
        row
        for row in video_report.get("videos", ())
        if row.get("media") == expected_media
    ]
    if len(video_rows) != 1 or video_rows[0].get("status") != "ok":
        raise ValueError("pilot VSI MP4 metadata row is missing or invalid")
    video = video_rows[0]
    if int(video["frame_count"]) != len(poses):
        raise ValueError("VSI MP4 frame count differs from iPhone pose count")
    if not math.isclose(float(video["avg_fps"]), 60.0, abs_tol=1e-6):
        raise ValueError("pilot VSI MP4 is not 60 FPS")
    if (
        image_width % int(video["width"]) != 0
        or image_height % int(video["height"]) != 0
        or image_width // int(video["width"])
        != image_height // int(video["height"])
    ):
        raise ValueError("VSI MP4 dimensions are not uniform-scale iPhone dimensions")
    image_downsample_factor = image_width // int(video["width"])
    candidate_projection = projection_metrics(
        frames=frames,
        poses=poses,
        centers=centers,
        source_index=lambda index: 2 * index,
        width=image_width,
        height=image_height,
    )
    control_projection = projection_metrics(
        frames=frames,
        poses=poses,
        centers=centers,
        source_index=lambda index: index,
        width=image_width,
        height=image_height,
    )
    if (
        candidate_projection["in_front_rate"] < 0.99
        or candidate_projection["center_in_image_rate"] < 0.6
        or candidate_projection["center_in_image_rate"]
        - control_projection["center_in_image_rate"]
        < 0.2
    ):
        raise ValueError("candidate stride-2 projection evidence is too weak")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_pending_pixel_projection_verification",
        "scene_id": args.scene_id,
        "artifacts": artifacts,
        "identity_contract": {
            "vsi_inst_id_field": "segGroups.index",
            "vsi_inst_id_is_objectId": False,
            "annotation_group_count": len(groups),
            "vsi_observed_instance_count": len(vsi_labels),
            "vsi_instance_observation_count": observation_count,
            "missing_annotation_indices": missing_annotation_indices,
            "multi_label_indices": multi_label_indices,
            "label_mismatches_after_aliases": label_mismatches,
            "passed": True,
        },
        "geometry_contract": {
            "ply_format": ply_format,
            "mesh_vertex_count": vertex_count,
            "segindex_count": len(segments),
            "unique_segment_count": len(available_segments),
            "annotated_segment_count": len(annotated_owner),
            "obb_count": len(groups),
            "obb_axes_determinant_min": min(obb_determinants),
            "obb_axes_determinant_max": max(obb_determinants),
            "passed": True,
        },
        "camera_frame_contract": {
            "official_stream": "iphone_rgb",
            "official_pose_count": len(poses),
            "official_exif_count": len(exif),
            "official_image_width": image_width,
            "official_image_height": image_height,
            "official_timestamp_step_seconds": {
                "min": float(timestamp_steps.min()),
                "median": float(np.median(timestamp_steps)),
                "max": float(timestamp_steps.max()),
            },
            "vsi_metainfo_frame_count": len(frames),
            "vsi_mp4_frame_count": int(video["frame_count"]),
            "vsi_mp4_fps": float(video["avg_fps"]),
            "vsi_mp4_width": int(video["width"]),
            "vsi_mp4_height": int(video["height"]),
            "image_downsample_factor": image_downsample_factor,
            "mp4_to_official_mapping": "vsi_mp4_frame_i_to_iphone_frame_i",
            "metainfo_source_stride": candidate_stride,
            "metainfo_to_mp4_mapping": "vsi_metainfo_frame_j_to_mp4_frame_2j",
            "mp4_pose_count_relation_verified": True,
            "metainfo_count_relation_verified": True,
            "candidate_projection_metrics": candidate_projection,
            "stride_1_control_projection_metrics": control_projection,
            "visibility_projection_hypothesis_verified": True,
            "mp4_frame_count_verified": True,
            "pixel_projection_verified": False,
            "passed": False,
        },
        "unresolved_contracts": [
            "MP4-to-pose identity mapping must pass extracted-image projection QC",
            "odd training frames require mesh-rendered visibility because VSI metainfo is stride-2",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frame-metainfo", required=True, type=Path)
    parser.add_argument("--video-metadata", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-pickled-npy", action="store_true")
    args = parser.parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
