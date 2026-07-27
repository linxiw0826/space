#!/usr/bin/env python3
"""Match ScanNet frame instance IDs to published 3D boxes.

The public VSI-590K metadata does not preserve a shared instance identifier
between per-frame ``inst_ids`` and scene-level 3D boxes.  This script builds a
32-view trajectory for each side and solves a category-wise one-to-one
assignment.  It is an auditable geometric estimate, not a ground-truth join.
"""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


DEFAULT_META_MEMBER = (
    "fianlver-vsibench/scannet_train_meta_info-20250130.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--frame-info-npy", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-member", default=DEFAULT_META_MEMBER)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-tolerance", type=float, default=0.75)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--high-score", type=float, default=0.60)
    parser.add_argument("--high-margin", type=float, default=0.15)
    return parser.parse_args()


def load_jsonl(path, max_scenes=0):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_scenes > 0 and len(rows) >= max_scenes:
                    break
    return rows


def load_metadata(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return json.load(handle)


def read_matrix(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid 4x4 matrix: {path}")
    return matrix


def bbox_center_and_corners(box, offset):
    center = np.asarray(box["centroid"], dtype=np.float64).reshape(3) + offset
    lengths = np.asarray(
        box.get("axesLengths") or box.get("axes_lengths"),
        dtype=np.float64,
    ).reshape(3)
    axes = np.asarray(
        box.get("normalizedAxes") or np.eye(3),
        dtype=np.float64,
    ).reshape(3, 3)
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float64)
    corners = center + (signs * (lengths / 2.0)) @ axes
    return center, corners


def depth_files_by_source(scene_root):
    result = {}
    for path in (scene_root / "video_depth").glob("frame*_*.png"):
        try:
            source = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        result[source] = path
    return result


def project_points(points_world, camera_to_world, intrinsic):
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1))], axis=1
    )
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    projected = np.full((len(points_world), 2), np.nan)
    valid = z > 1e-6
    projected[valid, 0] = (
        intrinsic[0, 0] * camera[valid, 0] / z[valid]
        + intrinsic[0, 2]
    )
    projected[valid, 1] = (
        intrinsic[1, 1] * camera[valid, 1] / z[valid]
        + intrinsic[1, 2]
    )
    return projected, z


def bbox_view_feature(
    center,
    corners,
    camera_to_world,
    intrinsic,
    depth_path,
    args,
):
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or depth_raw.ndim != 2:
        return {"valid": False, "visibility": 0.0, "area": 0.0}
    height, width = depth_raw.shape
    points = np.concatenate([center[None, :], corners], axis=0)
    uv, z = project_points(points, camera_to_world, intrinsic)
    center_uv = uv[0]
    center_z = z[0]
    corner_uv = uv[1:]
    corner_z = z[1:]
    front = center_z > 0 and np.count_nonzero(corner_z > 0) >= 4
    if not front or not np.isfinite(center_uv).all():
        return {"valid": True, "visibility": 0.0, "area": 0.0}

    finite_corners = np.isfinite(corner_uv).all(axis=1) & (corner_z > 0)
    if not finite_corners.any():
        return {"valid": True, "visibility": 0.0, "area": 0.0}
    visible_uv = corner_uv[finite_corners]
    x0 = float(np.clip(visible_uv[:, 0].min(), 0, width - 1))
    x1 = float(np.clip(visible_uv[:, 0].max(), 0, width - 1))
    y0 = float(np.clip(visible_uv[:, 1].min(), 0, height - 1))
    y1 = float(np.clip(visible_uv[:, 1].max(), 0, height - 1))
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_fraction = area / max(float(width * height), 1.0)

    u = int(round(center_uv[0]))
    v = int(round(center_uv[1]))
    in_image = 0 <= u < width and 0 <= v < height and area > 1.0
    depth_consistent = False
    observed_depth = None
    if in_image:
        radius = 2
        patch = depth_raw[
            max(0, v - radius):min(height, v + radius + 1),
            max(0, u - radius):min(width, u + radius + 1),
        ].astype(np.float64) / args.depth_scale
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size:
            observed_depth = float(np.median(patch))
            depth_consistent = (
                center_z <= observed_depth + args.depth_tolerance
            )
    visibility = float(in_image) * (1.0 if depth_consistent else 0.35)
    return {
        "valid": True,
        "visibility": visibility,
        "area": area_fraction,
        "center_uv": center_uv.tolist(),
        "center_depth": float(center_z),
        "observed_depth": observed_depth,
        "depth_consistent": depth_consistent,
    }


def instance_trajectories(records, pose_ids):
    by_category = defaultdict(lambda: defaultdict(dict))
    for view_index, frame_id in enumerate(pose_ids):
        record = records[frame_id]
        for category, payload in record.items():
            ids = np.asarray(payload.get("inst_ids", [])).reshape(-1)
            pixels = np.asarray(
                payload.get("inst_num_pixels", [])
            ).reshape(-1)
            for instance_id, pixel_count in zip(ids, pixels):
                by_category[category][int(instance_id)][view_index] = int(
                    pixel_count
                )
    result = {}
    for category, instances in by_category.items():
        result[category] = {}
        for instance_id, sparse in instances.items():
            trajectory = np.zeros(len(pose_ids), dtype=np.float64)
            for view_index, pixels in sparse.items():
                trajectory[view_index] = pixels
            result[category][instance_id] = trajectory
    return result


def normalize_trajectory(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.any(values > 0):
        return np.zeros_like(values)
    scale = np.percentile(values[values > 0], 90)
    return np.clip(values / max(scale, 1e-9), 0.0, 1.0)


def trajectory_score(actual_pixels, predicted_visibility, predicted_area):
    actual = normalize_trajectory(actual_pixels)
    predicted = normalize_trajectory(
        np.asarray(predicted_visibility)
        * np.sqrt(np.maximum(predicted_area, 0.0))
    )
    actual_binary = actual_pixels > 0
    predicted_binary = np.asarray(predicted_visibility) >= 0.5
    intersection = np.count_nonzero(actual_binary & predicted_binary)
    union = np.count_nonzero(actual_binary | predicted_binary)
    visibility_iou = intersection / union if union else 0.0
    denom = np.linalg.norm(actual) * np.linalg.norm(predicted)
    cosine = float(np.dot(actual, predicted) / denom) if denom else 0.0
    presence = 1.0 - abs(
        actual_binary.mean() - predicted_binary.mean()
    )
    total = 0.50 * visibility_iou + 0.35 * cosine + 0.15 * presence
    return float(total), {
        "visibility_iou": float(visibility_iou),
        "trajectory_cosine": cosine,
        "presence_similarity": float(presence),
    }


def classify(score, margin, args):
    if score >= args.high_score and margin >= args.high_margin:
        return "high"
    if score >= args.min_score and margin >= args.min_margin:
        return "medium"
    return "low"


def match_category(
    category,
    instance_map,
    boxes,
    box_features,
    args,
):
    instance_ids = sorted(instance_map)
    score_matrix = np.zeros((len(instance_ids), len(boxes)))
    components = {}
    for row, instance_id in enumerate(instance_ids):
        for col in range(len(boxes)):
            features = box_features[col]
            score, detail = trajectory_score(
                instance_map[instance_id],
                [item["visibility"] for item in features],
                [item["area"] for item in features],
            )
            score_matrix[row, col] = score
            components[(row, col)] = detail

    assignments = []
    if score_matrix.size:
        rows, cols = linear_sum_assignment(-score_matrix)
        for row, col in zip(rows.tolist(), cols.tolist()):
            score = float(score_matrix[row, col])
            alternatives = np.delete(score_matrix[row], col)
            second = float(alternatives.max()) if alternatives.size else 0.0
            margin = score - second
            singleton = len(instance_ids) == 1 and len(boxes) == 1
            if singleton:
                confidence = "high"
                status = "verified_singleton"
                source = "category_count_unique"
            else:
                confidence = classify(score, margin, args)
                status = "matched" if confidence != "low" else "ambiguous"
                source = "temporal_geometry_estimate"
            assignments.append({
                "category": category,
                "instance_id": instance_ids[row],
                "bbox_index": col,
                "assignment_scope": (
                    "singleton" if singleton else "multi_instance"
                ),
                "mapping_source": source,
                "match_score": score,
                "match_margin": margin,
                "second_best_score": second,
                "mapping_status": status,
                "mapping_confidence": confidence,
                "score_components": components[(row, col)],
                "actual_pixel_trajectory": (
                    instance_map[instance_ids[row]].astype(int).tolist()
                ),
                "bbox_visibility_trajectory": [
                    item["visibility"] for item in box_features[col]
                ],
                "bbox_area_trajectory": [
                    item["area"] for item in box_features[col]
                ],
            })
    return assignments, score_matrix.tolist(), instance_ids


def process_scene(item, metadata, frame_info, offsets, args):
    scene = item["scene_id"]
    scene_root = args.geometry_root / scene
    pose_ids = [
        int(view["mp4_frame_index"]) for view in item["candidate_views"]
    ]
    poses = [
        np.asarray(view["camera_to_world"], dtype=np.float64)
        for view in item["candidate_views"]
    ]
    intrinsic = read_matrix(
        scene_root / "intrinsic" / "intrinsic_depth.txt"
    )
    depth_paths = depth_files_by_source(scene_root)
    offset = np.asarray(offsets[scene], dtype=np.float64).reshape(3)
    trajectories = instance_trajectories(frame_info[scene], pose_ids)
    boxes_by_category = (
        metadata[scene].get("object_bbox")
        or metadata[scene].get("object_bboxes")
        or {}
    )

    categories = {}
    assignments = []
    for category in sorted(set(trajectories) & set(boxes_by_category)):
        instances = trajectories[category]
        boxes = boxes_by_category[category]
        if not instances or not boxes:
            continue
        features_by_box = []
        for box in boxes:
            center, corners = bbox_center_and_corners(box, offset)
            features = []
            for frame_id, pose in zip(pose_ids, poses):
                depth_path = depth_paths.get(frame_id)
                if depth_path is None:
                    features.append({
                        "valid": False,
                        "visibility": 0.0,
                        "area": 0.0,
                    })
                else:
                    features.append(bbox_view_feature(
                        center, corners, pose, intrinsic, depth_path, args
                    ))
            features_by_box.append(features)
        matched, matrix, instance_ids = match_category(
            category, instances, boxes, features_by_box, args
        )
        categories[category] = {
            "instance_ids": instance_ids,
            "bbox_count": len(boxes),
            "count_match": len(instance_ids) == len(boxes),
            "score_matrix": matrix,
            "assignments": matched,
        }
        assignments.extend(matched)

    return {
        "scene_id": scene,
        "schema_version": "scannet_instance_bbox_matching_v1",
        "method": "32-view_bbox_projection_depth_and_pixel_trajectory_hungarian",
        "limitations": [
            "No shared source instance ID is available.",
            "No instance mask is used; this is not mask-level 3D IoU.",
            "Low-confidence assignments must not supervise instance geometry.",
        ],
        "categories": categories,
        "assignments": assignments,
    }


def main():
    args = parse_args()
    items = load_jsonl(args.manifest, args.max_scenes)
    metadata = load_metadata(args.metadata_zip, args.metadata_member)
    frame_info = np.load(args.frame_info_npy, allow_pickle=True).item()
    offsets = json.loads(args.offsets.read_text(encoding="utf-8"))

    reports = []
    errors = {}
    confidence_counts = Counter()
    status_counts = Counter()
    scope_counts = defaultdict(Counter)
    category_counts = defaultdict(Counter)
    multi_scores = []
    multi_margins = []
    for item in items:
        scene = item["scene_id"]
        try:
            report = process_scene(
                item, metadata, frame_info, offsets, args
            )
            reports.append(report)
            for assignment in report["assignments"]:
                confidence = assignment["mapping_confidence"]
                status = assignment["mapping_status"]
                confidence_counts[confidence] += 1
                status_counts[status] += 1
                scope = assignment["assignment_scope"]
                scope_counts[scope][confidence] += 1
                category_counts[assignment["category"]][confidence] += 1
                if scope == "multi_instance":
                    multi_scores.append(assignment["match_score"])
                    multi_margins.append(assignment["match_margin"])
        except Exception as error:
            errors[scene] = f"{type(error).__name__}: {error}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "scannet_instance_bbox_matching_summary_v1",
        "method": "32-view_bbox_projection_depth_and_pixel_trajectory_hungarian",
        "requested_scenes": len(items),
        "processed_scenes": len(reports),
        "errors": errors,
        "thresholds": {
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "high_score": args.high_score,
            "high_margin": args.high_margin,
        },
        "assignment_status_counts": dict(status_counts),
        "assignment_confidence_counts": dict(confidence_counts),
        "scope_confidence_counts": {
            scope: dict(counts)
            for scope, counts in sorted(scope_counts.items())
        },
        "multi_instance_score_quantiles": quantiles(multi_scores),
        "multi_instance_margin_quantiles": quantiles(multi_margins),
        "category_confidence_counts": {
            category: dict(counts)
            for category, counts in sorted(category_counts.items())
        },
        "output": str(args.output),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def quantiles(values):
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


if __name__ == "__main__":
    main()
