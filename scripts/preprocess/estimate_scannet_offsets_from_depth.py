#!/usr/bin/env python3
"""Estimate ScanNet metadata-to-pose translations from RGB-D geometry.

GUIDE's processed ScanNet directory contains depth, intrinsics, and
camera-to-world poses but not the point-cloud normalization means used by the
VSI metadata generator. This script reconstructs a sparse world point cloud
and aligns the metadata room center to its robust world-space bounds center.
The output JSON is compatible with audit_scannet_coordinate_frames.py
--pc-means and must be validated there before it is used as supervision.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--pixel-stride", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    return parser.parse_args()


def load_manifest(path, max_scenes):
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if max_scenes > 0 and len(items) >= max_scenes:
                break
    return items


def read_matrix(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid 4x4 matrix: {path}")
    return matrix


def numeric_paths(directory, suffix):
    paths = []
    for path in directory.glob(f"*{suffix}"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        paths.append((index, path))
    return sorted(paths)


def reconstruct_scene_points(scene_root, args):
    depth_dir = scene_root / "image_depth"
    pose_dir = scene_root / "pose"
    intrinsic_path = (
        scene_root / "intrinsic" / "intrinsic_depth.txt"
    )
    intrinsic = read_matrix(intrinsic_path)
    pose_paths = dict(numeric_paths(pose_dir, ".txt"))
    depth_paths = numeric_paths(depth_dir, ".png")

    chunks = []
    used_frames = []
    skipped_frames = []
    for position, (frame_id, depth_path) in enumerate(depth_paths):
        if position % args.frame_stride:
            continue
        pose_path = pose_paths.get(frame_id)
        if pose_path is None:
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": "missing_pose",
            })
            continue
        depth_raw = cv2.imread(
            str(depth_path), cv2.IMREAD_UNCHANGED
        )
        if depth_raw is None or depth_raw.ndim != 2:
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": "invalid_depth",
            })
            continue
        try:
            camera_to_world = read_matrix(pose_path)
        except Exception as error:
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": (
                    f"invalid_pose:{type(error).__name__}"
                ),
            })
            continue
        height, width = depth_raw.shape
        rows = np.arange(
            args.pixel_stride // 2, height, args.pixel_stride
        )
        cols = np.arange(
            args.pixel_stride // 2, width, args.pixel_stride
        )
        grid_u, grid_v = np.meshgrid(cols, rows)
        depth = (
            depth_raw[grid_v, grid_u].astype(np.float64)
            / args.depth_scale
        )
        valid = (
            np.isfinite(depth)
            & (depth >= args.min_depth)
            & (depth <= args.max_depth)
        )
        if not valid.any():
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": "no_valid_depth",
            })
            continue
        z = depth[valid]
        u = grid_u[valid].astype(np.float64)
        v = grid_v[valid].astype(np.float64)
        x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
        camera_points = np.stack(
            [x, y, z, np.ones_like(z)], axis=1
        )
        world_points = (
            camera_to_world @ camera_points.T
        ).T[:, :3]
        finite = np.isfinite(world_points).all(axis=1)
        if finite.any():
            chunks.append(world_points[finite])
            used_frames.append(frame_id)

    if not chunks:
        raise ValueError(f"No reconstructed points: {scene_root}")
    return np.concatenate(chunks, axis=0), used_frames, skipped_frames


def main():
    args = parse_args()
    if not 0 <= args.lower_quantile < args.upper_quantile <= 1:
        raise ValueError("Invalid quantile range")
    items = load_manifest(args.manifest, args.max_scenes)
    offsets = {}
    report = {
        "schema_version": "scannet_depth_offset_estimate_v1",
        "manifest": str(args.manifest),
        "geometry_root": str(args.geometry_root),
        "method": (
            "robust reconstructed world AABB center minus published "
            "metadata room_center"
        ),
        "parameters": {
            "pixel_stride": args.pixel_stride,
            "frame_stride": args.frame_stride,
            "depth_scale": args.depth_scale,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "lower_quantile": args.lower_quantile,
            "upper_quantile": args.upper_quantile,
        },
        "scenes": {},
        "errors": {},
    }

    for item in items:
        scene = item["scene_id"]
        try:
            points, used_frames, skipped_frames = (
                reconstruct_scene_points(
                    args.geometry_root / scene, args
                )
            )
            lower = np.quantile(
                points, args.lower_quantile, axis=0
            )
            upper = np.quantile(
                points, args.upper_quantile, axis=0
            )
            reconstructed_center = (lower + upper) / 2.0
            metadata_center = np.asarray(
                item["room_center"], dtype=float
            ).reshape(3)
            offset = reconstructed_center - metadata_center
            if not np.isfinite(offset).all():
                raise ValueError("Non-finite estimated offset")
            offsets[scene] = offset.tolist()
            report["scenes"][scene] = {
                "point_count": int(points.shape[0]),
                "used_frame_count": len(used_frames),
                "used_frames": used_frames,
                "skipped_frames": skipped_frames,
                "robust_world_lower": lower.tolist(),
                "robust_world_upper": upper.tolist(),
                "reconstructed_center": reconstructed_center.tolist(),
                "metadata_room_center": metadata_center.tolist(),
                "estimated_offset": offset.tolist(),
            }
        except Exception as error:
            report["errors"][scene] = (
                f"{type(error).__name__}: {error}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(offsets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.report_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    norms = [
        float(np.linalg.norm(value))
        for value in offsets.values()
    ]
    print(json.dumps({
        "requested_scenes": len(items),
        "estimated_scenes": len(offsets),
        "errors": len(report["errors"]),
        "offset_norm_median": (
            float(np.median(norms)) if norms else None
        ),
        "offsets_output": str(args.output),
        "report_output": str(args.report_output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
