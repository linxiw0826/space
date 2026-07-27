#!/usr/bin/env python3
"""QC and package directly aligned Hypersim Part-A training data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--hypersim-root", required=True, type=Path)
    p.add_argument("--camera-parameters", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--archive-output", required=True, type=Path)
    p.add_argument("--knn", type=int, default=8)
    p.add_argument("--item-limit", type=int)
    return p.parse_args()


def matrix(row, prefix, n):
    return np.asarray(
        [[float(row[f"{prefix}_{i}{j}"]) for j in range(n)] for i in range(n)]
    )


def load_camera_parameters(path):
    result = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["scene_name"]] = {
                "height": int(float(row["settings_output_img_height"])),
                "width": int(float(row["settings_output_img_width"])),
                "projection": matrix(row, "M_proj", 4),
            }
    return result


def read_hdf5(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["dataset"])


def corners(node):
    extent = np.asarray(node["bbox_extent_m"], dtype=np.float64)
    center = np.asarray(node["bbox_center_m"], dtype=np.float64)
    rotation = np.asarray(node["bbox_orientation_raw"], dtype=np.float64)
    signs = np.asarray(
        [
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ]
    )
    return center + (rotation @ (signs * extent).T).T


def project(points_world, camera_position, r_world_from_camera, projection, h, w):
    r_camera_from_world = r_world_from_camera.T
    points_camera = (r_camera_from_world @ (points_world - camera_position).T).T
    homogeneous = np.concatenate(
        [points_camera, np.ones((len(points_camera), 1))], axis=1
    )
    clip = (projection @ homogeneous.T).T
    valid_w = np.abs(clip[:, 3]) > 1e-9
    ndc = np.full((len(points_world), 3), np.nan)
    ndc[valid_w] = clip[valid_w, :3] / clip[valid_w, 3:4]
    uv = np.stack(
        [
            0.5 * (ndc[:, 0] + 1.0) * (w - 1),
            (1.0 - 0.5 * (ndc[:, 1] + 1.0)) * (h - 1),
        ],
        axis=1,
    )
    return points_camera, uv


def bbox_from_pixels(ys, xs):
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def intersection_over_target(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0 + 1) * max(0.0, y1 - y0 + 1)
    target = max(1.0, (b[2] - b[0] + 1) * (b[3] - b[1] + 1))
    return float(intersection / target)


def compact_node(node):
    return {
        "object_id": node["object_id"],
        "node_id": node.get("node_id"),
        "node_name": node.get("node_name"),
        "object_name": node.get("object_name"),
        "center_world_m": node["bbox_center_m"],
        "extent_m": node["bbox_extent_m"],
        "rotation_world_from_object": node["bbox_orientation_raw"],
        "motion_type": "static",
        "identity_source": "hypersim_semantic_instance_id",
    }


def build_edges(nodes, knn):
    if len(nodes) < 2:
        return []
    ids = np.asarray([node["object_id"] for node in nodes])
    centers = np.asarray([node["center_world_m"] for node in nodes])
    extents = np.asarray([node["extent_m"] for node in nodes])
    edges = []
    for index, source in enumerate(nodes):
        delta = centers - centers[index]
        distance = np.linalg.norm(delta, axis=1)
        neighbors = np.argsort(distance)[1 : 1 + min(knn, len(nodes) - 1)]
        for target_index in neighbors:
            edges.append(
                {
                    "source_id": source["object_id"],
                    "target_id": int(ids[target_index]),
                    "delta_world_m": delta[target_index].tolist(),
                    "distance_m": float(distance[target_index]),
                    "source_extent_m": extents[index].tolist(),
                    "target_extent_m": extents[target_index].tolist(),
                }
            )
    return edges


def qc_visible(node, visible, instance_image, depth, camera, params):
    object_id = node["object_id"]
    mask = instance_image == object_id
    ys, xs = np.nonzero(mask)
    camera_position = np.asarray(camera["position_m"], dtype=np.float64)
    r_world_from_camera = np.asarray(camera["orientation_raw"], dtype=np.float64)
    center_world = np.asarray(node["bbox_center_m"], dtype=np.float64)
    center_camera = r_world_from_camera.T @ (center_world - camera_position)
    box_world = corners(node)
    box_camera, uv = project(
        box_world,
        camera_position,
        r_world_from_camera,
        params["projection"],
        params["height"],
        params["width"],
    )
    finite_projection = bool(np.isfinite(uv).all())
    in_front = bool(np.any(box_camera[:, 2] < 0))
    projected_bbox = None
    mask_bbox = None
    mask_coverage = 0.0
    if finite_projection and len(xs):
        projected_bbox = np.asarray(
            [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
        )
        mask_bbox = bbox_from_pixels(ys, xs)
        mask_coverage = intersection_over_target(projected_bbox, mask_bbox)
    depth_values = depth[mask]
    depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
    depth_median = float(np.median(depth_values)) if len(depth_values) else None
    corner_distances = np.linalg.norm(box_world - camera_position, axis=1)
    depth_low = float(corner_distances.min())
    depth_high = float(corner_distances.max())
    tolerance = max(0.25, 0.2 * max(depth_high - depth_low, 0.1))
    depth_consistent = (
        depth_median is not None
        and depth_low - tolerance <= depth_median <= depth_high + tolerance
    )
    rotation_valid = bool(
        np.isfinite(r_world_from_camera).all()
        and abs(np.linalg.det(r_world_from_camera) - 1.0) < 1e-3
    )
    geometry_valid = bool(
        visible["pixel_count"] >= 16
        and finite_projection
        and in_front
        and mask_coverage >= 0.25
        and depth_consistent
        and rotation_valid
    )
    return {
        **visible,
        "center_camera_m": center_camera.tolist(),
        "ego_right_m": float(center_camera[0]),
        "ego_up_m": float(center_camera[1]),
        "ego_forward_m": float(-center_camera[2]),
        "camera_distance_m": float(np.linalg.norm(center_camera)),
        "mask_bbox_xyxy": mask_bbox.tolist() if mask_bbox is not None else None,
        "projected_bbox_xyxy": (
            projected_bbox.tolist() if projected_bbox is not None else None
        ),
        "projected_mask_bbox_coverage": mask_coverage,
        "depth_median_m": depth_median,
        "bbox_corner_distance_range_m": [depth_low, depth_high],
        "projection_valid": finite_projection and in_front,
        "depth_consistent": bool(depth_consistent),
        "rotation_valid": rotation_valid,
        "geometry_valid": geometry_valid,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_parameters = load_camera_parameters(args.camera_parameters)
    scene_path = args.output_dir / "hypersim_scene_states.jsonl"
    frame_path = args.output_dir / "hypersim_frame_states.jsonl"
    qa_path = args.output_dir / "hypersim_qa_train.jsonl"
    report_path = args.output_dir / "hypersim_alignment_report.json"
    seen_scenes = set()
    stats = Counter()
    errors = []

    with (
        args.manifest.open(encoding="utf-8") as source,
        scene_path.open("w", encoding="utf-8") as scene_out,
        frame_path.open("w", encoding="utf-8") as frame_out,
        qa_path.open("w", encoding="utf-8") as qa_out,
    ):
        for frame_index, line in enumerate(source):
            if args.item_limit is not None and frame_index >= args.item_limit:
                break
            record = json.loads(line)
            scene_id = record["scene_id"]
            nodes = [compact_node(node) for node in record["nodes"]]
            nodes_by_id = {node["object_id"]: raw for node, raw in zip(nodes, record["nodes"])}
            if scene_id not in seen_scenes:
                scene_out.write(
                    json.dumps(
                        {
                            "schema_version": "hypersim_scene_state_v1",
                            "scene_id": scene_id,
                            "nodes": nodes,
                            "edges_knn": build_edges(nodes, args.knn),
                            "coordinate_frame": "hypersim_world_meters",
                            "supervision_tier": "gold",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                seen_scenes.add(scene_id)
                stats["scenes"] += 1
                stats["nodes"] += len(nodes)
            try:
                paths = {
                    name: args.hypersim_root / relative
                    for name, relative in record["source_files"].items()
                }
                instance_image = read_hdf5(paths["semantic_instance"])
                depth = read_hdf5(paths["depth"])
                params = camera_parameters[scene_id]
                visible_states = []
                for visible in record["visible_instances"]:
                    node = nodes_by_id[visible["object_id"]]
                    state = qc_visible(
                        node,
                        visible,
                        instance_image,
                        depth,
                        record["camera"],
                        params,
                    )
                    visible_states.append(state)
                    stats["visible_instances"] += 1
                    stats["geometry_valid"] += int(state["geometry_valid"])
                    stats["projection_valid"] += int(state["projection_valid"])
                    stats["depth_consistent"] += int(state["depth_consistent"])
                frame_key = (
                    f"{scene_id}/{record['camera_id']}/{record['frame_id']:04d}"
                )
                r_world_from_camera = np.asarray(
                    record["camera"]["orientation_raw"], dtype=np.float64
                )
                t_world_from_camera = np.asarray(
                    record["camera"]["position_m"], dtype=np.float64
                )
                frame_out.write(
                    json.dumps(
                        {
                            "schema_version": "hypersim_frame_state_v1",
                            "frame_key": frame_key,
                            "scene_id": scene_id,
                            "camera_id": record["camera_id"],
                            "frame_id": record["frame_id"],
                            "vsi_media": record["vsi_media"],
                            "rotation_world_from_camera": r_world_from_camera.tolist(),
                            "translation_world_from_camera_m": t_world_from_camera.tolist(),
                            "rotation_camera_from_world": r_world_from_camera.T.tolist(),
                            "translation_camera_from_world_m": (
                                -r_world_from_camera.T @ t_world_from_camera
                            ).tolist(),
                            "visible_nodes": visible_states,
                            "supervision_tier": "gold",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["frames"] += 1
                for qa in record["qa"]:
                    qa_out.write(
                        json.dumps(
                            {
                                "schema_version": "hypersim_qa_train_v1",
                                "vsi_row_index": qa["vsi_row_index"],
                                "scene_id": scene_id,
                                "frame_key": frame_key,
                                "vsi_media": record["vsi_media"],
                                "question_type": qa["question_type"],
                                "conversations": qa["conversations"],
                                "loss_masks": {
                                    "qa": True,
                                    "node_identity": True,
                                    "scene_geometry": True,
                                    "camera_geometry": any(
                                        state["geometry_valid"]
                                        for state in visible_states
                                    ),
                                    "node_dynamics": False,
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats["qa_rows"] += 1
            except Exception as error:
                stats["frame_errors"] += 1
                if len(errors) < 200:
                    errors.append(
                        {
                            "media": record["vsi_media"],
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    visible = max(stats["visible_instances"], 1)
    report = {
        "schema_version": "hypersim_alignment_report_v1",
        **dict(stats),
        "geometry_valid_rate": stats["geometry_valid"] / visible,
        "projection_valid_rate": stats["projection_valid"] / visible,
        "depth_consistent_rate": stats["depth_consistent"] / visible,
        "errors": errors,
        "coordinate_conventions": {
            "camera_orientation": "camera_to_world",
            "bbox_orientation": "object_to_world",
            "camera_forward_axis": "-z",
            "units": "meters",
        },
        "files": [scene_path.name, frame_path.name, qa_path.name],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.archive_output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive_output, "w:gz") as archive:
        for path in (scene_path, frame_path, qa_path, report_path):
            archive.add(path, arcname=path.name)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Archive: {args.archive_output}")


if __name__ == "__main__":
    main()
