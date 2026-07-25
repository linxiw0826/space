#!/usr/bin/env python3
"""Compare candidate ScanNet bbox-to-camera coordinate transforms.

The audit uses the manifest's published visibility labels as an independent
signal: a correct transform should place visible object centroids in front of
the camera, usually inside the image, and at plausible camera-object distance.
It does not modify the manifest.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pc-means", type=Path)
    parser.add_argument(
        "--axis-alignment-root",
        type=Path,
        help="Optional root containing ScanNet scene metadata .txt files.",
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path, max_scenes):
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
                if max_scenes > 0 and len(items) >= max_scenes:
                    break
    return items


def parse_axis_alignment(path):
    if path is None or not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "axisAlignment" not in line or "=" not in line:
            continue
        values = [float(value) for value in line.split("=", 1)[1].split()]
        if len(values) == 16:
            matrix = np.asarray(values, dtype=float).reshape(4, 4)
            return matrix if np.isfinite(matrix).all() else None
    return None


def find_axis_alignment(root, scene):
    if root is None:
        return None, None
    candidates = [
        root / scene / f"{scene}.txt",
        root / f"{scene}.txt",
        root / scene / f"{scene}.txt.txt",
    ]
    for path in candidates:
        matrix = parse_axis_alignment(path)
        if matrix is not None:
            return matrix, path
    return None, None


def as_mean(value):
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.isfinite(array).all():
        return None
    return array


def transform_point(point, mode, mean, axis):
    point_h = np.r_[np.asarray(point, dtype=float), 1.0]
    if mode == "identity":
        return point_h[:3]
    if mode == "add_pc_mean":
        return point_h[:3] + mean
    if mode == "axis":
        return (axis @ point_h)[:3]
    if mode == "inverse_axis":
        return (np.linalg.inv(axis) @ point_h)[:3]
    if mode == "axis_after_mean":
        return (axis @ np.r_[point_h[:3] + mean, 1.0])[:3]
    if mode == "mean_after_axis":
        return (axis @ point_h)[:3] + mean
    raise ValueError(mode)


def safe_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "median": None, "p90": None}
    return {
        "count": int(values.size),
        "median": safe_number(np.median(values)),
        "p90": safe_number(np.percentile(values, 90)),
    }


def evaluate_scene(item, mode, mean, axis):
    nodes = {
        node["node_id"]: np.asarray(
            node["bbox_3d"]["centroid"], dtype=float
        )
        for node in item["nodes"]
    }
    intrinsic = np.asarray(item["intrinsic_color"], dtype=float)
    width = 1296.0
    height = 968.0
    if intrinsic[0, 2] > 0 and intrinsic[1, 2] > 0:
        width = 2.0 * intrinsic[0, 2]
        height = 2.0 * intrinsic[1, 2]

    camera_centers = []
    object_centers = []
    visible_total = 0
    in_front = 0
    in_image = 0
    distances = []

    transformed = {
        node_id: transform_point(center, mode, mean, axis)
        for node_id, center in nodes.items()
    }
    object_centers.extend(transformed.values())

    for view in item["candidate_views"]:
        camera_to_world = np.asarray(view["camera_to_world"], dtype=float)
        world_to_camera = np.linalg.inv(camera_to_world)
        camera_centers.append(camera_to_world[:3, 3])
        for visible in view["visible_nodes"]:
            center = transformed.get(visible["node_id"])
            if center is None:
                continue
            visible_total += 1
            distances.append(np.linalg.norm(
                center - camera_to_world[:3, 3]
            ))
            camera_point = world_to_camera @ np.r_[center, 1.0]
            if camera_point[2] <= 1e-6:
                continue
            in_front += 1
            pixel = intrinsic @ camera_point
            u = pixel[0] / pixel[2]
            v = pixel[1] / pixel[2]
            if 0 <= u < width and 0 <= v < height:
                in_image += 1

    camera_centers = np.asarray(camera_centers, dtype=float)
    object_centers = np.asarray(object_centers, dtype=float)
    spatial_gap = None
    if camera_centers.size and object_centers.size:
        spatial_gap = np.linalg.norm(
            np.median(camera_centers, axis=0)
            - np.median(object_centers, axis=0)
        )

    return {
        "visible_observations": visible_total,
        "front_rate": in_front / max(visible_total, 1),
        "image_rate": in_image / max(visible_total, 1),
        "visible_distance": summarize(distances),
        "camera_object_median_gap": safe_number(spatial_gap),
    }


def main():
    args = parse_args()
    items = load_manifest(args.manifest, args.max_scenes)
    pc_means = load_json(args.pc_means) if args.pc_means else {}

    modes = ["identity"]
    if args.pc_means:
        modes.append("add_pc_mean")
    if args.axis_alignment_root:
        modes.extend(["axis", "inverse_axis"])
    if args.pc_means and args.axis_alignment_root:
        modes.extend(["axis_after_mean", "mean_after_axis"])

    aggregate = {
        mode: defaultdict(list)
        for mode in modes
    }
    scene_results = {}
    missing_means = []
    missing_axis = []

    for item in items:
        scene = item["scene_id"]
        mean = as_mean(pc_means.get(scene)) if scene in pc_means else None
        axis, axis_path = find_axis_alignment(
            args.axis_alignment_root, scene
        )
        if args.pc_means and mean is None:
            missing_means.append(scene)
        if args.axis_alignment_root and axis is None:
            missing_axis.append(scene)

        scene_results[scene] = {
            "pc_mean": mean.tolist() if mean is not None else None,
            "axis_alignment_path": str(axis_path) if axis_path else None,
            "candidates": {},
        }
        for mode in modes:
            if "mean" in mode and mean is None:
                continue
            if "axis" in mode and axis is None:
                continue
            result = evaluate_scene(item, mode, mean, axis)
            scene_results[scene]["candidates"][mode] = result
            for key in [
                "front_rate",
                "image_rate",
                "camera_object_median_gap",
            ]:
                aggregate[mode][key].append(result[key])
            median_distance = result["visible_distance"]["median"]
            if median_distance is not None:
                aggregate[mode]["visible_distance_median"].append(
                    median_distance
                )

    summary = {}
    for mode, metrics in aggregate.items():
        summary[mode] = {
            key: summarize(values)
            for key, values in metrics.items()
        }

    report = {
        "schema_version": "scannet_coordinate_frame_audit_v1",
        "manifest": str(args.manifest),
        "pc_means": str(args.pc_means) if args.pc_means else None,
        "axis_alignment_root": (
            str(args.axis_alignment_root)
            if args.axis_alignment_root else None
        ),
        "scenes": len(items),
        "missing_pc_means": missing_means,
        "missing_axis_alignments": missing_axis,
        "summary": summary,
        "scene_results": scene_results,
        "selection_rule": (
            "Prefer the transform with higher visible-object image/front "
            "rates and lower plausible camera-object median distance/gap. "
            "Confirm the winner with bbox projection visual QC."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "scenes": len(items),
        "missing_pc_means": len(missing_means),
        "missing_axis_alignments": len(missing_axis),
        "summary": summary,
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
