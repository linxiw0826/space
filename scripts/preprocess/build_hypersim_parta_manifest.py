#!/usr/bin/env python3
"""Build a directly joined Part-A manifest for the VSI Hypersim subset.

The semantic-instance pixel value is the source-native object_id. It directly
indexes the released object-aligned bounding-box arrays and metadata_nodes.csv;
no learned or geometric matching is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


MEDIA_PATTERN = re.compile(
    r"^hypersim/(?P<scene>ai_\d{3}_\d{3})/images/"
    r"scene_(?P<camera>cam_\d{2})_final_preview/"
    r"frame\.(?P<frame>\d{4})\.tonemap\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--hypersim-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--item-limit", type=int)
    return parser.parse_args()


def read_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["dataset"])


def load_vsi_rows(
    path: Path,
) -> tuple[dict[str, list[dict]], int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            media = row.get("image") or row.get("video") or ""
            if not media.startswith("hypersim/"):
                continue
            total += 1
            grouped[media].append(
                {
                    "vsi_row_index": row_index,
                    "question_type": row.get("question_type"),
                    "conversations": row.get("conversations"),
                }
            )
    return grouped, total


def read_scale(scene_root: Path) -> float:
    path = scene_root / "_detail/metadata_scene.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        values = {
            row["parameter_name"]: row["parameter_value"]
            for row in csv.DictReader(handle)
        }
    return float(values["meters_per_asset_unit"])


def read_object_names(scene_root: Path) -> dict[int, dict]:
    path = scene_root / "_detail/metadata_nodes.csv"
    objects = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_id = int(row["object_id"])
            objects[object_id] = {
                "node_id": int(row["node_id"]),
                "node_name": row["node_name"],
                "object_name": row["object_name"],
            }
    return objects


def finite_vector(value: np.ndarray) -> bool:
    return bool(np.isfinite(value).all())


def load_scene_nodes(scene_root: Path) -> tuple[dict[int, dict], float]:
    scale = read_scale(scene_root)
    mesh_root = scene_root / "_detail/mesh"
    positions = read_hdf5(
        mesh_root
        / (
            "metadata_semantic_instance_bounding_box_"
            "object_aligned_2d_positions.hdf5"
        )
    )
    extents = read_hdf5(
        mesh_root
        / (
            "metadata_semantic_instance_bounding_box_"
            "object_aligned_2d_extents.hdf5"
        )
    )
    orientations = read_hdf5(
        mesh_root
        / (
            "metadata_semantic_instance_bounding_box_"
            "object_aligned_2d_orientations.hdf5"
        )
    )
    names = read_object_names(scene_root)
    count = min(len(positions), len(extents), len(orientations))
    nodes = {}
    for object_id in range(count):
        if not (
            finite_vector(positions[object_id])
            and finite_vector(extents[object_id])
            and finite_vector(orientations[object_id])
        ):
            continue
        metadata = names.get(object_id, {})
        nodes[object_id] = {
            "object_id": object_id,
            **metadata,
            "bbox_center_m": (positions[object_id] * scale).tolist(),
            "bbox_extent_m": (extents[object_id] * scale).tolist(),
            "bbox_orientation_raw": orientations[object_id].tolist(),
            "motion_type": "static",
            "source_identity": "direct_semantic_instance_id",
        }
    return nodes, scale


def load_camera_frame(
    scene_root: Path, camera: str, frame_id: int, scale: float
) -> dict:
    camera_root = scene_root / "_detail" / camera
    frame_ids = read_hdf5(
        camera_root / "camera_keyframe_frame_indices.hdf5"
    ).astype(np.int64)
    matches = np.flatnonzero(frame_ids == frame_id)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one camera pose for frame {frame_id}, got {len(matches)}"
        )
    index = int(matches[0])
    positions = read_hdf5(
        camera_root / "camera_keyframe_positions.hdf5"
    )
    orientations = read_hdf5(
        camera_root / "camera_keyframe_orientations.hdf5"
    )
    position = positions[index] * scale
    orientation = orientations[index]
    if not finite_vector(position) or not finite_vector(orientation):
        raise ValueError(f"Non-finite camera pose at frame {frame_id}")
    return {
        "frame_id": frame_id,
        "keyframe_index": index,
        "position_m": position.tolist(),
        "orientation_raw": orientation.tolist(),
        "orientation_convention": "hypersim_camera_orientation_raw",
    }


def frame_paths(
    scene_root: Path, camera: str, frame_text: str
) -> dict[str, Path]:
    geometry = (
        scene_root / "images" / f"scene_{camera}_geometry_hdf5"
    )
    prefix = geometry / f"frame.{frame_text}"
    return {
        "depth": Path(f"{prefix}.depth_meters.hdf5"),
        "semantic": Path(f"{prefix}.semantic.hdf5"),
        "semantic_instance": Path(
            f"{prefix}.semantic_instance.hdf5"
        ),
    }


def visible_instances(
    path: Path, scene_nodes: dict[int, dict]
) -> tuple[list[dict], list[int]]:
    instance_image = read_hdf5(path)
    ids, counts = np.unique(instance_image, return_counts=True)
    visible = []
    unknown = []
    total_pixels = int(instance_image.size)
    for object_id, pixels in zip(ids.tolist(), counts.tolist()):
        object_id = int(object_id)
        if object_id < 0:
            continue
        node = scene_nodes.get(object_id)
        if node is None:
            unknown.append(object_id)
            continue
        visible.append(
            {
                "object_id": object_id,
                "pixel_count": int(pixels),
                "image_fraction": float(pixels / total_pixels),
            }
        )
    return visible, unknown


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    grouped, qa_rows = load_vsi_rows(args.jsonl)
    media_items = sorted(grouped)
    if args.item_limit is not None:
        media_items = media_items[: args.item_limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene_cache: dict[str, tuple[dict[int, dict], float]] = {}
    status = Counter()
    question_types = Counter()
    errors = []
    written = 0
    visible_total = 0

    with args.output.open("w", encoding="utf-8") as output:
        for media in media_items:
            match = MEDIA_PATTERN.match(media)
            if not match:
                status["invalid_media_path"] += 1
                errors.append({"media": media, "error": "invalid_media_path"})
                continue
            scene = match["scene"]
            camera = match["camera"]
            frame_text = match["frame"]
            frame_id = int(frame_text)
            scene_root = args.hypersim_root / scene
            try:
                if scene not in scene_cache:
                    scene_cache[scene] = load_scene_nodes(scene_root)
                scene_nodes, scale = scene_cache[scene]
                paths = frame_paths(scene_root, camera, frame_text)
                missing = [
                    name for name, path in paths.items() if not path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(f"Missing modalities: {missing}")
                camera_state = load_camera_frame(
                    scene_root, camera, frame_id, scale
                )
                visible, unknown = visible_instances(
                    paths["semantic_instance"], scene_nodes
                )
                if unknown:
                    raise ValueError(
                        f"Visible IDs without finite 3D bbox: {unknown[:20]}"
                    )
                qas = grouped[media]
                for qa in qas:
                    question_types[qa["question_type"]] += 1
                record = {
                    "schema_version": "hypersim_parta_gold_v1",
                    "source": "hypersim",
                    "scene_id": scene,
                    "camera_id": camera,
                    "frame_id": frame_id,
                    "vsi_media": media,
                    "meters_per_asset_unit": scale,
                    "camera": camera_state,
                    "nodes": list(scene_nodes.values()),
                    "visible_instances": visible,
                    "qa": qas,
                    "source_files": {
                        name: str(path) for name, path in paths.items()
                    },
                    "supervision_tier": "gold",
                    "identity_join": "direct",
                }
                output.write(
                    json.dumps(
                        json_safe(record),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                status["written"] += 1
                written += 1
                visible_total += len(visible)
            except Exception as error:
                status["error"] += 1
                if len(errors) < 200:
                    errors.append(
                        {
                            "media": media,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    summary = {
        "schema_version": "hypersim_parta_gold_summary_v1",
        "requested_media": len(media_items),
        "all_hypersim_media": len(grouped),
        "all_hypersim_qa_rows": qa_rows,
        "written_media": written,
        "written_qa_rows": sum(question_types.values()),
        "scenes_loaded": len(scene_cache),
        "visible_instance_observations": visible_total,
        "status": dict(status),
        "question_types": dict(question_types),
        "errors": errors,
        "output": str(args.output),
        "identity_statement": (
            "semantic_instance pixel IDs directly index source-native "
            "object metadata and 3D bbox arrays; no matching is used"
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
