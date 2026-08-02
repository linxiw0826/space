#!/usr/bin/env python3
"""Fail-closed full-corpus contract audit for the VSI ScanNet++ V2 subset."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import multiprocessing
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import src.scannetppv2_labels as label_contract
from src.scannetppv2_labels import LABEL_ALIASES, normalize_scannetppv2_label


SCHEMA_VERSION = "scannetppv2_full_contract_audit_v2"
FRAME_RE = re.compile(r"^frame_(\d{6})$")
REQUIRED_RELATIVE_PATHS = {
    "mesh": "scans/mesh_aligned_0.05.ply",
    "segments": "scans/segments.json",
    "annotation": "scans/segments_anno.json",
    "pose": "iphone/pose_intrinsic_imu.json",
    "exif": "iphone/exif.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_label(value: Any) -> str:
    return normalize_scannetppv2_label(value)


def finite(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite shape {shape}, got {array.shape}")
    return array


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for _ in range(256):
            line = handle.readline()
            if not line:
                break
            text = line.decode("ascii").strip()
            if text.startswith("element vertex "):
                return int(text.split()[2])
            if text == "end_header":
                break
    raise ValueError("PLY header lacks vertex count")


def video_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in report.get("videos", []):
        media = row.get("media")
        if isinstance(media, str) and media.startswith("scannetppv2/"):
            scene = Path(media).stem
            if scene in rows:
                raise ValueError(f"duplicate video metadata scene: {scene}")
            rows[scene] = row
    return rows


def summarize_vsi_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    observed: dict[int, set[str]] = defaultdict(set)
    observations = 0
    for frame in frames:
        for category, record in frame.items():
            for instance_id in record["inst_ids"]:
                observed[int(instance_id)].add(str(category))
                observations += 1
    return {
        "observed_labels": {
            str(instance_id): sorted(labels)
            for instance_id, labels in observed.items()
        },
        "instance_observations": observations,
        "frame_count": len(frames),
    }


def audit_scene(
    scene: str,
    scene_root: Path,
    frame_summary: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    paths = {name: scene_root / relative for name, relative in REQUIRED_RELATIVE_PATHS.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required assets: {missing}")

    annotation = load_json(paths["annotation"])
    groups = annotation.get("segGroups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("annotation lacks nonempty segGroups")
    indices = [int(group["index"]) for group in groups]
    if indices != list(range(len(groups))):
        raise ValueError("segGroups.index is not zero-based list position")
    object_ids = [int(group["objectId"]) for group in groups]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("annotation objectId is not unique")
    if any(int(group["id"]) != int(group["objectId"]) for group in groups):
        raise ValueError("annotation id/objectId mismatch")
    for group in groups:
        obb = group["obb"]
        finite(obb["centroid"], (3,))
        lengths = finite(obb["axesLengths"], (3,))
        axes = finite(obb["normalizedAxes"], (9,)).reshape(3, 3)
        if (lengths <= 0).any() or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
            raise ValueError("invalid annotation OBB")

    segments = load_json(paths["segments"]).get("segIndices")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments.json lacks segIndices")
    if ply_vertex_count(paths["mesh"]) != len(segments):
        raise ValueError("mesh vertex count differs from segIndices")
    available_segments = set(map(int, segments))
    segment_owners: dict[int, list[int]] = defaultdict(list)
    for group in groups:
        owner = int(group["index"])
        for segment in map(int, group["segments"]):
            if segment not in available_segments:
                raise ValueError("annotation references missing segment")
            if owner in segment_owners[segment]:
                raise ValueError("annotation repeats a segment within one group")
            segment_owners[segment].append(owner)
    owner_counts = Counter(len(owners) for owners in segment_owners.values())
    multilabel_segments = sum(
        count for owner_count, count in owner_counts.items() if owner_count > 1
    )
    # ScanNet++'s official MeshToLabel transform retains at most the first
    # three labels for a vertex, then uses the smallest instance for its
    # single-label representation.  Overlap is therefore native annotation
    # semantics, not corruption.
    multilabel_overflow_segments = sum(
        count for owner_count, count in owner_counts.items() if owner_count > 3
    )

    observed = {
        int(instance_id): set(labels)
        for instance_id, labels in frame_summary["observed_labels"].items()
    }
    observations = int(frame_summary["instance_observations"])
    metainfo_frame_count = int(frame_summary["frame_count"])
    by_index = {int(group["index"]): group for group in groups}
    for instance_id, labels in observed.items():
        if instance_id not in by_index:
            raise ValueError(f"VSI inst_id absent from segGroups.index: {instance_id}")
        if len(labels) != 1:
            raise ValueError(f"VSI inst_id has multiple labels: {instance_id}")
        if normalized_label(next(iter(labels))) != normalized_label(by_index[instance_id]["label"]):
            raise ValueError(f"VSI/official label mismatch: {instance_id}")

    poses = load_json(paths["pose"])
    if not isinstance(poses, dict) or not poses:
        raise ValueError("pose metadata is empty")
    for expected, (key, pose) in enumerate(poses.items()):
        match = FRAME_RE.fullmatch(key)
        if match is None or int(match.group(1)) != expected:
            raise ValueError("pose frame keys are not contiguous")
        finite(pose["intrinsic"], (3, 3))
        finite(pose["aligned_pose"], (4, 4))
    exif = load_json(paths["exif"])
    if not isinstance(exif, dict) or len(exif) != len(poses):
        raise ValueError("EXIF count differs from pose count")
    if video.get("status") != "ok" or int(video["frame_count"]) != len(poses):
        raise ValueError("VSI MP4 frame count differs from official pose count")
    if not math.isclose(float(video["avg_fps"]), 60.0, abs_tol=1e-6):
        raise ValueError("VSI MP4 is not 60 FPS")
    if metainfo_frame_count != (len(poses) + 1) // 2:
        raise ValueError("VSI metainfo count is not ceil(MP4 frame count / 2)")
    widths = {int(record["PixelXDimension"]) for record in exif.values()}
    heights = {int(record["PixelYDimension"]) for record in exif.values()}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("EXIF image dimensions are not constant")
    width, height = next(iter(widths)), next(iter(heights))
    if width * int(video["height"]) != height * int(video["width"]):
        raise ValueError("official/VSI image aspect ratios differ")

    return {
        "scene_id": scene,
        "status": "passed",
        "annotation_groups": len(groups),
        "vsi_observed_instances": len(observed),
        "vsi_instance_observations": observations,
        "identity_join_evidence": (
            "verified_observed_instances"
            if observed
            else "not_applicable_no_vsi_instance_observations"
        ),
        "mesh_vertices": len(segments),
        "annotated_segments": len(segment_owners),
        "segment_owner_count_distribution": {
            str(key): value for key, value in sorted(owner_counts.items())
        },
        "multilabel_segments": multilabel_segments,
        "multilabel_max_owners": max(owner_counts, default=0),
        "multilabel_overflow_segments": multilabel_overflow_segments,
        "single_label_policy": "official_first3_then_smallest_instance_v1",
        "pose_frames": len(poses),
        "metainfo_frames": metainfo_frame_count,
        "video_frames": int(video["frame_count"]),
        "image_scale_x": int(video["width"]) / width,
        "image_scale_y": int(video["height"]) / height,
        "assets": {
            name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def run_scene_task(
    item: tuple[int, str, Path, dict[str, Any], dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    offset, scene, scene_root, frame_summary, video = item
    try:
        result = audit_scene(scene, scene_root, frame_summary, video)
    except Exception as error:  # retain all scene failures in one report
        result = {
            "scene_id": scene,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return offset, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frame-metainfo", required=True, type=Path)
    parser.add_argument("--video-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-pickled-npy", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--official-mesh-transform", required=True, type=Path)
    args = parser.parse_args()
    if not args.allow_pickled_npy:
        raise ValueError("trusted pickled NPY requires --allow-pickled-npy")

    selection = load_json(args.selection_manifest)
    scenes = selection["selected_scenes"]
    raw = np.load(args.frame_metainfo, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise ValueError("frame metainfo must be scalar object NPY")
    metainfo = raw.item()
    videos = video_index(load_json(args.video_metadata))
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    frame_summaries = {
        scene: summarize_vsi_frames(metainfo[scene]) for scene in scenes
    }
    del metainfo, raw
    gc.collect()

    results = []
    failures = Counter()
    items = [
        (
            offset,
            scene,
            args.data_root / "data" / scene,
            frame_summaries[scene],
            videos[scene],
        )
        for offset, scene in enumerate(scenes, 1)
    ]
    if args.workers == 1:
        iterator = map(run_scene_task, items)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        iterator = executor.map(run_scene_task, items, chunksize=1)
    try:
        for offset, result in iterator:
            scene = result["scene_id"]
            if result["status"] == "passed":
                print(f"[{offset}/{len(scenes)}] {scene}: passed", flush=True)
            else:
                failures[result["error_type"]] += 1
                print(
                    f"[{offset}/{len(scenes)}] {scene}: failed: "
                    f"{result['error_type']}: {result['error']}",
                    flush=True,
                )
            results.append(result)
    finally:
        if args.workers != 1:
            executor.shutdown(wait=True)

    passed = sum(row["status"] == "passed" for row in results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_passed" if passed == len(scenes) else "complete_failed",
        "requested_scenes": len(scenes),
        "passed_scenes": passed,
        "failed_scenes": len(scenes) - passed,
        "failure_types": dict(failures),
        "workers": args.workers,
        "identity_join": "vsi_inst_id_equals_segGroups_index_v1",
        "label_normalization": {
            "policy": "lowercase_whitespace_then_explicit_alias_v1",
            "aliases": dict(sorted(LABEL_ALIASES.items())),
        },
        "multilabel_policy": "official_first3_then_smallest_instance_v1",
        "scenes_without_vsi_instance_observations": [
            row["scene_id"]
            for row in results
            if row.get("status") == "passed"
            and row.get("vsi_instance_observations") == 0
        ],
        "audit_source_sha256": sha256_file(Path(__file__)),
        "label_normalization_source_sha256": sha256_file(
            Path(label_contract.__file__)
        ),
        "official_mesh_transform": {
            "path": str(args.official_mesh_transform.resolve()),
            "sha256": sha256_file(args.official_mesh_transform),
        },
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "frame_metainfo_sha256": sha256_file(args.frame_metainfo),
        "video_metadata_sha256": sha256_file(args.video_metadata),
        "scenes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, indent=2))
    if report["status"] != "complete_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
