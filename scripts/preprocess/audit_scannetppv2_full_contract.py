#!/usr/bin/env python3
"""Fail-closed full-corpus contract audit for the VSI ScanNet++ V2 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "scannetppv2_full_contract_audit_v1"
FRAME_RE = re.compile(r"^frame_(\d{6})$")
LABEL_ALIASES = {
    "ceiling lamp": "ceiling light",
    "office chair": "chair",
    "trash bin": "trash can",
    "mouse": "computer mouse",
}
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
    label = " ".join(str(value).strip().lower().split())
    return LABEL_ALIASES.get(label, label)


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


def audit_scene(
    scene: str,
    scene_root: Path,
    frames: list[dict[str, Any]],
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
    owned_segments: dict[int, int] = {}
    for group in groups:
        owner = int(group["index"])
        for segment in map(int, group["segments"]):
            if segment not in available_segments:
                raise ValueError("annotation references missing segment")
            previous = owned_segments.setdefault(segment, owner)
            if previous != owner:
                raise ValueError("segment has multiple annotation owners")

    observed: dict[int, set[str]] = defaultdict(set)
    observations = 0
    for frame in frames:
        for category, record in frame.items():
            for instance_id in record["inst_ids"]:
                observed[int(instance_id)].add(str(category))
                observations += 1
    by_index = {int(group["index"]): group for group in groups}
    if not observed:
        raise ValueError("VSI metainfo contains no instance observations")
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
    if len(frames) != (len(poses) + 1) // 2:
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
        "mesh_vertices": len(segments),
        "annotated_segments": len(owned_segments),
        "pose_frames": len(poses),
        "metainfo_frames": len(frames),
        "video_frames": int(video["frame_count"]),
        "image_scale_x": int(video["width"]) / width,
        "image_scale_y": int(video["height"]) / height,
        "assets": {
            name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frame-metainfo", required=True, type=Path)
    parser.add_argument("--video-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-pickled-npy", action="store_true")
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
    results = []
    failures = Counter()
    for offset, scene in enumerate(scenes, 1):
        try:
            result = audit_scene(scene, args.data_root / "data" / scene, metainfo[scene], videos[scene])
            print(f"[{offset}/{len(scenes)}] {scene}: passed", flush=True)
        except Exception as error:  # retain all scene failures in one report
            code = type(error).__name__
            failures[code] += 1
            result = {"scene_id": scene, "status": "failed", "error_type": code, "error": str(error)}
            print(f"[{offset}/{len(scenes)}] {scene}: failed: {code}: {error}", flush=True)
        results.append(result)

    passed = sum(row["status"] == "passed" for row in results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_passed" if passed == len(scenes) else "complete_failed",
        "requested_scenes": len(scenes),
        "passed_scenes": passed,
        "failed_scenes": len(scenes) - passed,
        "failure_types": dict(failures),
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
