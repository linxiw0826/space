#!/usr/bin/env python3
"""Build exact-frame ScanNet++ V2 PartA gold source tables and certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (
    GUIDE_WHOLE_MP4_SAMPLING_POLICY,
    content_sha256,
    guide_frame_indices,
    guide_sampling_binding_sha256,
    stable_json,
)
from src.scannetppv2_rasterizer import (
    Rasterizer,
    instance_pixel_counts,
    load_mesh_instances,
    sha256_file,
)
from src.scannetppv2_labels import normalize_scannetppv2_label
from src.scannetppv2_support import build_support_certificate


REQUIRED_ASSETS = {
    "mesh": "scans/mesh_aligned_0.05.ply",
    "segments": "scans/segments.json",
    "annotation": "scans/segments_anno.json",
    "pose": "iphone/pose_intrinsic_imu.json",
    "exif": "iphone/exif.json",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
            count += 1
    return count


def artifact(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_qa(path: Path) -> tuple[dict[str, list[dict]], Counter]:
    result = defaultdict(list)
    counts = Counter()
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            media = row.get("video") or ""
            source = media.split("/", 1)[0] if "/" in media else ""
            counts[source] += 1
            if not media.startswith("scannetppv2/"):
                continue
            result[Path(media).stem].append({
                "vsi_row_index": row_index,
                "vsi_media": media,
                "question_type": row.get("question_type"),
                "conversations": row["conversations"],
            })
    return result, counts


def video_index(report: dict) -> dict[str, dict]:
    result = {}
    for row in report["videos"]:
        if row.get("status") == "ok" and str(row.get("media", "")).startswith("scannetppv2/"):
            result[Path(row["media"]).stem] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--video-metadata", required=True, type=Path)
    parser.add_argument("--rasterizer-library", required=True, type=Path)
    parser.add_argument("--rasterizer-source", required=True, type=Path)
    parser.add_argument("--instance-assignment-source", required=True, type=Path)
    parser.add_argument("--label-normalization-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive-output", required=True, type=Path)
    parser.add_argument("--base-interval", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--scene-limit", type=int)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.archive_output.exists():
        raise FileExistsError(args.archive_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection = load_json(args.selection_manifest)
    scenes = list(selection["selected_scenes"])
    if args.scene_id:
        requested = set(args.scene_id)
        missing = requested - set(scenes)
        if missing:
            raise ValueError(f"requested scenes absent from selection: {sorted(missing)}")
        scenes = [scene for scene in scenes if scene in requested]
    if args.scene_limit is not None:
        scenes = scenes[: args.scene_limit]
    if not scenes:
        raise ValueError("no scenes selected")

    qa_by_scene, source_counts = load_qa(args.jsonl)
    videos = video_index(load_json(args.video_metadata))
    rasterizer = Rasterizer(args.rasterizer_library)
    rasterizer_source_sha = sha256_file(args.rasterizer_source)
    instance_assignment_source_sha = sha256_file(args.instance_assignment_source)
    label_normalization_source_sha = sha256_file(args.label_normalization_source)
    scene_rows, frame_rows, qa_rows, certificates = [], [], [], []
    errors = []
    start_time = time.time()

    for offset, scene_id in enumerate(scenes, 1):
        print(f"[{offset}/{len(scenes)}] {scene_id}", flush=True)
        try:
            scene_root = args.data_root / "data" / scene_id
            paths = {name: scene_root / relative for name, relative in REQUIRED_ASSETS.items()}
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing assets: {missing}")
            if scene_id not in videos or scene_id not in qa_by_scene:
                raise ValueError("scene lacks video metadata or VSI QA")
            video = videos[scene_id]
            total_frames = int(video["frame_count"])
            fps = float(video["avg_fps"])
            indices = guide_frame_indices(
                total_frames, fps,
                base_interval=args.base_interval,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
            )
            frame_keys = [f"{scene_id}/frame_{index:06d}" for index in indices]
            vsi_media = f"scannetppv2/{scene_id}.mp4"
            sampling_binding = guide_sampling_binding_sha256(
                source_dataset="scannetppv2", scene_id=scene_id,
                vsi_media=vsi_media, frame_keys=frame_keys,
                frame_indices=indices, total_frames=total_frames, fps=fps,
                base_interval=args.base_interval, min_frames=args.min_frames,
                max_frames=args.max_frames,
                sampling_policy=GUIDE_WHOLE_MP4_SAMPLING_POLICY,
            )

            mesh = load_mesh_instances(
                paths["mesh"], paths["segments"], paths["annotation"]
            )
            poses = load_json(paths["pose"])
            exif = load_json(paths["exif"])
            if len(poses) != total_frames or len(exif) != total_frames:
                raise ValueError("pose/EXIF/video frame counts differ")
            first_exif = next(iter(exif.values()))
            official_width = int(first_exif["PixelXDimension"])
            official_height = int(first_exif["PixelYDimension"])
            width, height = int(video["width"]), int(video["height"])
            scale_x, scale_y = width / official_width, height / official_height

            nodes = []
            for group in mesh.annotation_groups:
                obb = group["obb"]
                nodes.append({
                    "object_id": str(group["index"]),
                    "category": normalize_scannetppv2_label(group["label"]),
                    "source_category": group["label"],
                    "center_world_m": obb["centroid"],
                    "extent_m": obb["axesLengths"],
                    "rotation_world_from_object": np.asarray(
                        obb["normalizedAxes"], dtype=float
                    ).reshape(3, 3).tolist(),
                    "motion_type": "static",
                    "identity_source": "scannetppv2_segGroups_index",
                    "geometry_valid": True,
                })
            scene_rows.append({
                "schema_version": "scannetppv2_scene_state_v1",
                "scene_id": scene_id,
                "vsi_media": vsi_media,
                "coordinate_frame": "scannetppv2_aligned_mesh_z_up_m",
                "supervision_tier": "gold",
                "identity_source": "scannetppv2_segGroups_index",
                "nodes": nodes,
            })

            certificate_frames = []
            for frame_key, index in zip(frame_keys, indices):
                pose = poses[f"frame_{index:06d}"]
                world_from_camera = np.asarray(pose["aligned_pose"], dtype=float)
                camera_from_world = np.linalg.inv(world_from_camera)
                intrinsic = np.asarray(pose["intrinsic"], dtype=float).copy()
                intrinsic[0, :] *= scale_x
                intrinsic[1, :] *= scale_y
                labels = rasterizer.render(
                    mesh, camera_from_world, intrinsic, width, height
                )
                counts = instance_pixel_counts(labels)
                visible_nodes = [{
                    "object_id": str(instance_id),
                    "visible": True,
                    "geometry_valid": True,
                    "pixel_count": pixel_count,
                } for instance_id, pixel_count in sorted(counts.items())]
                frame_rows.append({
                    "schema_version": "scannetppv2_frame_state_v1",
                    "scene_id": scene_id,
                    "frame_key": frame_key,
                    "frame_index": index,
                    "vsi_media": vsi_media,
                    "rotation_world_from_camera": world_from_camera[:3, :3].tolist(),
                    "translation_world_from_camera_m": world_from_camera[:3, 3].tolist(),
                    "camera_projection": intrinsic.tolist(),
                    "visible_nodes": visible_nodes,
                    "supervision_tier": "gold",
                })
                certificate_frames.append({
                    "frame_index": index,
                    "frame_key": frame_key,
                    "pose_sha256": content_sha256(pose["aligned_pose"]),
                    "intrinsic_sha256": content_sha256(intrinsic.tolist()),
                    "instance_mask_sha256": hashlib.sha256(
                        np.ascontiguousarray(labels, dtype="<i4").tobytes()
                    ).hexdigest(),
                    "visible_instance_pixel_counts": {
                        str(key): value for key, value in sorted(counts.items())
                    },
                })

            source_assets = {name: artifact(path) for name, path in paths.items()}
            certificate = build_support_certificate(
                scene_id=scene_id, vsi_media=vsi_media,
                sampling_binding_sha256=sampling_binding,
                video_total_frames=total_frames, video_fps=fps,
                video_width=width, video_height=height,
                source_assets=source_assets,
                video_metadata_sha256=sha256_file(args.video_metadata),
                instance_assignment_source_sha256=instance_assignment_source_sha,
                label_normalization_source_sha256=label_normalization_source_sha,
                rasterizer_source_sha256=rasterizer_source_sha,
                rasterizer_library_sha256=rasterizer.library_sha256,
                frames=certificate_frames,
            )
            certificates.append(certificate)
            for qa in qa_by_scene[scene_id]:
                qa_rows.append({
                    "schema_version": "scannetppv2_qa_train_v1",
                    "source_dataset": "scannetppv2",
                    "scene_id": scene_id,
                    "vsi_row_index": qa["vsi_row_index"],
                    "vsi_media": qa["vsi_media"],
                    "question_type": qa["question_type"],
                    "conversations": qa["conversations"],
                    "candidate_frame_keys": frame_keys,
                    "candidate_frame_indices": indices,
                    "qa_evidence_scope": "scene_associated_unlocalized",
                    "qa_visual_support_verified": False,
                    "evidence_frame_indices": None,
                    "duration_coverage_ratio": 1.0,
                    "coverage_bin": "high",
                    "loss_masks": {"scene_geometry": True},
                    "sampling_policy": GUIDE_WHOLE_MP4_SAMPLING_POLICY,
                    "total_frames": total_frames,
                    "fps": fps,
                    "base_interval": args.base_interval,
                    "min_frames": args.min_frames,
                    "max_frames": args.max_frames,
                    "sampling_binding_sha256": sampling_binding,
                    "clip_provenance": None,
                    "support_certificate_sha256": certificate["certificate_sha256"],
                })
        except Exception as error:
            errors.append({
                "scene_id": scene_id,
                "error_type": type(error).__name__,
                "error": str(error),
            })

    files = {
        "scenes": args.output_dir / "scannetppv2_scene_states.jsonl",
        "frames": args.output_dir / "scannetppv2_frame_states.jsonl",
        "qa": args.output_dir / "scannetppv2_qa_train.jsonl",
        "certificates": args.output_dir / "scannetppv2_support_certificates.jsonl",
    }
    write_jsonl(files["scenes"], scene_rows)
    write_jsonl(files["frames"], frame_rows)
    write_jsonl(files["qa"], qa_rows)
    write_jsonl(files["certificates"], certificates)
    registry = artifact(files["certificates"])
    report = {
        "schema_version": "scannetppv2_alignment_report_v1",
        "status": "complete_passed" if not errors and len(scene_rows) == len(scenes) else "complete_failed",
        "requested_scenes": len(scenes),
        "completed_scenes": len(scene_rows),
        "frames": len(frame_rows),
        "qa_rows": len(qa_rows),
        "nodes": sum(len(row["nodes"]) for row in scene_rows),
        "visible_node_observations": sum(len(row["visible_nodes"]) for row in frame_rows),
        "source_counts": dict(source_counts),
        "errors": errors,
        "sampling": {
            "policy": GUIDE_WHOLE_MP4_SAMPLING_POLICY,
            "base_interval": args.base_interval,
            "min_frames": args.min_frames,
            "max_frames": args.max_frames,
        },
        "support_certificate_registry": {
            "path": files["certificates"].name,
            "sha256": registry["sha256"],
            "trust_stage": "scannetppv2_finalizer_output_v1",
        },
        "rasterizer": {
            "source_sha256": rasterizer_source_sha,
            "library_sha256": rasterizer.library_sha256,
        },
        "instance_assignment_source_sha256": instance_assignment_source_sha,
        "label_normalization_source_sha256": label_normalization_source_sha,
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "video_metadata_sha256": sha256_file(args.video_metadata),
        "elapsed_seconds": time.time() - start_time,
        "files": {name: artifact(path) for name, path in files.items()},
    }
    report_path = args.output_dir / "scannetppv2_alignment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete_passed":
        raise SystemExit(1)
    with tarfile.open(args.archive_output, "w:gz") as archive:
        for path in [*files.values(), report_path]:
            archive.add(path, arcname=path.name)
    print(f"Archive: {args.archive_output}")


if __name__ == "__main__":
    main()
