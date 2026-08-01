#!/usr/bin/env python3
"""Project official ScanNet++ OBBs onto extracted VSI MP4 pilot frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = "scannetppv2_pixel_projection_qc_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"expected finite {shape}, got {array.shape}")
    return array


def obb_corners(obb: dict) -> np.ndarray:
    center = finite(obb["centroid"], (3,))
    lengths = finite(obb["axesLengths"], (3,))
    axes = finite(obb["normalizedAxes"], (9,)).reshape(3, 3)
    signs = np.asarray(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=np.float64,
    )
    return center + (signs * lengths[None, :] / 2.0) @ axes.T


def project(points: np.ndarray, camera_from_world: np.ndarray, k: np.ndarray):
    homogeneous = np.c_[points, np.ones(len(points))]
    camera = (camera_from_world @ homogeneous.T).T[:, :3]
    pixels_h = (k @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return camera[:, 2], pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--frame-bundle", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--visual-review-passed", action="store_true")
    args = parser.parse_args()

    scene_root = args.data_root / "data" / args.scene_id
    annotation_path = scene_root / "scans/segments_anno.json"
    pose_path = scene_root / "iphone/pose_intrinsic_imu.json"
    manifest_path = args.frame_bundle / "manifest.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    poses = json.loads(pose_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["scene_id"] != args.scene_id:
        raise ValueError("frame bundle scene does not match requested scene")
    if manifest["candidate_frame_mapping"] != "vsi_mp4_frame_i_to_iphone_frame_i":
        raise ValueError("frame bundle does not use the frozen identity mapping")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in manifest["frames"]:
        index = int(frame["vsi_frame_index"])
        if int(frame["candidate_official_iphone_frame_index"]) != index:
            raise ValueError("frame manifest violates identity mapping")
        image_path = args.frame_bundle / frame["path"]
        if sha256_file(image_path) != frame["sha256"]:
            raise ValueError(f"frame hash mismatch: {image_path}")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        record = poses[f"frame_{index:06d}"]
        world_from_camera = finite(record["aligned_pose"], (4, 4))
        camera_from_world = np.linalg.inv(world_from_camera)
        intrinsic = finite(record["intrinsic"], (3, 3)).copy()
        scale_x = width / 1920.0
        scale_y = height / 1440.0
        intrinsic[0, :] *= scale_x
        intrinsic[1, :] *= scale_y

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        in_front = in_image = boxes_drawn = 0
        for group in annotation["segGroups"]:
            center = finite(group["obb"]["centroid"], (3,))[None, :]
            center_depth, center_pixel = project(center, camera_from_world, intrinsic)
            if center_depth[0] <= 0:
                continue
            in_front += 1
            u, v = center_pixel[0]
            if not (0 <= u < width and 0 <= v < height):
                continue
            in_image += 1
            corners = obb_corners(group["obb"])
            corner_depth, corner_pixels = project(corners, camera_from_world, intrinsic)
            valid = corner_depth > 1e-6
            if valid.any():
                visible = corner_pixels[valid]
                x0, y0 = visible.min(axis=0)
                x1, y1 = visible.max(axis=0)
                x0, x1 = np.clip([x0, x1], 0, width - 1)
                y0, y1 = np.clip([y0, y1], 0, height - 1)
                if x1 > x0 and y1 > y0:
                    draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 0), width=2)
                    boxes_drawn += 1
            radius = 4
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 0, 0))
            draw.text((u + 5, v + 2), f"{group['index']}:{group['label']}", fill=(255, 255, 0))

        output = args.output_dir / f"frame_{index:06d}_overlay.png"
        overlay.save(output)
        rows.append(
            {
                "frame_index": index,
                "input": str(image_path.resolve()),
                "input_sha256": sha256_file(image_path),
                "overlay": str(output.resolve()),
                "overlay_sha256": sha256_file(output),
                "object_count": len(annotation["segGroups"]),
                "centers_in_front": in_front,
                "centers_in_image": in_image,
                "boxes_drawn": boxes_drawn,
                "intrinsic_scale_x": scale_x,
                "intrinsic_scale_y": scale_y,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_passed"
            if args.visual_review_passed
            else "generated_pending_visual_review"
        ),
        "visual_review_passed": args.visual_review_passed,
        "scene_id": args.scene_id,
        "mapping": "vsi_mp4_frame_i_to_iphone_frame_i",
        "annotation_sha256": sha256_file(annotation_path),
        "pose_sha256": sha256_file(pose_path),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "frames": rows,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
