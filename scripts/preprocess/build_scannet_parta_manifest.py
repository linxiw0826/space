#!/usr/bin/env python3
"""Build a high-confidence VSI-590K ScanNet Part A scene manifest."""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


DEFAULT_META_MEMBER = (
    "fianlver-vsibench/scannet_train_meta_info-20250130.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--frame-info-npy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-member", default=DEFAULT_META_MEMBER)
    parser.add_argument("--scene-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-fully-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(child) for child in value]
    return value


def load_qa(path):
    by_scene = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row = json.loads(line)
            media = row.get("video") or ""
            if not media.startswith("scannet/"):
                continue
            scene = Path(media).stem
            by_scene[scene].append({
                "row_index": row_index,
                "video": media,
                "question_type": row.get("question_type"),
                "conversations": row.get("conversations"),
            })
    return by_scene


def load_metadata(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return json.load(handle)


def frame_count(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def read_matrix(path):
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid 4x4 matrix: {path} shape={matrix.shape}")
    return matrix.tolist()


def collect_visible_ids(records):
    visible_ids = defaultdict(set)
    id_categories = defaultdict(set)
    for record in records:
        for category, payload in record.items():
            for raw_id in np.asarray(payload.get("inst_ids", [])).reshape(-1):
                instance_id = int(raw_id)
                visible_ids[category].add(instance_id)
                id_categories[instance_id].add(category)
    return visible_ids, id_categories


def exact_category_mapping(scene_meta, records):
    boxes_by_category = (
        scene_meta.get("object_bbox")
        or scene_meta.get("object_bboxes")
        or {}
    )
    visible_ids, id_categories = collect_visible_ids(records)
    exact = {}
    rejected = {}

    for category, boxes in boxes_by_category.items():
        ids = sorted(visible_ids.get(category, set()))
        if len(ids) == len(boxes):
            exact[category] = {
                instance_id: to_builtin(box)
                for instance_id, box in zip(ids, boxes)
            }
        else:
            rejected[category] = {
                "reason": "instance_bbox_count_mismatch",
                "instance_ids": ids,
                "bbox_count": len(boxes),
            }

    for category in sorted(set(visible_ids) - set(boxes_by_category)):
        rejected[category] = {
            "reason": "frame_category_without_bbox",
            "instance_ids": sorted(visible_ids[category]),
            "bbox_count": 0,
        }

    conflicts = {
        str(instance_id): sorted(categories)
        for instance_id, categories in id_categories.items()
        if len(categories) > 1
    }
    fully_exact = not rejected and not conflicts
    return exact, rejected, conflicts, fully_exact


def selected_visibility(record, exact_mapping):
    result = []
    for category, payload in record.items():
        allowed = exact_mapping.get(category)
        if not allowed:
            continue
        ids = [int(value) for value in np.asarray(
            payload.get("inst_ids", [])
        ).reshape(-1)]
        counts = [int(value) for value in np.asarray(
            payload.get("inst_num_pixels", [])
        ).reshape(-1)]
        for instance_id, pixels in zip(ids, counts):
            if instance_id in allowed:
                result.append({
                    "node_id": f"{category}:{instance_id}",
                    "category": category,
                    "instance_id": instance_id,
                    "visible_pixels": pixels,
                })
    return result


def build_scene(
    scene,
    qa_rows,
    scene_meta,
    records,
    video_root,
    geometry_root,
):
    video_path = video_root / f"{scene}.mp4"
    n_frames = frame_count(video_path)
    if n_frames is None or n_frames != len(records):
        raise ValueError(
            f"{scene}: video frames={n_frames}, NPY records={len(records)}"
        )

    exact, rejected, conflicts, fully_exact = exact_category_mapping(
        scene_meta, records
    )
    pose_dir = geometry_root / scene / "video_pose"
    pose_ids = sorted(int(path.stem) for path in pose_dir.glob("*.txt"))
    if len(pose_ids) != 32:
        raise ValueError(f"{scene}: expected 32 video poses, got {len(pose_ids)}")

    step = n_frames / 8
    sft_indices = [int(step * index + step / 2) for index in range(8)]
    expected_positions = list(range(2, 32, 4))
    if [pose_ids[index] for index in expected_positions] != sft_indices:
        raise ValueError(f"{scene}: 8-of-32 mapping mismatch")

    intrinsic_path = (
        geometry_root / scene / "intrinsic" / "intrinsic_color.txt"
    )
    nodes = []
    for category, instances in exact.items():
        for instance_id, bbox in instances.items():
            nodes.append({
                "node_id": f"{category}:{instance_id}",
                "category": category,
                "instance_id": instance_id,
                "bbox_3d": bbox,
                "supervision": "gold_exact_order_recovery",
            })

    views = []
    for candidate_index, source_frame_index in enumerate(pose_ids):
        pose_path = pose_dir / f"{source_frame_index}.txt"
        views.append({
            "candidate_index": candidate_index,
            "mp4_frame_index": source_frame_index,
            "pose_path": str(pose_path),
            "camera_to_world": read_matrix(pose_path),
            "is_sft_view": source_frame_index in sft_indices,
            "visible_nodes": selected_visibility(
                records[source_frame_index], exact
            ),
        })

    return {
        "schema_version": "scannet_parta_manifest_v1",
        "scene_id": scene,
        "dataset": "scannet",
        "video_path": str(video_path),
        "video_frame_count": n_frames,
        "intrinsic_path": str(intrinsic_path),
        "intrinsic_color": read_matrix(intrinsic_path),
        "room_size": scene_meta.get("room_size"),
        "room_center": scene_meta.get("room_center"),
        "fully_exact": fully_exact,
        "nodes": nodes,
        "rejected_categories": rejected,
        "instance_id_category_conflicts": conflicts,
        "candidate_views": views,
        "sft_mp4_frame_indices": sft_indices,
        "qa": qa_rows,
    }


def main():
    args = parse_args()
    qa_by_scene = load_qa(args.jsonl)
    metadata = load_metadata(args.metadata_zip, args.metadata_member)
    frame_info = np.load(args.frame_info_npy, allow_pickle=True).item()

    eligible = []
    scene_audit = {}
    for scene in sorted(qa_by_scene):
        if scene not in metadata or scene not in frame_info:
            continue
        exact, rejected, conflicts, fully_exact = exact_category_mapping(
            metadata[scene], frame_info[scene]
        )
        scene_audit[scene] = {
            "fully_exact": fully_exact,
            "exact_categories": len(exact),
            "rejected_categories": len(rejected),
            "conflicts": len(conflicts),
            "qa_rows": len(qa_by_scene[scene]),
        }
        if fully_exact or not args.require_fully_exact:
            eligible.append(scene)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if args.scene_limit > 0:
        selected = sorted(eligible[:args.scene_limit])
    else:
        selected = sorted(eligible)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_qa = 0
    total_nodes = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for scene in selected:
            item = build_scene(
                scene=scene,
                qa_rows=qa_by_scene[scene],
                scene_meta=metadata[scene],
                records=frame_info[scene],
                video_root=args.video_root,
                geometry_root=args.geometry_root,
            )
            total_qa += len(item["qa"])
            total_nodes += len(item["nodes"])
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": "scannet_parta_manifest_summary_v1",
        "seed": args.seed,
        "scene_limit": args.scene_limit,
        "require_fully_exact": args.require_fully_exact,
        "qa_scenes": len(qa_by_scene),
        "eligible_scenes": len(eligible),
        "selected_scenes": len(selected),
        "selected_qa_rows": total_qa,
        "selected_nodes": total_nodes,
        "output": str(args.output),
        "selected_scene_ids": selected,
        "scene_audit": scene_audit,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps({
        key: value for key, value in summary.items()
        if key not in {"selected_scene_ids", "scene_audit"}
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
