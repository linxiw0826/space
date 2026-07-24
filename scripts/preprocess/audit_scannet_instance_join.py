#!/usr/bin/env python3
"""Audit whether VSI ScanNet frame instance IDs can recover 3D bbox identity."""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
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
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--frame-info-npy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-member", default=DEFAULT_META_MEMBER)
    parser.add_argument("--max-examples", type=int, default=30)
    return parser.parse_args()


def load_qa_scenes(path):
    scenes = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            media = row.get("video") or ""
            if media.startswith("scannet/"):
                scenes.add(Path(media).stem)
                rows += 1
    return scenes, rows


def load_metadata(path, member):
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return json.load(handle)


def video_frame_count(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def main():
    args = parse_args()
    qa_scenes, qa_rows = load_qa_scenes(args.jsonl)
    metadata = load_metadata(args.metadata_zip, args.metadata_member)
    frame_info = np.load(args.frame_info_npy, allow_pickle=True).item()

    category_status = Counter()
    frame_length_status = Counter()
    scene_status = Counter()
    total_bbox_instances = 0
    exactly_recoverable_instances = 0
    recoverable_visible_instances = 0
    examples = defaultdict(list)

    for scene in sorted(qa_scenes):
        scene_meta = metadata.get(scene)
        records = frame_info.get(scene)
        if scene_meta is None or records is None:
            scene_status["missing_metadata_or_frame_info"] += 1
            if len(examples["missing_scene"]) < args.max_examples:
                examples["missing_scene"].append(scene)
            continue

        frame_count = video_frame_count(args.video_root / f"{scene}.mp4")
        if frame_count is None:
            frame_length_status["invalid_video"] += 1
        elif frame_count == len(records):
            frame_length_status["exact"] += 1
        else:
            frame_length_status["mismatch"] += 1
            if len(examples["frame_length_mismatch"]) < args.max_examples:
                examples["frame_length_mismatch"].append({
                    "scene": scene,
                    "video_frames": frame_count,
                    "npy_records": len(records),
                })

        visible_ids = defaultdict(set)
        id_categories = defaultdict(set)
        for record in records:
            for category, payload in record.items():
                ids = payload.get("inst_ids", [])
                for raw_id in np.asarray(ids).reshape(-1):
                    instance_id = int(raw_id)
                    visible_ids[category].add(instance_id)
                    id_categories[instance_id].add(category)

        boxes_by_category = (
            scene_meta.get("object_bbox")
            or scene_meta.get("object_bboxes")
            or {}
        )
        scene_all_exact = True

        for category, boxes in boxes_by_category.items():
            bbox_count = len(boxes)
            ids = sorted(visible_ids.get(category, set()))
            id_count = len(ids)
            total_bbox_instances += bbox_count
            recoverable_visible_instances += min(bbox_count, id_count)

            if id_count == bbox_count:
                category_status["exact"] += 1
                exactly_recoverable_instances += bbox_count
            elif id_count < bbox_count:
                category_status["fewer_visible_ids_than_boxes"] += 1
                scene_all_exact = False
                if len(examples["fewer_ids"]) < args.max_examples:
                    examples["fewer_ids"].append({
                        "scene": scene,
                        "category": category,
                        "bbox_count": bbox_count,
                        "visible_ids": ids,
                    })
            else:
                category_status["more_visible_ids_than_boxes"] += 1
                scene_all_exact = False
                if len(examples["more_ids"]) < args.max_examples:
                    examples["more_ids"].append({
                        "scene": scene,
                        "category": category,
                        "bbox_count": bbox_count,
                        "visible_ids": ids,
                    })

        extra_categories = sorted(set(visible_ids) - set(boxes_by_category))
        if extra_categories:
            category_status["frame_category_without_bbox"] += len(extra_categories)
            scene_all_exact = False
            if len(examples["extra_categories"]) < args.max_examples:
                examples["extra_categories"].append({
                    "scene": scene,
                    "categories": extra_categories,
                })

        conflicts = {
            str(instance_id): sorted(categories)
            for instance_id, categories in id_categories.items()
            if len(categories) > 1
        }
        if conflicts:
            scene_status["instance_id_category_conflict"] += 1
            scene_all_exact = False
            if len(examples["id_category_conflicts"]) < args.max_examples:
                examples["id_category_conflicts"].append({
                    "scene": scene,
                    "conflicts": conflicts,
                })

        scene_status[
            "fully_exact" if scene_all_exact else "requires_filter_or_source_join"
        ] += 1

    report = {
        "qa_rows": qa_rows,
        "qa_scenes": len(qa_scenes),
        "metadata_scenes": len(metadata),
        "frame_info_scenes": len(frame_info),
        "frame_length_status": dict(frame_length_status),
        "category_status": dict(category_status),
        "scene_status": dict(scene_status),
        "total_bbox_instances": total_bbox_instances,
        "exactly_recoverable_instances": exactly_recoverable_instances,
        "recoverable_visible_instances_upper_bound": recoverable_visible_instances,
        "exact_instance_recovery_rate": (
            exactly_recoverable_instances / total_bbox_instances
            if total_bbox_instances else 0.0
        ),
        "examples": dict(examples),
        "mapping_assumption": (
            "Within an exact-count category, sorted visible instance IDs are paired "
            "with the published bbox list order. This follows the generator's "
            "np.unique(instance_id) ordering and must still be spot-checked."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps({
        key: value for key, value in report.items() if key != "examples"
    }, indent=2, ensure_ascii=False))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
