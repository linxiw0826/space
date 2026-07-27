#!/usr/bin/env python3
"""Inventory files needed for direct-ID VSI-590K Gold preprocessing.

This is a read-only, file-presence audit.  It intentionally does not claim
that a source is joinable merely because files with promising names exist.
The next audit stage must inspect schemas and measure scene/frame/instance
join coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


SOURCES = ("scannet", "adt", "scannetppv2", "hypersim", "procthor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum example paths retained for each file class.",
    )
    return parser.parse_args()


def file_record(path: Path) -> dict:
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "exists": False}
    record = {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
    return record


def count_files(root: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not root.is_dir():
        return 0
    total = 0
    for directory, _, filenames in os.walk(root):
        del directory
        if suffixes is None:
            total += len(filenames)
        else:
            total += sum(name.lower().endswith(suffixes) for name in filenames)
    return total


def collect_named_files(
    roots: list[Path],
    *,
    contains: tuple[str, ...],
    suffixes: tuple[str, ...] = (),
    max_examples: int,
) -> dict:
    examples: list[str] = []
    count = 0
    lowered_contains = tuple(token.lower() for token in contains)
    lowered_suffixes = tuple(suffix.lower() for suffix in suffixes)

    for root in roots:
        if not root.is_dir():
            continue
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                lowered = filename.lower()
                if lowered_contains and not any(
                    token in lowered for token in lowered_contains
                ):
                    continue
                if lowered_suffixes and not lowered.endswith(lowered_suffixes):
                    continue
                count += 1
                if len(examples) < max_examples:
                    examples.append(str(Path(directory) / filename))
    return {"count": count, "examples": examples}


def media_counts(data_root: Path) -> dict:
    media_root = data_root / "VSI-590K"
    output = {}
    for source in SOURCES:
        root = media_root / source
        output[source] = {
            "root": str(root),
            "exists": root.is_dir(),
            "mp4_count": count_files(root, (".mp4",)),
        }
    return output


def source_lists(data_root: Path) -> dict:
    root = data_root / "vsi590k_gold_source_lists"
    names = {
        "scannet": None,
        "adt": "adt_sequences.txt",
        "scannetppv2": "scannetppv2_scenes.txt",
        "hypersim": "hypersim_items.txt",
        "procthor": "procthor_house_ids.txt",
    }
    output = {"root": str(root), "sources": {}}
    for source, name in names.items():
        if name is None:
            output["sources"][source] = {
                "note": "ScanNet IDs are obtained directly from VSI JSONL."
            }
            continue
        path = root / name
        record = file_record(path)
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                record["nonempty_lines"] = sum(
                    bool(line.strip()) for line in handle
                )
        output["sources"][source] = record
    return output


def core_released_files(data_root: Path) -> dict:
    vsi_root = data_root / "VSI-590K"
    meta_root = (
        data_root / "vsi590k_parta_audit" / "VSI-590K-MetaInfo"
    )
    paths = {
        "vsi_jsonl": vsi_root / "vsi_590k.jsonl",
        "metainfo_zip": meta_root / "metainfo_mega.zip",
        "adt_metainfo": (
            meta_root / "adt_video_meta_info_jy_20250430.json"
        ),
        "hypersim_metainfo": meta_root / "hypersim_meta_info.json",
        "scannet_frame_info": (
            meta_root / "scannet_train_frame_category_info_20250304.npy"
        ),
        "scannetppv2_frame_info": (
            meta_root
            / "scannetpp_v2_train_frame_category_info_20250420.npy"
        ),
    }
    return {name: file_record(path) for name, path in paths.items()}


def source_native_inventory(
    data_root: Path, max_examples: int
) -> dict:
    gold_raw = data_root / "vsi590k_gold_raw"
    guide_scannet = (
        data_root / "guide_repro" / "media" / "spar" / "scannet"
    )
    hypersim_roots = [
        gold_raw / "hypersim",
        data_root / "hypersim",
    ]
    adt_roots = [gold_raw / "adt", data_root / "adt"]
    scannetpp_roots = [
        gold_raw / "scannetppv2",
        gold_raw / "scannetpp",
        data_root / "scannetppv2",
    ]
    scannet_roots = [
        gold_raw / "scannet",
        guide_scannet,
    ]
    procthor_root = gold_raw / "procthor10k"

    return {
        "scannet": {
            "candidate_roots": [
                file_record(path) for path in scannet_roots
            ],
            "aggregation": collect_named_files(
                scannet_roots,
                contains=(".aggregation",),
                max_examples=max_examples,
            ),
            "segmentation": collect_named_files(
                scannet_roots,
                contains=(".segs", "instance", "label"),
                suffixes=(".json", ".png", ".ply"),
                max_examples=max_examples,
            ),
            "sensor_or_pose": collect_named_files(
                scannet_roots,
                contains=(".sens", "pose", "intrinsic"),
                max_examples=max_examples,
            ),
            "axis_alignment": collect_named_files(
                scannet_roots,
                contains=("axisalign",),
                max_examples=max_examples,
            ),
        },
        "adt": {
            "candidate_roots": [file_record(path) for path in adt_roots],
            "trajectory_or_pose": collect_named_files(
                adt_roots,
                contains=("trajectory", "pose", "aria"),
                max_examples=max_examples,
            ),
            "instance_or_bbox": collect_named_files(
                adt_roots,
                contains=("instance", "object", "bbox", "bounding"),
                max_examples=max_examples,
            ),
            "depth_or_mesh": collect_named_files(
                adt_roots,
                contains=("depth", "mesh", "point"),
                max_examples=max_examples,
            ),
        },
        "scannetppv2": {
            "candidate_roots": [
                file_record(path) for path in scannetpp_roots
            ],
            "pose_or_camera": collect_named_files(
                scannetpp_roots,
                contains=("pose", "transform", "camera", "colmap"),
                max_examples=max_examples,
            ),
            "instance_or_segments": collect_named_files(
                scannetpp_roots,
                contains=("instance", "segment", "semantic", "anno"),
                max_examples=max_examples,
            ),
            "depth_or_mesh": collect_named_files(
                scannetpp_roots,
                contains=("depth", "mesh", "point", "scan"),
                max_examples=max_examples,
            ),
        },
        "hypersim": {
            "candidate_roots": [
                file_record(path) for path in hypersim_roots
            ],
            "camera": collect_named_files(
                hypersim_roots,
                contains=(
                    "camera_keyframe_positions",
                    "camera_keyframe_orientations",
                    "metadata_camera",
                ),
                max_examples=max_examples,
            ),
            "semantic_instance": collect_named_files(
                hypersim_roots,
                contains=("semantic_instance",),
                max_examples=max_examples,
            ),
            "bbox": collect_named_files(
                hypersim_roots,
                contains=("bounding_box",),
                max_examples=max_examples,
            ),
            "depth": collect_named_files(
                hypersim_roots,
                contains=("depth_meters", "depth"),
                suffixes=(".hdf5", ".h5"),
                max_examples=max_examples,
            ),
        },
        "procthor": {
            "candidate_root": file_record(procthor_root),
            "train_jsonl": file_record(procthor_root / "train.jsonl.gz"),
            "val_jsonl": file_record(procthor_root / "val.jsonl.gz"),
            "test_jsonl": file_record(procthor_root / "test.jsonl.gz"),
            "trajectory_or_actions": collect_named_files(
                [procthor_root, data_root / "VSI-590K" / "procthor"],
                contains=("trajectory", "action", "pose", "event"),
                max_examples=max_examples,
            ),
        },
    }


def readiness_summary(native: dict) -> dict:
    checks = {
        "scannet": {
            "direct_id_files_present": (
                native["scannet"]["aggregation"]["count"] > 0
                and native["scannet"]["segmentation"]["count"] > 0
            ),
            "camera_files_present": (
                native["scannet"]["sensor_or_pose"]["count"] > 0
            ),
        },
        "adt": {
            "direct_id_files_present": (
                native["adt"]["instance_or_bbox"]["count"] > 0
            ),
            "camera_files_present": (
                native["adt"]["trajectory_or_pose"]["count"] > 0
            ),
            "geometry_files_present": (
                native["adt"]["depth_or_mesh"]["count"] > 0
            ),
        },
        "scannetppv2": {
            "direct_id_files_present": (
                native["scannetppv2"]["instance_or_segments"]["count"] > 0
            ),
            "camera_files_present": (
                native["scannetppv2"]["pose_or_camera"]["count"] > 0
            ),
            "geometry_files_present": (
                native["scannetppv2"]["depth_or_mesh"]["count"] > 0
            ),
        },
        "hypersim": {
            "direct_id_files_present": (
                native["hypersim"]["semantic_instance"]["count"] > 0
                and native["hypersim"]["bbox"]["count"] > 0
            ),
            "camera_files_present": (
                native["hypersim"]["camera"]["count"] > 0
            ),
            "depth_files_present": (
                native["hypersim"]["depth"]["count"] > 0
            ),
        },
        "procthor": {
            "house_json_present": all(
                native["procthor"][f"{split}_jsonl"]["exists"]
                for split in ("train", "val", "test")
            ),
            "video_trajectory_files_present": (
                native["procthor"]["trajectory_or_actions"]["count"] > 0
            ),
        },
    }
    return {
        source: {
            **values,
            "all_presence_checks_pass": all(values.values()),
            "caveat": (
                "Presence only. Direct scene/frame/instance joins must still "
                "be verified before training."
            ),
        }
        for source, values in checks.items()
    }


def main() -> None:
    args = parse_args()
    native = source_native_inventory(args.data_root, args.max_examples)
    report = {
        "schema_version": "vsi590k_gold_download_audit_v1",
        "scope": (
            "Read-only presence audit for source-native files needed to "
            "replace Hungarian instance-to-bbox recovery with direct IDs."
        ),
        "data_root": str(args.data_root),
        "released_files": core_released_files(args.data_root),
        "source_lists": source_lists(args.data_root),
        "vsi_media": media_counts(args.data_root),
        "source_native": native,
        "readiness": readiness_summary(native),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    printable = {
        "schema_version": report["schema_version"],
        "readiness": report["readiness"],
        "vsi_media": report["vsi_media"],
        "output": str(args.output),
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
