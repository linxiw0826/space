#!/usr/bin/env python3
"""Audit the released VSI-590K ScanNet++ V2 inputs before source download.

The frame-category NPY is a pickled scalar object array.  Loading it is unsafe
for untrusted files, so callers must explicitly acknowledge that the artifact
came from the trusted VSI-590K release with ``--allow-pickled-npy``.  The audit
records the artifact digest and fails closed on scene-set or schema mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "scannetppv2_vsi_input_audit_v1"
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.int64), q))


def parse_scannetpp_media(media: Any) -> str | None:
    if not isinstance(media, str):
        return None
    path = PurePosixPath(media)
    if len(path.parts) != 2 or path.parts[0] != "scannetppv2":
        return None
    if path.suffix.lower() != ".mp4" or not path.stem:
        return None
    return path.stem


def load_scene_list(path: Path) -> tuple[list[str], list[str]]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    invalid = [value for value in values if SCENE_ID_RE.fullmatch(value) is None]
    return values, invalid


def audit_jsonl(path: Path) -> dict[str, Any]:
    total_rows = 0
    scannetpp_rows = 0
    malformed_media_rows = 0
    scene_counts: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    conversation_shape_errors = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            media = row.get("video") or row.get("image")
            if not isinstance(media, str) or not media.startswith("scannetppv2/"):
                continue
            scannetpp_rows += 1
            scene_id = parse_scannetpp_media(media)
            if scene_id is None:
                malformed_media_rows += 1
            else:
                scene_counts[scene_id] += 1
            question_types[str(row.get("question_type", ""))] += 1
            conversations = row.get("conversations")
            if (
                not isinstance(conversations, list)
                or len(conversations) != 2
                or [turn.get("from") for turn in conversations if isinstance(turn, dict)]
                != ["human", "gpt"]
            ):
                conversation_shape_errors += 1
    counts = list(scene_counts.values())
    return {
        "total_rows": total_rows,
        "scannetpp_rows": scannetpp_rows,
        "scenes": sorted(scene_counts),
        "scene_count": len(scene_counts),
        "malformed_media_rows": malformed_media_rows,
        "conversation_shape_errors": conversation_shape_errors,
        "qa_per_scene": {
            "min": min(counts) if counts else None,
            "median": percentile(counts, 50),
            "max": max(counts) if counts else None,
        },
        "question_type_counts": dict(sorted(question_types.items())),
        "scene_qa_counts": dict(sorted(scene_counts.items())),
    }


def _as_integer_vector(value: Any) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        return None
    return array


def audit_frame_metainfo(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise ValueError(
            "frame metainfo must be a scalar object NPY; "
            f"got shape={raw.shape}, dtype={raw.dtype}"
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(f"frame metainfo root must be dict, got {type(payload).__name__}")

    schema_errors: Counter[str] = Counter()
    frame_counts: list[int] = []
    category_observations = 0
    instance_observations = 0
    empty_frames = 0
    instance_id_min: int | None = None
    instance_id_max: int | None = None
    categories: Counter[str] = Counter()
    invalid_scenes: list[dict[str, Any]] = []

    for scene_id, frames in payload.items():
        if not isinstance(scene_id, str):
            schema_errors["non_string_scene_id"] += 1
            invalid_scenes.append(
                {"scene_id": repr(scene_id), "reason": "non_string_scene_id"}
            )
            continue
        if not isinstance(frames, list) or not frames:
            schema_errors["frames_not_nonempty_list"] += 1
            invalid_scenes.append(
                {
                    "scene_id": scene_id,
                    "reason": "frames_not_nonempty_list",
                    "value_type": type(frames).__name__,
                    "frame_count": len(frames) if isinstance(frames, list) else None,
                }
            )
            continue
        frame_counts.append(len(frames))
        for frame in frames:
            if not isinstance(frame, dict):
                schema_errors["frame_not_dict"] += 1
                continue
            if not frame:
                empty_frames += 1
            frame_instance_ids: list[int] = []
            for category, record in frame.items():
                category_observations += 1
                categories[str(category)] += 1
                if not isinstance(category, str) or not category:
                    schema_errors["invalid_category"] += 1
                if not isinstance(record, dict):
                    schema_errors["category_record_not_dict"] += 1
                    continue
                if set(record) != {"num_pixels", "inst_ids", "inst_num_pixels"}:
                    schema_errors["category_record_keys"] += 1
                    continue
                num_pixels = record["num_pixels"]
                inst_ids = _as_integer_vector(record["inst_ids"])
                inst_pixels = _as_integer_vector(record["inst_num_pixels"])
                if not isinstance(num_pixels, (int, np.integer)) or int(num_pixels) < 0:
                    schema_errors["invalid_num_pixels"] += 1
                if inst_ids is None:
                    schema_errors["invalid_inst_ids"] += 1
                if inst_pixels is None or (inst_pixels < 0).any():
                    schema_errors["invalid_inst_num_pixels"] += 1
                if inst_ids is None or inst_pixels is None:
                    continue
                if len(inst_ids) != len(inst_pixels):
                    schema_errors["instance_vector_length_mismatch"] += 1
                    continue
                if len(np.unique(inst_ids)) != len(inst_ids):
                    schema_errors["duplicate_instance_within_category"] += 1
                if int(inst_pixels.sum()) != int(num_pixels):
                    schema_errors["category_pixel_sum_mismatch"] += 1
                instance_observations += len(inst_ids)
                frame_instance_ids.extend(int(value) for value in inst_ids)
                if len(inst_ids):
                    local_min = int(inst_ids.min())
                    local_max = int(inst_ids.max())
                    instance_id_min = local_min if instance_id_min is None else min(instance_id_min, local_min)
                    instance_id_max = local_max if instance_id_max is None else max(instance_id_max, local_max)
            if len(frame_instance_ids) != len(set(frame_instance_ids)):
                schema_errors["duplicate_instance_across_categories"] += 1

    return {
        "scene_count": len(payload),
        "scenes": sorted(str(value) for value in payload),
        "frame_index_contract": "implicit_zero_based_list_position",
        "frame_index_verified_against_mp4": False,
        "frame_count": sum(frame_counts),
        "frame_count_per_scene": {
            "min": min(frame_counts) if frame_counts else None,
            "p10": percentile(frame_counts, 10),
            "median": percentile(frame_counts, 50),
            "p90": percentile(frame_counts, 90),
            "max": max(frame_counts) if frame_counts else None,
        },
        "empty_frames": empty_frames,
        "category_observations": category_observations,
        "instance_observations": instance_observations,
        "instance_id_min": instance_id_min,
        "instance_id_max": instance_id_max,
        "unique_categories": len(categories),
        "category_observation_counts": dict(sorted(categories.items())),
        "schema_errors": dict(sorted(schema_errors.items())),
        "invalid_scenes": invalid_scenes,
    }


def difference(left: Iterable[str], right: Iterable[str]) -> list[str]:
    return sorted(set(left) - set(right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--frame-metainfo", required=True, type=Path)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-pickled-npy",
        action="store_true",
        help="Acknowledge that frame-metainfo is a trusted pickled NumPy artifact.",
    )
    args = parser.parse_args()
    if not args.allow_pickled_npy:
        parser.error("--allow-pickled-npy is required for the trusted VSI release artifact")
    for path in (args.jsonl, args.frame_metainfo, args.scenes):
        if not path.is_file():
            raise FileNotFoundError(path)

    scene_values, invalid_scene_ids = load_scene_list(args.scenes)
    duplicate_scene_ids = sorted(
        value for value, count in Counter(scene_values).items() if count > 1
    )
    jsonl = audit_jsonl(args.jsonl)
    metainfo = audit_frame_metainfo(args.frame_metainfo)
    listed = sorted(set(scene_values))
    comparisons = {
        "listed_missing_from_jsonl": difference(listed, jsonl["scenes"]),
        "jsonl_missing_from_listed": difference(jsonl["scenes"], listed),
        "listed_missing_from_metainfo": difference(listed, metainfo["scenes"]),
        "metainfo_missing_from_listed": difference(metainfo["scenes"], listed),
        "jsonl_missing_from_metainfo": difference(jsonl["scenes"], metainfo["scenes"]),
        "metainfo_missing_from_jsonl": difference(metainfo["scenes"], jsonl["scenes"]),
    }
    failures = []
    if invalid_scene_ids:
        failures.append("invalid_scene_ids")
    if duplicate_scene_ids:
        failures.append("duplicate_scene_ids")
    if any(comparisons.values()):
        failures.append("scene_set_mismatch")
    if jsonl["malformed_media_rows"]:
        failures.append("malformed_scannetpp_media")
    if jsonl["conversation_shape_errors"]:
        failures.append("conversation_shape_errors")
    if metainfo["schema_errors"]:
        failures.append("frame_metainfo_schema_errors")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "artifacts": {
            "jsonl": {
                "path": str(args.jsonl.resolve()),
                "size_bytes": args.jsonl.stat().st_size,
                "sha256": sha256_file(args.jsonl),
            },
            "frame_metainfo": {
                "path": str(args.frame_metainfo.resolve()),
                "size_bytes": args.frame_metainfo.stat().st_size,
                "sha256": sha256_file(args.frame_metainfo),
                "trusted_pickle_acknowledged": True,
            },
            "scenes": {
                "path": str(args.scenes.resolve()),
                "size_bytes": args.scenes.stat().st_size,
                "sha256": sha256_file(args.scenes),
            },
        },
        "scene_list": {
            "count": len(scene_values),
            "unique_count": len(listed),
            "invalid_ids": invalid_scene_ids,
            "duplicate_ids": duplicate_scene_ids,
        },
        "jsonl": jsonl,
        "frame_metainfo": metainfo,
        "scene_set_comparisons": comparisons,
        "unusable_scene_summary": [
            {
                **record,
                "qa_rows": jsonl["scene_qa_counts"].get(record["scene_id"]),
            }
            for record in metainfo["invalid_scenes"]
        ],
        "unresolved_contracts": [
            "metainfo list position must match zero-based VSI MP4 frame index",
            "metainfo inst_id must match ScanNet++ V2 segments_anno object identity",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
