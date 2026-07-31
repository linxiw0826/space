#!/usr/bin/env python3
"""Fail closed unless A0/A1 manifests are matched on frozen QA inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (  # noqa: E402
    ContractError,
    content_sha256,
    read_jsonl,
    validate_records,
)


ARM_SPECIFIC_RUNTIME_FIELDS = frozenset(
    {
        "arm_name",
        "run_id",
        "state_head_enabled",
        "state_loss_enabled",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def causal_payload(row):
    return {
        key: value
        for key, value in row.items()
        if key not in ARM_SPECIFIC_RUNTIME_FIELDS
    }


def load_manifest(path: Path) -> tuple[list[dict], dict[str, dict]]:
    ordered = []
    rows = {}
    for row in read_jsonl(path):
        qa_id = row["qa_id"]
        if qa_id in rows:
            raise ContractError(f"Duplicate qa_id in {path}: {qa_id}")
        ordered.append(row)
        rows[qa_id] = row
    return ordered, rows


def coverage_counts(rows):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in rows:
        counts[row["source_dataset"]][
            str(row.get("question_type") or "__unknown__")
        ][row["coverage_bin"]] += 1
    return {
        source: {
            question: dict(sorted(bins.items()))
            for question, bins in sorted(questions.items())
        }
        for source, questions in sorted(counts.items())
    }


def load_losses(path: Path | None, manifest):
    if path is None:
        return {
            "status": "not_provided",
            "required_fields": ["qa_id", "qa_loss"],
            "aggregates": None,
        }
    values = defaultdict(list)
    seen = set()
    for row in read_jsonl(path):
        qa_id = row["qa_id"]
        if qa_id in seen or qa_id not in manifest:
            raise ContractError(f"Invalid QA-loss row: {qa_id}")
        seen.add(qa_id)
        value = float(row["qa_loss"])
        if not math.isfinite(value):
            raise ContractError(f"Non-finite qa_loss: {qa_id}")
        qa = manifest[qa_id]
        key = (
            qa["source_dataset"],
            str(qa.get("question_type") or "__unknown__"),
            qa["coverage_bin"],
        )
        values[key].append(value)
    if seen != set(manifest):
        raise ContractError("QA-loss rows must exactly cover the manifest")
    return {
        "status": "provided",
        "required_fields": ["qa_id", "qa_loss"],
        "aggregates": [
            {
                "source_dataset": key[0],
                "question_type": key[1],
                "coverage_bin": key[2],
                "count": len(group),
                "mean_qa_loss": sum(group) / len(group),
            }
            for key, group in sorted(values.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-manifest", required=True, type=Path)
    parser.add_argument("--a1-manifest", required=True, type=Path)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--a0-loss-jsonl", type=Path)
    parser.add_argument("--a1-loss-jsonl", type=Path)
    args = parser.parse_args()

    scenes = list(read_jsonl(args.scenes))
    frames = list(read_jsonl(args.frames))
    a0_ordered, a0 = load_manifest(args.a0_manifest)
    a1_ordered, a1 = load_manifest(args.a1_manifest)
    validate_records(scenes, frames, a0_ordered)
    validate_records(scenes, frames, a1_ordered)
    if set(a0) != set(a1):
        raise ContractError("A0/A1 qa_id sets differ")
    if [row["qa_id"] for row in a0_ordered] != [
        row["qa_id"] for row in a1_ordered
    ]:
        raise ContractError("A0/A1 ordered qa_id sequence differs")
    for left, right in zip(a0_ordered, a1_ordered):
        if causal_payload(left) != causal_payload(right):
            raise ContractError(
                f"A0/A1 causal QA payload differs: {left['qa_id']}"
            )
    a0_causal_hash = content_sha256(
        [causal_payload(row) for row in a0_ordered]
    )
    a1_causal_hash = content_sha256(
        [causal_payload(row) for row in a1_ordered]
    )

    report = {
        "schema_version": "parta_matched_arms_report_v1",
        "status": "pass",
        "qa_count": len(a0),
        "arm_specific_runtime_fields_excluded": sorted(
            ARM_SPECIFIC_RUNTIME_FIELDS
        ),
        "artifacts": {
            "a0": {
                "file_sha256": file_sha256(args.a0_manifest),
                "ordered_causal_payload_sha256": a0_causal_hash,
            },
            "a1": {
                "file_sha256": file_sha256(args.a1_manifest),
                "ordered_causal_payload_sha256": a1_causal_hash,
            },
        },
        "coverage_counts": coverage_counts(a0.values()),
        "qa_loss": {
            "a0": load_losses(args.a0_loss_jsonl, a0),
            "a1": load_losses(args.a1_loss_jsonl, a1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
