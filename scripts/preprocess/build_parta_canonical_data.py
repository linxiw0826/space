#!/usr/bin/env python3
"""Adapt ADT v2 or Hypersim three-table outputs to canonical Part A JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (  # noqa: E402
    adapt_frame,
    adapt_qa,
    adapt_scene,
    build_manifest_rows,
    read_jsonl,
    source_visibility_contract,
    validate_records,
    write_jsonl,
)


FILES = {
    "adt": ("adt_scene_states.jsonl", "adt_frame_states.jsonl", "adt_qa_train.jsonl"),
    "hypersim": (
        "hypersim_scene_states.jsonl",
        "hypersim_frame_states.jsonl",
        "hypersim_qa_train.jsonl",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=FILES, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-t0-fixtures", action="store_true")
    parser.add_argument("--expected-source", action="append", default=[])
    args = parser.parse_args()

    scene_name, frame_name, qa_name = FILES[args.source]
    scenes = [adapt_scene(args.source, row) for row in read_jsonl(args.input_dir / scene_name)]
    raw_frames = list(read_jsonl(args.input_dir / frame_name))
    expected_visible_observations = sum(
        source_visibility_contract(args.source, observation)[1]
        for row in raw_frames
        for observation in row.get("visible_nodes", ())
    )
    frames = [adapt_frame(args.source, row) for row in raw_frames]
    raw_qa = list(read_jsonl(args.input_dir / qa_name))
    # Hypersim source QA does not repeat frame_id; recover it from frame_key.
    frame_by_media = {row.get("vsi_media"): row for row in frames}
    qa_rows = []
    for raw in raw_qa:
        if args.source == "hypersim":
            frame = frame_by_media.get(raw.get("vsi_media"))
            if frame is None:
                raise ValueError(f"No Hypersim frame for media {raw.get('vsi_media')}")
            raw = {**raw, "frame_index": frame["frame_index"]}
        qa_rows.append(adapt_qa(args.source, raw))
    frame_lookup = {(row["source_dataset"], row["frame_key"]): row for row in frames}
    manifest = list(build_manifest_rows(qa_rows, frame_lookup))
    report = validate_records(
        scenes,
        frames,
        manifest,
        require_fixtures=args.require_t0_fixtures,
        expected_sources=args.expected_source or None,
        expected_visible_observations={
            args.source: expected_visible_observations
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scene_states.jsonl", scenes)
    write_jsonl(args.output_dir / "frame_states.jsonl", frames)
    write_jsonl(args.output_dir / "qa_manifest.jsonl", manifest)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
