#!/usr/bin/env python3
"""Bind canonical QA to deterministic GUIDE 16-32 exact candidate frames."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (  # noqa: E402
    ContractError,
    GUIDE_EXACT_SAMPLING_POLICY,
    build_manifest_rows,
    guide_frame_indices,
    read_jsonl,
    validate_guide_sampling_binding,
    validate_records,
    write_jsonl,
)


def video_metadata(video_path: Path) -> tuple[int, float]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,avg_frame_rate",
            "-of", "json", str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    return int(stream["nb_frames"]), numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument(
        "--video-metadata",
        type=Path,
        help="Optional JSONL keyed by source_dataset, scene_id, vsi_media.",
    )
    parser.add_argument("--base-interval", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--require-t0-fixtures", action="store_true")
    parser.add_argument("--expected-source", action="append", default=[])
    args = parser.parse_args()

    scenes = list(read_jsonl(args.scenes))
    frames = list(read_jsonl(args.frames))
    qa_rows = list(read_jsonl(args.qa))
    frame_lookup = {
        (row["source_dataset"], row["frame_key"]): row for row in frames
    }
    metadata = {}
    if args.video_metadata:
        for row in read_jsonl(args.video_metadata):
            metadata[
                (row["source_dataset"], row["scene_id"], row["vsi_media"])
            ] = (int(row["total_frames"]), float(row["fps"]))

    rebound = []
    for qa in qa_rows:
        if qa["media_kind"] == "image":
            rebound.append(dict(qa))
            continue
        validate_guide_sampling_binding(qa)
        candidate_frames = {
            frame_lookup[(qa["source_dataset"], key)]["frame_index"]:
            frame_lookup[(qa["source_dataset"], key)]
            for key in qa["actual_frame_keys"]
        }
        meta_key = (
            qa["source_dataset"], qa["scene_id"], qa["vsi_media"]
        )
        if meta_key in metadata:
            total_frames, fps = metadata[meta_key]
        elif args.video_root is not None:
            total_frames, fps = video_metadata(args.video_root / qa["vsi_media"])
        else:
            raise ContractError(
                f"Missing video metadata for {meta_key}; provide "
                "--video-metadata or --video-root"
            )
        guide_indices = guide_frame_indices(
            total_frames,
            fps,
            base_interval=args.base_interval,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        missing = [
            index for index in guide_indices if index not in candidate_frames
        ]
        if missing:
            raise ContractError(
                "Canonical frame states do not cover exact GUIDE raw frame "
                f"IDs for {qa['qa_id']}; missing={missing}"
            )
        source_policy = qa["sampling_policy"]
        if source_policy != GUIDE_EXACT_SAMPLING_POLICY:
            raise ContractError("Unreachable non-GUIDE sampling policy")
        declared_total = qa.get("video_total_frames")
        declared_fps = qa.get("video_fps")
        if int(declared_total) != total_frames:
            raise ContractError(
                f"Video total-frame provenance mismatch for {qa['qa_id']}"
            )
        if float(declared_fps).hex() != float(fps).hex():
            raise ContractError(
                f"Video FPS provenance mismatch for {qa['qa_id']}"
            )
        parameter_pairs = (
            ("base_interval", float(qa["sampling_base_interval"]),
             float(args.base_interval)),
            ("min_frames", int(qa["sampling_min_frames"]),
             int(args.min_frames)),
            ("max_frames", int(qa["sampling_max_frames"]),
             int(args.max_frames)),
        )
        for name, source_value, cli_value in parameter_pairs:
            equal = (
                source_value.hex() == cli_value.hex()
                if isinstance(source_value, float)
                else source_value == cli_value
            )
            if not equal:
                raise ContractError(
                    f"Sampling {name} mismatch for {qa['qa_id']}: "
                    f"source={source_value}, cli={cli_value}"
                )
        if list(qa["actual_frame_indices"]) != guide_indices:
            raise ContractError(
                "Source exact raw frame IDs differ from recomputed GUIDE IDs "
                f"for {qa['qa_id']}: source={qa['actual_frame_indices']}, "
                f"guide={guide_indices}"
            )
        selected = [candidate_frames[index] for index in guide_indices]
        if list(qa["actual_frame_keys"]) != [
            frame["frame_key"] for frame in selected
        ]:
            raise ContractError(
                f"Source exact frame keys are not GUIDE ordered: {qa['qa_id']}"
            )
        row = dict(qa)
        row["actual_frame_keys"] = [frame["frame_key"] for frame in selected]
        row["actual_frame_indices"] = [
            frame["frame_index"] for frame in selected
        ]
        row["sampling_policy"] = "guide_exact_raw_mp4_v1"
        row["video_total_frames"] = total_frames
        row["video_fps"] = fps
        rebound.append(row)

    manifest = list(build_manifest_rows(rebound, frame_lookup))
    report = validate_records(
        scenes,
        frames,
        manifest,
        require_fixtures=args.require_t0_fixtures,
        expected_sources=args.expected_source or None,
    )
    write_jsonl(args.output, manifest)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
