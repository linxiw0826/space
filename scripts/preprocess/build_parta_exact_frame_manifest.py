#!/usr/bin/env python3
"""Bind canonical QA to deterministic GUIDE 16-32 exact candidate frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.parta_data_contract import (  # noqa: E402
    ContractError,
    GUIDE_EXACT_SAMPLING_POLICY,
    GUIDE_WHOLE_MP4_SAMPLING_POLICY,
    build_manifest_rows,
    guide_frame_indices,
    read_jsonl,
    validate_guide_sampling_binding,
    validate_records,
    write_jsonl,
)
from src.adt_gt_supported_clip import (  # noqa: E402
    GT_SUPPORTED_CLIP_POLICY,
    select_maximal_run,
    validate_support_certificate as validate_adt_support_certificate,
)
from src.scannetppv2_support import (  # noqa: E402
    validate_support_certificate as validate_scannetppv2_support_certificate,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    video_sources = {
        row.get("source_dataset")
        for row in qa_rows
        if row.get("media_kind") == "video"
    }
    has_certified_video = bool(
        video_sources & {"adt", "scannetppv2"}
    )
    certificates = {}
    if has_certified_video:
        canonical_report_path = args.scenes.parent / "validation_report.json"
        if not canonical_report_path.is_file():
            raise ContractError(
                "Exact video build requires canonical sibling validation_report.json"
            )
        canonical_report = json.loads(canonical_report_path.read_text())
        registry = canonical_report.get(
            "trusted_support_certificate_registry"
        )
        if not isinstance(registry, dict):
            raise ContractError(
                "Canonical report lacks trusted support-certificate anchor"
            )
        certificate_path = args.scenes.parent / registry["path"]
        actual_digest = file_sha256(certificate_path)
        if actual_digest != registry["sha256"]:
            raise ContractError(
                "Canonical support-certificate anchor SHA256 mismatch"
            )
        for certificate in read_jsonl(certificate_path):
            key = (certificate["scene_id"], certificate["vsi_media"])
            if key in certificates:
                raise ContractError(f"Duplicate external support certificate: {key}")
            if certificate.get("schema_version", "").startswith("adt_"):
                validate_adt_support_certificate(certificate)
            else:
                validate_scannetppv2_support_certificate(certificate)
            certificates[key] = certificate
    frame_lookup = {
        (row["source_dataset"], row["frame_key"]): row for row in frames
    }
    metadata = {}
    if args.video_metadata:
        text = args.video_metadata.read_text(encoding="utf-8")
        payload = json.loads(text) if text.lstrip().startswith("{") else None
        rows = (
            payload["videos"]
            if isinstance(payload, dict) and isinstance(payload.get("videos"), list)
            else read_jsonl(args.video_metadata)
        )
        for row in rows:
            media = row.get("vsi_media", row.get("media"))
            source = row.get("source_dataset", str(media).split("/", 1)[0])
            scene = row.get("scene_id", Path(str(media)).stem)
            total = row.get("total_frames", row.get("frame_count"))
            fps_value = row.get("fps", row.get("avg_fps"))
            metadata[(source, scene, media)] = (int(total), float(fps_value))

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
        certificate_key = (qa["scene_id"], qa["vsi_media"])
        external_certificate = certificates.get(certificate_key)
        if qa["source_dataset"] == "scannetppv2":
            if external_certificate is None:
                raise ContractError(
                    f"Missing ScanNet++ support certificate: {certificate_key}"
                )
            validate_scannetppv2_support_certificate(external_certificate)
            guide_indices = guide_frame_indices(
                total_frames,
                fps,
                base_interval=args.base_interval,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
            )
            if qa["sampling_policy"] != GUIDE_WHOLE_MP4_SAMPLING_POLICY:
                raise ContractError("ScanNet++ exact build requires whole-MP4 GUIDE")
            if int(qa["video_total_frames"]) != total_frames:
                raise ContractError("ScanNet++ video total-frame provenance mismatch")
            if float(qa["video_fps"]).hex() != float(fps).hex():
                raise ContractError("ScanNet++ video FPS provenance mismatch")
            parameter_pairs = (
                ("base_interval", float(qa["sampling_base_interval"]), float(args.base_interval)),
                ("min_frames", int(qa["sampling_min_frames"]), int(args.min_frames)),
                ("max_frames", int(qa["sampling_max_frames"]), int(args.max_frames)),
            )
            for name, source_value, cli_value in parameter_pairs:
                equal = (
                    source_value.hex() == cli_value.hex()
                    if isinstance(source_value, float)
                    else source_value == cli_value
                )
                if not equal:
                    raise ContractError(f"ScanNet++ sampling {name} mismatch")
            if list(qa["actual_frame_indices"]) != guide_indices:
                raise ContractError("ScanNet++ exact frame indices differ from GUIDE")
            missing = [index for index in guide_indices if index not in candidate_frames]
            if missing:
                raise ContractError(f"ScanNet++ frame states missing GUIDE IDs: {missing}")
            selected = [candidate_frames[index] for index in guide_indices]
            if list(qa["actual_frame_keys"]) != [frame["frame_key"] for frame in selected]:
                raise ContractError("ScanNet++ exact frame keys are not GUIDE ordered")
            certified_frames = external_certificate["frames"]
            if [int(frame["frame_index"]) for frame in certified_frames] != guide_indices:
                raise ContractError("ScanNet++ certificate frame IDs differ from GUIDE")
            if external_certificate["sampling_binding_sha256"] != qa["source_sampling_binding_sha256"]:
                raise ContractError("ScanNet++ certificate sampling binding mismatch")
            rebound.append(dict(qa))
            continue
        provenance = qa["clip_provenance"]
        embedded_certificate = provenance["support_certificate"]
        if external_certificate != embedded_certificate:
            raise ContractError(
                f"External support certificate mismatch: {certificate_key}"
            )
        _, certified_runs = validate_adt_support_certificate(external_certificate)
        certified_clip = select_maximal_run(certified_runs)
        clip_start = int(provenance["clip_start_raw_frame"])
        clip_end = int(provenance["clip_end_raw_frame"])
        if certified_clip != (clip_start, clip_end):
            raise ContractError(
                f"D-59 clip is not externally certified maximal: {qa['qa_id']}"
            )
        local_indices = guide_frame_indices(
            clip_end - clip_start + 1,
            fps,
            base_interval=args.base_interval,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        guide_indices = [clip_start + index for index in local_indices]
        missing = [
            index for index in guide_indices if index not in candidate_frames
        ]
        if missing:
            raise ContractError(
                "Canonical frame states do not cover exact GUIDE raw frame "
                f"IDs for {qa['qa_id']}; missing={missing}"
            )
        source_policy = qa["sampling_policy"]
        if source_policy != GT_SUPPORTED_CLIP_POLICY:
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
        source_timestamps = [
            int(value)
            for value in provenance["selected_device_timestamps_ns"]
        ]
        frame_timestamps = [
            int(frame["device_timestamp_ns"]) for frame in selected
        ]
        if source_timestamps != frame_timestamps:
            raise ContractError(
                f"D-59 selected timestamp provenance mismatch: {qa['qa_id']}"
            )
        for frame in selected:
            trajectory_error = frame.get("trajectory_timestamp_error_ns")
            calibration_error = frame.get("calibration_timestamp_error_ns")
            if (
                trajectory_error is None
                or int(trajectory_error)
                > int(provenance["max_trajectory_error_ns"])
                or calibration_error is None
                or int(calibration_error)
                > int(provenance["max_calibration_error_ns"])
            ):
                raise ContractError(
                    "D-59 selected frame violates frozen temporal support: "
                    f"{frame['frame_key']}"
                )
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
        row["sampling_policy"] = GUIDE_EXACT_SAMPLING_POLICY
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
