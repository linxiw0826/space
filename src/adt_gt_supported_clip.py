"""Frozen D-59 temporal-support and clip-local GUIDE sampling contract."""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
from typing import Any, Sequence

from src.parta_data_contract import ContractError, guide_frame_indices


GT_SUPPORTED_CLIP_POLICY = "guide_exact_over_gt_supported_clip_v1"
WHOLE_MP4_POLICY = "guide_exact_raw_mp4_v1"
LEGACY_POLICY = "legacy_valid_frame_linspace_v1"
TIE_POLICY = "longest_run_then_earliest_start_v1"
HARD_SUPPORT_FIELDS = ("trajectory", "calibration")
SUPPORT_CERTIFICATE_SCHEMA = "adt_temporal_support_certificate_v1"


def nearest_index(sorted_values: Sequence[int], query: int) -> int:
    if not sorted_values:
        raise ContractError("Cannot search an empty timestamp sequence")
    index = bisect_left(sorted_values, query)
    candidates = []
    if index < len(sorted_values):
        candidates.append(index)
    if index:
        candidates.append(index - 1)
    return min(candidates, key=lambda i: abs(sorted_values[i] - query))


def temporal_support(
    frame_timestamps: Sequence[int],
    trajectory_timestamps: Sequence[int],
    calibration_timestamps: Sequence[int],
    *,
    max_trajectory_error_ns: int,
    max_calibration_error_ns: int,
) -> tuple[list[bool], list[dict[str, Any]]]:
    if not trajectory_timestamps:
        raise ContractError("Trajectory contains no timestamps")
    if not calibration_timestamps:
        raise ContractError("Calibration contains no timestamps")
    valid: list[bool] = []
    diagnostics: list[dict[str, Any]] = []
    for frame_index, timestamp in enumerate(frame_timestamps):
        trajectory_error = None
        reasons = []
        if not trajectory_timestamps[0] <= timestamp <= trajectory_timestamps[-1]:
            reasons.append("outside_trajectory_span")
        else:
            trajectory_index = nearest_index(trajectory_timestamps, timestamp)
            trajectory_error = abs(
                trajectory_timestamps[trajectory_index] - timestamp
            )
            if trajectory_error > max_trajectory_error_ns:
                reasons.append("trajectory_time_gap")
        calibration_index = nearest_index(calibration_timestamps, timestamp)
        calibration_error = abs(
            calibration_timestamps[calibration_index] - timestamp
        )
        if calibration_error > max_calibration_error_ns:
            reasons.append("calibration_time_gap")
        valid.append(not reasons)
        diagnostics.append({
            "frame_index": frame_index,
            "device_timestamp_ns": int(timestamp),
            "trajectory_timestamp_error_ns": trajectory_error,
            "calibration_timestamp_error_ns": calibration_error,
            "valid": not reasons,
            "reasons": reasons,
        })
    return valid, diagnostics


def contiguous_true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate([*mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    return runs


def encode_support_mask(mask: Sequence[bool]) -> str:
    value = bytearray((len(mask) + 7) // 8)
    for index, supported in enumerate(mask):
        if supported:
            value[index // 8] |= 1 << (index % 8)
    return bytes(value).hex()


def decode_support_mask(encoded: str, total_frames: int) -> list[bool]:
    try:
        value = bytes.fromhex(encoded)
    except ValueError as error:
        raise ContractError("Invalid support-certificate bitset hex") from error
    expected_bytes = (total_frames + 7) // 8
    if len(value) != expected_bytes:
        raise ContractError("Support-certificate bitset length mismatch")
    if total_frames % 8:
        unused_mask = ~((1 << (total_frames % 8)) - 1) & 0xFF
        if value[-1] & unused_mask:
            raise ContractError("Support-certificate has nonzero padding bits")
    return [
        bool(value[index // 8] & (1 << (index % 8)))
        for index in range(total_frames)
    ]


def certificate_payload(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: certificate[key]
        for key in (
            "schema_version",
            "scene_id",
            "vsi_media",
            "whole_video_total_frames",
            "whole_video_fps_hex",
            "whole_video_start_device_timestamp_ns",
            "whole_video_end_device_timestamp_ns",
            "frame_timestamps_sha256",
            "hard_support_fields",
            "max_trajectory_error_ns",
            "max_calibration_error_ns",
            "support_mask_bitset_hex",
        )
    }


def support_certificate_sha256(certificate: dict[str, Any]) -> str:
    serialized = json.dumps(
        certificate_payload(certificate),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def validate_support_certificate(
    certificate: dict[str, Any],
) -> tuple[list[bool], list[tuple[int, int]]]:
    if certificate.get("schema_version") != SUPPORT_CERTIFICATE_SCHEMA:
        raise ContractError("Unsupported support-certificate schema")
    total_frames = int(certificate["whole_video_total_frames"])
    if total_frames <= 0:
        raise ContractError("Invalid support-certificate frame count")
    if certificate["hard_support_fields"] != list(HARD_SUPPORT_FIELDS):
        raise ContractError("Invalid support-certificate capabilities")
    if (
        int(certificate["max_trajectory_error_ns"]) != 5_000_000
        or int(certificate["max_calibration_error_ns"]) != 50_000_000
    ):
        raise ContractError("Support-certificate thresholds are not frozen")
    actual = certificate.get("certificate_sha256")
    expected = support_certificate_sha256(certificate)
    if actual != expected:
        raise ContractError("Invalid support-certificate content SHA256")
    mask = decode_support_mask(
        certificate["support_mask_bitset_hex"], total_frames
    )
    runs = contiguous_true_runs(mask)
    if not runs:
        raise ContractError("Support-certificate has no supported run")
    return mask, runs


def build_support_certificate(
    *,
    scene_id: str,
    vsi_media: str,
    frame_timestamps: Sequence[int],
    fps: float,
    support_mask: Sequence[bool],
    max_trajectory_error_ns: int,
    max_calibration_error_ns: int,
) -> dict[str, Any]:
    timestamp_bytes = json.dumps(
        [int(value) for value in frame_timestamps],
        separators=(",", ":"),
    ).encode()
    certificate = {
        "schema_version": SUPPORT_CERTIFICATE_SCHEMA,
        "scene_id": scene_id,
        "vsi_media": vsi_media,
        "whole_video_total_frames": len(frame_timestamps),
        "whole_video_fps_hex": float(fps).hex(),
        "whole_video_start_device_timestamp_ns": int(frame_timestamps[0]),
        "whole_video_end_device_timestamp_ns": int(frame_timestamps[-1]),
        "frame_timestamps_sha256": hashlib.sha256(timestamp_bytes).hexdigest(),
        "hard_support_fields": list(HARD_SUPPORT_FIELDS),
        "max_trajectory_error_ns": int(max_trajectory_error_ns),
        "max_calibration_error_ns": int(max_calibration_error_ns),
        "support_mask_bitset_hex": encode_support_mask(support_mask),
    }
    certificate["certificate_sha256"] = support_certificate_sha256(certificate)
    return certificate


def select_maximal_run(runs: Sequence[tuple[int, int]]) -> tuple[int, int]:
    if not runs:
        raise ContractError("No GT-supported raw-frame run")
    return min(runs, key=lambda run: (-(run[1] - run[0] + 1), run[0]))


def sample_supported_run(
    run: tuple[int, int],
    fps: float,
    *,
    base_interval: float,
    min_frames: int,
    max_frames: int,
) -> tuple[list[int], list[int]]:
    """Return clip-local GUIDE positions and their raw-MP4 mapping."""
    start, end = run
    local_indices = guide_frame_indices(
        end - start + 1,
        fps,
        base_interval=base_interval,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    return local_indices, [start + index for index in local_indices]


def supported_clip_sampling(
    frame_timestamps: Sequence[int],
    trajectory_timestamps: Sequence[int],
    calibration_timestamps: Sequence[int],
    fps: float,
    *,
    base_interval: float,
    min_frames: int,
    max_frames: int,
    max_trajectory_error_ns: int,
    max_calibration_error_ns: int,
) -> dict[str, Any]:
    """Compute the unique D-59 supported clip and mapped raw MP4 IDs."""
    mask, diagnostics = temporal_support(
        frame_timestamps,
        trajectory_timestamps,
        calibration_timestamps,
        max_trajectory_error_ns=max_trajectory_error_ns,
        max_calibration_error_ns=max_calibration_error_ns,
    )
    runs = contiguous_true_runs(mask)
    start, end = select_maximal_run(runs)
    local_indices, raw_indices = sample_supported_run(
        (start, end),
        fps,
        base_interval=base_interval,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    if any(not diagnostics[index]["valid"] for index in raw_indices):
        raise ContractError("Selected frame lost frozen temporal support")
    return {
        "hard_support_fields": list(HARD_SUPPORT_FIELDS),
        "tie_policy": TIE_POLICY,
        "support_runs": [
            {
                "start_raw_frame": run_start,
                "end_raw_frame": run_end,
                "frame_count": run_end - run_start + 1,
                "start_device_timestamp_ns": int(frame_timestamps[run_start]),
                "end_device_timestamp_ns": int(frame_timestamps[run_end]),
            }
            for run_start, run_end in runs
        ],
        "clip_start_raw_frame": start,
        "clip_end_raw_frame": end,
        "clip_frame_count": end - start + 1,
        "clip_start_device_timestamp_ns": int(frame_timestamps[start]),
        "clip_end_device_timestamp_ns": int(frame_timestamps[end]),
        "local_frame_indices": local_indices,
        "raw_frame_indices": raw_indices,
        "selected_device_timestamps_ns": [
            int(frame_timestamps[index]) for index in raw_indices
        ],
        "selected_diagnostics": [diagnostics[index] for index in raw_indices],
    }
