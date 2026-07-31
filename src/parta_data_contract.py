"""Versioned, source-neutral data contract for Part A A1-O.

The contract deliberately contains only JSON-compatible values.  Missing
supervision is represented by ``None`` plus an explicit false field mask; a
numeric zero is always a real measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "parta_canonical_v1"
QA_EVIDENCE_SCOPES = {
    "frame_verified",
    "scene_associated_unlocalized",
}
GUIDE_EXACT_SAMPLING_POLICY = "guide_exact_over_gt_supported_clip_v1"
GUIDE_WHOLE_MP4_SAMPLING_POLICY = "guide_exact_raw_mp4_v1"
T0_FIXTURES = {
    "adt": (
        "Apartment_release_clean_seq131_M1292",
        "Apartment_release_clean_seq133_M1292",
        "Apartment_release_clean_seq134_M1292",
    ),
    "hypersim": ("ai_001_001", "ai_001_002"),
}
STATE_FIELDS = ("category", "center", "extent", "visibility")
CANONICAL_COORDINATE_CONTRACT = {
    "name": "parta_right_handed_xright_yup_zback_m_v1",
    "handedness": "right",
    "axes": {"x": "right", "y": "up", "z": "back"},
    "camera_forward_axis": "-z",
    "units": "meters",
    "transform_semantics": "R_canonical_world_from_local",
}
SOURCE_CONTRACTS = {
    "adt": {
        "source_schema_prefixes": ("adt_scene_state_v1", "adt_frame_state_v1"),
        "source_axes": {"x": "right", "y": "forward", "z": "up"},
        "source_handedness": "right",
        "source_units": "meters",
        # (x, y, z)_ADT -> (x, z, -y)_canonical.
        "rotation_canonical_from_source_world": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
    },
    "hypersim": {
        "source_schema_prefixes": (
            "hypersim_scene_state_v1",
            "hypersim_frame_state_v1",
        ),
        "source_axes": {"x": "right", "y": "up", "z": "back"},
        "source_handedness": "right",
        "source_units": "meters",
        "rotation_canonical_from_source_world": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    },
}
CANONICAL_CATEGORIES = (
    "__unknown__",
    "backpack", "bathtub", "bed", "bench", "bicycle", "bin", "blanket",
    "blind", "book", "bookshelf", "bottle", "bowl", "cabinet", "camera",
    "car", "ceiling", "chair", "clock", "clothes", "computer", "counter",
    "curtain", "desk", "dishwasher", "door", "dresser", "floor", "fridge",
    "lamp", "laptop", "microwave", "mirror", "monitor", "ottoman", "oven",
    "person", "picture", "pillow", "plant", "printer", "refrigerator",
    "shelf", "sink", "sofa", "stool", "table", "television", "toilet",
    "towel", "wall", "wardrobe", "window",
)
CATEGORY_ALIASES = {
    "couch": "sofa",
    "tv": "television",
    "tv monitor": "television",
    "trash can": "bin",
    "garbage bin": "bin",
    "refridgerator": "refrigerator",
}


class ContractError(ValueError):
    """Raised when a record violates a fail-closed Part A invariant."""


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ContractError(f"{path}:{line_number}: {error}") from error


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
            count += 1
    return count


def guide_frame_indices(
    total_frames: int,
    fps: float,
    *,
    base_interval: float = 1.0,
    min_frames: int = 16,
    max_frames: int = 32,
) -> list[int]:
    """Reproduce GUIDE's deterministic dynamic linspace sampling."""
    if total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ContractError(
            f"Invalid video metadata total_frames={total_frames}, fps={fps}"
        )
    if not (1 <= min_frames <= max_frames):
        raise ContractError("Expected 1 <= min_frames <= max_frames")
    target = min(
        max(round((total_frames / fps) / base_interval), min_frames),
        max_frames,
    )
    indices = np.unique(
        np.linspace(0, total_frames - 1, target, dtype=np.int64)
    ).tolist()
    if not min_frames <= len(indices) <= max_frames:
        raise ContractError(
            "Video is too short to provide the required exact frame count: "
            f"total={total_frames}, sampled={len(indices)}"
        )
    return [int(index) for index in indices]


def guide_sampling_payload(
    *,
    source_dataset: str,
    scene_id: str,
    vsi_media: str,
    frame_keys: Sequence[str],
    frame_indices: Sequence[int],
    total_frames: int,
    fps: float,
    base_interval: float,
    min_frames: int,
    max_frames: int,
    sampling_policy: str,
    clip_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the unique stable payload for exact GUIDE frame binding.

    Floats use their exact IEEE-754 hexadecimal representation so every
    producer and consumer hashes one unambiguous value.
    """
    fps = float(fps)
    base_interval = float(base_interval)
    if not math.isfinite(fps) or not math.isfinite(base_interval):
        raise ContractError("Sampling FPS/base interval must be finite")
    payload = {
        "source_dataset": str(source_dataset),
        "scene_id": str(scene_id),
        "vsi_media": str(vsi_media),
        "sampling_policy": str(sampling_policy),
        "frame_keys": [str(value) for value in frame_keys],
        "frame_indices": [int(value) for value in frame_indices],
        "total_frames": int(total_frames),
        "fps_hex": fps.hex(),
        "base_interval_hex": base_interval.hex(),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
    }
    if clip_provenance is not None:
        payload["clip_provenance"] = dict(clip_provenance)
    return payload


def guide_sampling_binding_sha256(**kwargs: Any) -> str:
    return content_sha256(guide_sampling_payload(**kwargs))


def frame_binding_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": row["source_dataset"],
        "scene_id": row["scene_id"],
        "vsi_media": row["vsi_media"],
        "frame_keys": list(row["actual_frame_keys"]),
        "frame_indices": [int(value) for value in row["actual_frame_indices"]],
    }


def frame_binding_sha256(row: Mapping[str, Any]) -> str:
    return content_sha256(frame_binding_payload(row))


def duration_coverage_ratio(clip_provenance: Mapping[str, Any]) -> float:
    whole = (
        int(clip_provenance["whole_video_end_device_timestamp_ns"])
        - int(clip_provenance["whole_video_start_device_timestamp_ns"])
    )
    clip = (
        int(clip_provenance["clip_end_device_timestamp_ns"])
        - int(clip_provenance["clip_start_device_timestamp_ns"])
    )
    if whole < 0 or clip < 0 or clip > whole:
        raise ContractError("Invalid duration coverage timestamps")
    return 1.0 if whole == 0 and clip == 0 else clip / whole


def coverage_bin(ratio: float) -> str:
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ContractError(f"Invalid duration coverage ratio: {ratio}")
    if ratio >= 0.75:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def validate_qa_evidence_contract(row: Mapping[str, Any]) -> None:
    scope = row.get("qa_evidence_scope")
    verified = row.get("qa_visual_support_verified")
    evidence = row.get("evidence_frame_indices")
    if scope not in QA_EVIDENCE_SCOPES:
        raise ContractError(f"Invalid qa_evidence_scope: {scope!r}")
    if not isinstance(verified, bool):
        raise ContractError("qa_visual_support_verified must be boolean")
    if row.get("source_dataset") == "adt" and (
        scope != "scene_associated_unlocalized"
        or verified
        or evidence is not None
    ):
        raise ContractError(
            "ADT QA must be scene_associated_unlocalized/false/null"
        )
    if scope == "scene_associated_unlocalized":
        if verified or evidence is not None:
            raise ContractError(
                "Unlocalized QA must have verified=false and evidence=null"
            )
        return
    if not verified or not isinstance(evidence, list) or not evidence:
        raise ContractError(
            "frame_verified QA requires verified=true and nonempty evidence"
        )
    actual = {int(value) for value in row["actual_frame_indices"]}
    normalized = [int(value) for value in evidence]
    if len(normalized) != len(set(normalized)) or not set(normalized) <= actual:
        raise ContractError(
            "Evidence frame indices must be a unique actual-frame subset"
        )


def validate_guide_sampling_binding(row: Mapping[str, Any]) -> None:
    """Fail closed unless a QA row carries a valid exact GUIDE binding."""
    policy = row.get("sampling_policy")
    if policy != GUIDE_EXACT_SAMPLING_POLICY:
        raise ContractError(
            "ADT sampling_policy must be exactly "
            f"{GUIDE_EXACT_SAMPLING_POLICY}, got {policy!r}"
        )
    field_map = {
        "frame_keys": row.get(
            "actual_frame_keys", row.get("candidate_frame_keys")
        ),
        "frame_indices": row.get(
            "actual_frame_indices", row.get("candidate_frame_indices")
        ),
        "total_frames": row.get(
            "video_total_frames", row.get("total_frames")
        ),
        "fps": row.get("video_fps", row.get("fps")),
        "base_interval": row.get(
            "sampling_base_interval", row.get("base_interval")
        ),
        "min_frames": row.get(
            "sampling_min_frames", row.get("min_frames")
        ),
        "max_frames": row.get(
            "sampling_max_frames", row.get("max_frames")
        ),
    }
    clip_provenance = row.get("clip_provenance")
    missing = [key for key, value in field_map.items() if value is None]
    if missing:
        raise ContractError(f"Missing GUIDE sampling fields: {missing}")
    expected = guide_sampling_binding_sha256(
        source_dataset=row.get("source_dataset", "adt"),
        scene_id=row["scene_id"],
        vsi_media=row["vsi_media"],
        sampling_policy=policy,
        clip_provenance=clip_provenance,
        **field_map,
    )
    actual = row.get(
        "source_sampling_binding_sha256",
        row.get("sampling_binding_sha256"),
    )
    if actual != expected:
        raise ContractError(
            "Invalid GUIDE source sampling binding SHA256: "
            f"expected={expected}, actual={actual}"
        )
    if not isinstance(clip_provenance, Mapping):
        raise ContractError("Missing D-59 clip_provenance")
    required_clip = {
        "whole_video_total_frames",
        "whole_video_fps_hex",
        "whole_video_start_device_timestamp_ns",
        "whole_video_end_device_timestamp_ns",
        "clip_start_raw_frame",
        "clip_end_raw_frame",
        "clip_start_device_timestamp_ns",
        "clip_end_device_timestamp_ns",
        "clip_frame_count",
        "support_runs",
        "tie_policy",
        "hard_support_fields",
        "max_trajectory_error_ns",
        "max_calibration_error_ns",
        "local_frame_indices",
        "selected_device_timestamps_ns",
        "support_certificate",
    }
    missing_clip = required_clip - set(clip_provenance)
    if missing_clip:
        raise ContractError(
            f"Missing D-59 clip provenance fields: {sorted(missing_clip)}"
        )
    total_frames = int(field_map["total_frames"])
    if int(clip_provenance["whole_video_total_frames"]) != total_frames:
        raise ContractError("D-59 whole-video total-frame mismatch")
    if clip_provenance["whole_video_fps_hex"] != float(field_map["fps"]).hex():
        raise ContractError("D-59 whole-video FPS mismatch")
    start = int(clip_provenance["clip_start_raw_frame"])
    end = int(clip_provenance["clip_end_raw_frame"])
    if not 0 <= start <= end < total_frames:
        raise ContractError("Invalid D-59 clip bounds")
    if int(clip_provenance["clip_frame_count"]) != end - start + 1:
        raise ContractError("Invalid D-59 clip frame count")
    if __name__.startswith("src."):
        from src.adt_gt_supported_clip import (
            select_maximal_run,
            validate_support_certificate,
        )
    else:
        from adt_gt_supported_clip import (  # type: ignore[no-redef]
            select_maximal_run,
            validate_support_certificate,
        )
    certificate = clip_provenance["support_certificate"]
    if not isinstance(certificate, dict):
        raise ContractError("Invalid D-59 support certificate")
    _, certified_runs = validate_support_certificate(certificate)
    for key in (
        "scene_id",
        "vsi_media",
        "whole_video_total_frames",
        "whole_video_fps_hex",
        "whole_video_start_device_timestamp_ns",
        "whole_video_end_device_timestamp_ns",
        "hard_support_fields",
        "max_trajectory_error_ns",
        "max_calibration_error_ns",
    ):
        expected_value = (
            row[key] if key in ("scene_id", "vsi_media")
            else clip_provenance[key]
        )
        if certificate[key] != expected_value:
            raise ContractError(
                f"D-59 support certificate/provenance mismatch: {key}"
            )
    recomputed_local = guide_frame_indices(
        end - start + 1,
        float(field_map["fps"]),
        base_interval=float(field_map["base_interval"]),
        min_frames=int(field_map["min_frames"]),
        max_frames=int(field_map["max_frames"]),
    )
    recomputed_indices = [start + index for index in recomputed_local]
    if list(clip_provenance["local_frame_indices"]) != recomputed_local:
        raise ContractError("Invalid D-59 clip-local GUIDE positions")
    runs = clip_provenance["support_runs"]
    if not isinstance(runs, Sequence) or not runs:
        raise ContractError("Missing D-59 support runs")
    normalized_runs = [
        (int(run["start_raw_frame"]), int(run["end_raw_frame"]))
        for run in runs
    ]
    if normalized_runs != certified_runs:
        raise ContractError("D-59 declared runs differ from support certificate")
    previous_end = -1
    for run, (run_start, run_end) in zip(runs, normalized_runs):
        if not 0 <= run_start <= run_end < total_frames:
            raise ContractError("Invalid D-59 support-run bounds")
        if run_start <= previous_end:
            raise ContractError("D-59 support runs are not ordered/disjoint")
        if int(run["frame_count"]) != run_end - run_start + 1:
            raise ContractError("Invalid D-59 support-run frame count")
        previous_end = run_end
    chosen = select_maximal_run(certified_runs)
    if chosen != (start, end):
        raise ContractError("D-59 clip violates longest-earliest tie policy")
    chosen_run = next(
        run for run, bounds in zip(runs, normalized_runs)
        if bounds == chosen
    )
    if (
        int(chosen_run["start_device_timestamp_ns"])
        != int(clip_provenance["clip_start_device_timestamp_ns"])
        or int(chosen_run["end_device_timestamp_ns"])
        != int(clip_provenance["clip_end_device_timestamp_ns"])
    ):
        raise ContractError("D-59 clip timestamp/run mismatch")
    if clip_provenance["tie_policy"] != (
        "longest_run_then_earliest_start_v1"
    ):
        raise ContractError("Unsupported D-59 tie policy")
    if list(clip_provenance["hard_support_fields"]) != [
        "trajectory", "calibration"
    ]:
        raise ContractError("Unsupported D-59 hard-support capability set")
    if (
        int(clip_provenance["max_trajectory_error_ns"]) != 5_000_000
        or int(clip_provenance["max_calibration_error_ns"]) != 50_000_000
    ):
        raise ContractError("D-59 temporal thresholds differ from frozen values")
    selected_timestamps = clip_provenance["selected_device_timestamps_ns"]
    if len(selected_timestamps) != len(recomputed_indices):
        raise ContractError("D-59 selected timestamp/ID lengths differ")
    if list(field_map["frame_indices"]) != recomputed_indices:
        raise ContractError(
            "Source sampling binding contains non-GUIDE raw frame IDs: "
            f"source={list(field_map['frame_indices'])}, "
            f"guide={recomputed_indices}"
        )
    if len(field_map["frame_keys"]) != len(recomputed_indices):
        raise ContractError(
            "Source sampling frame key/index lengths differ"
        )


def _source_object_id(source: str, scene_id: str, object_id: Any) -> str:
    return f"{source}:{scene_id}:{object_id}"


def _mask(value: Any) -> bool:
    return value is not None


def _source_rotation(source: str) -> np.ndarray:
    if source not in SOURCE_CONTRACTS:
        raise ContractError(f"Unsupported source contract: {source}")
    return np.asarray(
        SOURCE_CONTRACTS[source]["rotation_canonical_from_source_world"],
        dtype=np.float64,
    )


def _vector_to_canonical(source: str, value: Any) -> Any:
    if value is None:
        return None
    return (_source_rotation(source) @ np.asarray(value, dtype=np.float64)).tolist()


def _extent_to_canonical(source: str, value: Any) -> Any:
    if value is None:
        return None
    return (
        np.abs(_source_rotation(source)) @ np.asarray(value, dtype=np.float64)
    ).tolist()


def _rotation_to_canonical(source: str, value: Any) -> Any:
    if value is None:
        return None
    return (
        _source_rotation(source) @ np.asarray(value, dtype=np.float64)
    ).tolist()


def canonical_category(value: Any) -> tuple[str, str | None]:
    if value is None:
        return "__unknown__", None
    raw = str(value).strip().lower().replace("_", " ")
    mapped = CATEGORY_ALIASES.get(raw, raw)
    if mapped not in CANONICAL_CATEGORIES:
        mapped = "__unknown__"
    return mapped, str(value)


def _node(
    source: str, scene_id: str, raw: Mapping[str, Any]
) -> dict[str, Any]:
    geometry_valid = raw.get("geometry_valid", True)
    if source == "adt" and not isinstance(geometry_valid, bool):
        raise ContractError("ADT scene-node geometry_valid must be boolean")
    if source == "adt" and not geometry_valid and any(
        raw.get(field) is not None
        for field in (
            "center_world_m",
            "extent_m",
            "rotation_world_from_object",
        )
    ):
        raise ContractError("ADT invalid scene-node geometry must be nulled")
    center = raw.get("center_world_m", raw.get("bbox_center_m"))
    extent = raw.get("extent_m", raw.get("bbox_extent_m"))
    rotation = raw.get(
        "rotation_world_from_object", raw.get("bbox_orientation_raw")
    )
    category, source_category = canonical_category(
        raw.get("category") or raw.get("object_name")
    )
    center = _vector_to_canonical(source, center)
    extent = _extent_to_canonical(source, extent)
    rotation = _rotation_to_canonical(source, rotation)
    masks = {
        "category": source_category is not None,
        "center": _mask(center),
        "extent": _mask(extent),
        "orientation": _mask(rotation),
        "motion": _mask(raw.get("velocity_world_mps")),
    }
    if source == "adt" and not geometry_valid:
        masks["center"] = False
        masks["extent"] = False
        masks["orientation"] = False
    return {
        "object_id": _source_object_id(source, scene_id, raw["object_id"]),
        "source_object_id": str(raw["object_id"]),
        "category": category,
        "source_category": source_category,
        "center_world_m": center,
        "extent_m": extent,
        "rotation_world_from_object": rotation,
        "motion_type": raw.get("motion_type"),
        "source_geometry_valid": geometry_valid,
        "reference_pose_timestamp_error_ns": raw.get(
            "reference_pose_timestamp_error_ns"
        ),
        "velocity_world_mps": _vector_to_canonical(
            source, raw.get("velocity_world_mps")
        ),
        "field_mask": masks,
    }


def adapt_scene(source: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    contract = SOURCE_CONTRACTS.get(source)
    if contract is None:
        raise ContractError(f"Unsupported source: {source}")
    if not str(raw.get("schema_version", "")).startswith(
        contract["source_schema_prefixes"][0]
    ):
        raise ContractError(
            f"Unexpected {source} scene schema: {raw.get('schema_version')}"
        )
    nodes = [
        _node(source, raw["scene_id"], node) for node in raw.get("nodes", ())
    ]
    capabilities = {
        "metric_scale": True,
        "camera_pose": True,
        "direct_instance_id": True,
        "oriented_3d_box": any(
            node["field_mask"]["orientation"] for node in nodes
        ),
        "per_view_visibility": True,
        "object_motion": source == "adt" and any(
            node.get("motion_type") not in (None, "static") for node in nodes
        ),
        "camera_velocity": source == "adt",
        "depth_verified": source == "hypersim",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "scene",
        "source_dataset": source,
        "scene_id": raw["scene_id"],
        "coordinate_frame": CANONICAL_COORDINATE_CONTRACT["name"],
        "metric_scale": 1.0,
        "coordinate_contract": CANONICAL_COORDINATE_CONTRACT,
        "source_adapter_contract": contract,
        "capabilities": capabilities,
        "nodes": nodes,
        "source_schema_version": raw.get("schema_version"),
    }


def source_visibility_contract(
    source: str, observation: Mapping[str, Any]
) -> tuple[bool, bool]:
    """Return ``(evidence_present, supervision_visible)`` for one observation.

    ADT ``visible_nodes`` contains direct GT instance-identity observations.
    Membership therefore means visible unless the source row explicitly marks
    the observation invisible.  Hypersim visibility supervision additionally
    requires valid geometry and at least 16 supporting pixels.
    """
    evidence_present = True
    explicit_visible = observation.get(
        "visible", observation.get("visibility", True)
    )
    if not isinstance(explicit_visible, bool):
        raise ContractError(
            f"{source} explicit visibility must be boolean, got "
            f"{type(explicit_visible).__name__}"
        )
    if source == "adt":
        return evidence_present, explicit_visible
    if source == "hypersim":
        geometry_valid = observation.get("geometry_valid", False)
        pixel_count = observation.get("pixel_count")
        if not isinstance(geometry_valid, bool):
            raise ContractError("Hypersim geometry_valid must be boolean")
        if not isinstance(pixel_count, (int, float)) or isinstance(
            pixel_count, bool
        ):
            raise ContractError(
                "Hypersim visibility evidence requires numeric pixel_count"
            )
        return (
            evidence_present,
            bool(explicit_visible and geometry_valid and pixel_count >= 16),
        )
    raise ContractError(f"Unsupported source: {source}")


def adapt_frame(source: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    contract = SOURCE_CONTRACTS.get(source)
    if contract is None or not str(raw.get("schema_version", "")).startswith(
        contract["source_schema_prefixes"][1]
    ):
        raise ContractError(
            f"Unexpected {source} frame schema: {raw.get('schema_version')}"
        )
    visible = []
    seen_object_ids: set[str] = set()
    for observation in raw.get("visible_nodes", ()):
        source_object_id = str(observation["object_id"])
        if source_object_id in seen_object_ids:
            raise ContractError(
                f"Duplicate visible_nodes object_id in {raw['frame_key']}: "
                f"{source_object_id}"
            )
        seen_object_ids.add(source_object_id)
        evidence_present, supervision_valid = source_visibility_contract(
            source, observation
        )
        source_geometry_valid = observation.get("object_geometry_valid")
        if source == "adt":
            if not isinstance(source_geometry_valid, bool):
                # Backward-compatible source rows lacked the frozen D-59
                # dynamic-pose mask; formal new rows must set it explicitly.
                source_geometry_valid = False
            if not source_geometry_valid and any(
                observation.get(field) is not None
                for field in (
                    "center_world_m",
                    "rotation_world_from_object",
                    "center_camera_m",
                    "camera_distance_m",
                )
            ):
                raise ContractError(
                    "ADT invalid object geometry must be nulled"
                )
        visible.append(
            {
                "object_id": _source_object_id(
                    source, raw["scene_id"], observation["object_id"]
                ),
                "evidence_present": evidence_present,
                "visible": supervision_valid,
                # A1-O v1 predicts global center and per-view visibility only.
                # Source camera coordinates have different axis conventions,
                # so this adapter deliberately does not expose them as GT.
                "center_camera_m": None,
                "camera_distance_m": None,
                "camera_coordinate_contract": None,
                "source_object_geometry_valid": source_geometry_valid,
                "source_object_pose_timestamp_error_ns": observation.get(
                    "object_pose_timestamp_error_ns"
                ),
                "field_mask": {
                    "visibility": supervision_valid,
                    "camera_geometry": False,
                },
            }
        )
    frame_index = raw.get("frame_index", raw.get("frame_id"))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "frame",
        "source_dataset": source,
        "scene_id": raw["scene_id"],
        "frame_key": raw["frame_key"],
        "frame_index": int(frame_index) if frame_index is not None else None,
        "device_timestamp_ns": raw.get("device_timestamp_ns"),
        "trajectory_timestamp_error_ns": raw.get(
            "trajectory_timestamp_error_ns"
        ),
        "calibration_timestamp_error_ns": raw.get(
            "calibration_timestamp_error_ns"
        ),
        "vsi_media": raw.get("vsi_media"),
        "rotation_world_from_camera": _rotation_to_canonical(
            source, raw.get("rotation_world_from_camera")
        ),
        "translation_world_from_camera_m": _vector_to_canonical(
            source, raw.get("translation_world_from_camera_m")
        ),
        "visible_nodes": visible,
        "source_schema_version": raw.get("schema_version"),
    }


def adapt_qa(source: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    if source == "adt":
        validate_guide_sampling_binding(raw)
        keys = list(raw["candidate_frame_keys"])
        indices = [int(value) for value in raw["candidate_frame_indices"]]
        media_kind = "video"
    else:
        keys = [raw["frame_key"]]
        indices = [int(raw.get("frame_index", 0))]
        media_kind = "image"
    validate_source_media_contract(source, media_kind, raw["vsi_media"])
    row = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "qa",
        "source_dataset": source,
        "scene_id": raw["scene_id"],
        "qa_id": f"{source}:{raw['vsi_row_index']}",
        "vsi_row_index": int(raw["vsi_row_index"]),
        "vsi_media": raw["vsi_media"],
        "media_kind": media_kind,
        "actual_frame_keys": keys,
        "actual_frame_indices": indices,
        "question_type": raw.get("question_type"),
        "conversations": raw["conversations"],
        "loss_masks": {
            "qa": True,
            "existence": True,
            "category": True,
            "center": bool(raw["loss_masks"].get("scene_geometry", False)),
            "extent": bool(raw["loss_masks"].get("scene_geometry", False)),
            "visibility": True,
            "orientation": False,
            "motion": False,
        },
        "source_schema_version": raw.get("schema_version"),
    }
    if source == "adt":
        if (
            raw.get("qa_evidence_scope")
            != "scene_associated_unlocalized"
            or raw.get("qa_visual_support_verified") is not False
            or raw.get("evidence_frame_indices") is not None
        ):
            raise ContractError(
                "ADT source QA must be "
                "scene_associated_unlocalized/false/null"
            )
        row.update({
            "qa_evidence_scope": raw.get("qa_evidence_scope"),
            "evidence_frame_indices": raw.get("evidence_frame_indices"),
            "qa_visual_support_verified": raw.get(
                "qa_visual_support_verified"
            ),
            "duration_coverage_ratio": raw.get(
                "duration_coverage_ratio"
            ),
            "coverage_bin": raw.get("coverage_bin"),
        })
        row.update({
            "sampling_policy": raw.get("sampling_policy"),
            "video_total_frames": raw.get("total_frames"),
            "video_fps": raw.get("fps"),
            "sampling_base_interval": raw.get("base_interval"),
            "sampling_min_frames": raw.get("min_frames"),
            "sampling_max_frames": raw.get("max_frames"),
            "source_sampling_binding_sha256": raw.get(
                "sampling_binding_sha256"
            ),
            "clip_provenance": raw.get("clip_provenance"),
        })
        validate_guide_sampling_binding(row)
        expected_ratio = duration_coverage_ratio(row["clip_provenance"])
        if float(row["duration_coverage_ratio"]).hex() != expected_ratio.hex():
            raise ContractError("ADT duration coverage provenance mismatch")
        if row["coverage_bin"] != coverage_bin(expected_ratio):
            raise ContractError("ADT coverage bin mismatch")
    else:
        row.update({
            "qa_evidence_scope": "frame_verified",
            "evidence_frame_indices": indices,
            "qa_visual_support_verified": True,
            "duration_coverage_ratio": 1.0,
            "coverage_bin": "high",
        })
    validate_qa_evidence_contract(row)
    return row


@dataclass
class ValidationReport:
    scenes: int = 0
    frames: int = 0
    qa: int = 0
    visible_observations: int = 0
    visibility_evidence_observations: int = 0
    scene_capacity_overflow_scenes: int = 0
    scene_capacity_excess_objects: int = 0
    scene_capacity_scope: str = "whole_scene_nodes_vs_k384"
    source_counts: dict[str, dict[str, int]] | None = None
    scene_object_counts: dict[str, int] | None = None
    qa_coverage_counts: dict[str, dict[str, dict[str, int]]] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "parta_validation_report_v2", **vars(self)}


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def validate_records(
    scenes: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    qa_rows: Sequence[Mapping[str, Any]],
    *,
    max_slots: int = 384,
    require_fixtures: bool = False,
    expected_sources: Sequence[str] | None = None,
    expected_visible_observations: Mapping[str, int] | None = None,
) -> ValidationReport:
    """Validate references, masks, exact input visibility, and frame binding."""
    report = ValidationReport(
        source_counts={}, scene_object_counts={}, qa_coverage_counts={}
    )
    scene_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    frame_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    for scene in scenes:
        if scene.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("Unsupported scene schema")
        key = (scene["source_dataset"], scene["scene_id"])
        if key in scene_map:
            raise ContractError(f"Duplicate scene {key}")
        ids = [node["object_id"] for node in scene["nodes"]]
        if len(ids) != len(set(ids)):
            raise ContractError(f"Duplicate object ID in {key}")
        if len(ids) > max_slots:
            report.scene_capacity_overflow_scenes += 1
            report.scene_capacity_excess_objects += len(ids) - max_slots
        for node in scene["nodes"]:
            for name, valid in node["field_mask"].items():
                field = {
                    "category": "source_category",
                    "center": "center_world_m",
                    "extent": "extent_m",
                    "orientation": "rotation_world_from_object",
                    "motion": "velocity_world_mps",
                }.get(name, name)
                if valid != (node.get(field) is not None):
                    raise ContractError(
                        f"Mask/value mismatch {key} {node['object_id']} {name}"
                    )
            if not _finite(node):
                raise ContractError(f"Non-finite node in {key}")
        scene_map[key] = scene
        report.scenes += 1
        source_stats = report.source_counts.setdefault(
            scene["source_dataset"],
            {
                "scenes": 0,
                "frames": 0,
                "qa": 0,
                "visible_observations": 0,
                "visibility_evidence_observations": 0,
            },
        )
        source_stats["scenes"] += 1
        report.scene_object_counts[f"{key[0]}:{key[1]}"] = len(ids)
    for frame in frames:
        key = (frame["source_dataset"], frame["scene_id"])
        validate_source_media_contract(
            frame["source_dataset"],
            "video" if frame["source_dataset"] == "adt" else "image",
            frame.get("vsi_media"),
        )
        if key not in scene_map:
            raise ContractError(f"Frame references missing scene {key}")
        frame_key = (frame["source_dataset"], frame["frame_key"])
        if frame_key in frame_map:
            raise ContractError(f"Duplicate frame {frame_key}")
        node_ids = {node["object_id"] for node in scene_map[key]["nodes"]}
        observation_ids = [
            observation["object_id"] for observation in frame["visible_nodes"]
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ContractError(
                f"Duplicate visible_nodes object_id in frame {frame_key}"
            )
        for observation in frame["visible_nodes"]:
            evidence_present = observation.get("evidence_present")
            if not isinstance(evidence_present, bool):
                raise ContractError("evidence_present must be boolean")
            if evidence_present:
                report.visibility_evidence_observations += 1
                report.source_counts[key[0]][
                    "visibility_evidence_observations"
                ] += 1
            if observation["visible"] and not evidence_present:
                raise ContractError(
                    "Visible supervision requires source evidence"
                )
            if observation["object_id"] not in node_ids:
                raise ContractError(
                    f"Visible object absent from scene: {observation['object_id']}"
                )
            if observation["visible"] != observation["field_mask"]["visibility"]:
                raise ContractError("Visible observation mask/value mismatch")
            geometry_valid = observation["field_mask"]["camera_geometry"]
            if geometry_valid != (
                observation.get("center_camera_m") is not None
                and observation.get("camera_distance_m") is not None
            ):
                raise ContractError("Camera geometry mask/value mismatch")
            if observation["visible"]:
                report.visible_observations += 1
                report.source_counts[key[0]]["visible_observations"] += 1
        if not _finite(frame):
            raise ContractError(f"Non-finite frame {frame_key}")
        frame_map[frame_key] = frame
        report.frames += 1
        report.source_counts[key[0]]["frames"] += 1
    scene_qa_contracts: dict[tuple[str, str], tuple[Any, ...]] = {}
    for qa in qa_rows:
        scene_key = (qa["source_dataset"], qa["scene_id"])
        validate_source_media_contract(
            qa["source_dataset"], qa.get("media_kind"), qa.get("vsi_media")
        )
        if scene_key not in scene_map:
            raise ContractError(f"QA references missing scene {scene_key}")
        if qa["source_dataset"] == "adt":
            validate_guide_sampling_binding(qa)
        validate_qa_evidence_contract(qa)
        ratio = float(qa.get("duration_coverage_ratio"))
        expected_bin = coverage_bin(ratio)
        if qa.get("coverage_bin") != expected_bin:
            raise ContractError(f"QA coverage bin mismatch: {qa['qa_id']}")
        if qa["source_dataset"] == "adt":
            provenance_ratio = duration_coverage_ratio(
                qa["clip_provenance"]
            )
            if ratio.hex() != provenance_ratio.hex():
                raise ContractError(
                    f"ADT QA coverage/provenance mismatch: {qa['qa_id']}"
                )
        keys = list(qa["actual_frame_keys"])
        indices = list(qa["actual_frame_indices"])
        if len(keys) != len(indices) or len(set(indices)) != len(indices):
            raise ContractError(f"Invalid exact frame binding {qa['qa_id']}")
        declared_frame_binding = qa.get("frame_binding_sha256")
        expected_frame_binding = frame_binding_sha256(qa)
        if declared_frame_binding != expected_frame_binding:
            raise ContractError(
                "Invalid final frame binding SHA256 "
                f"for {qa['qa_id']}: expected={expected_frame_binding}, "
                f"actual={declared_frame_binding}"
            )
        shared_contract = (
            tuple(keys),
            tuple(indices),
            declared_frame_binding,
            qa.get("source_sampling_binding_sha256"),
            qa.get("sampling_policy"),
            ratio.hex(),
            qa.get("coverage_bin"),
            content_sha256(qa.get("clip_provenance")),
        )
        if qa["source_dataset"] == "adt":
            prior_contract = scene_qa_contracts.setdefault(
                scene_key, shared_contract
            )
            if shared_contract != prior_contract:
                raise ContractError(
                    "Same-scene QA selection/binding/coverage mismatch: "
                    f"{scene_key}"
                )
        if qa["media_kind"] == "video" and not 16 <= len(keys) <= 32:
            raise ContractError(f"Video QA must bind 16-32 frames: {qa['qa_id']}")
        if qa["media_kind"] == "image" and len(keys) != 1:
            raise ContractError(f"Image QA must bind one image: {qa['qa_id']}")
        actual_frames = []
        for key, expected_index in zip(keys, indices):
            frame_key = (qa["source_dataset"], key)
            if frame_key not in frame_map:
                raise ContractError(f"QA references missing frame {frame_key}")
            frame = frame_map[frame_key]
            if frame.get("frame_index") != expected_index:
                raise ContractError(
                    "Exact frame key/index mismatch "
                    f"{qa['qa_id']}: key={key}, manifest={expected_index}, "
                    f"frame={frame.get('frame_index')}"
                )
            actual_frames.append(frame)
        visible = {
            item["object_id"]
            for frame in actual_frames
            for item in frame["visible_nodes"]
            if item["visible"] and item["field_mask"]["visibility"]
        }
        if qa["loss_masks"]["existence"] and not visible:
            # Empty-GT is valid, but it must be explicit for the loss path.
            qa_mask = qa.get("empty_gt", False)
            if not qa_mask:
                raise ContractError(
                    f"No actual-input-visible GT without empty_gt: {qa['qa_id']}"
                )
        declared_visible = qa.get("actual_visible_object_ids")
        if declared_visible is not None and set(declared_visible) != visible:
            raise ContractError(
                f"Actual-input-visible GT mismatch: {qa['qa_id']}"
            )
        if not _finite(qa):
            raise ContractError(f"Non-finite QA {qa['qa_id']}")
        report.qa += 1
        report.source_counts[scene_key[0]]["qa"] += 1
        question_type = str(qa.get("question_type") or "__unknown__")
        report.qa_coverage_counts.setdefault(
            scene_key[0], {}
        ).setdefault(question_type, {}).setdefault(expected_bin, 0)
        report.qa_coverage_counts[scene_key[0]][question_type][
            expected_bin
        ] += 1
    if expected_sources is not None:
        unknown = set(expected_sources) - set(T0_FIXTURES)
        if unknown:
            raise ContractError(f"Unknown expected sources: {sorted(unknown)}")
        present_sources = {s["source_dataset"] for s in scenes}
        missing_sources = set(expected_sources) - present_sources
        if missing_sources:
            raise ContractError(
                f"Missing expected sources: {sorted(missing_sources)}"
            )
    if expected_visible_observations is not None:
        for source, expected_count in expected_visible_observations.items():
            if source not in report.source_counts:
                raise ContractError(
                    f"Missing source for visibility audit: {source}"
                )
            actual_count = report.source_counts[source][
                "visible_observations"
            ]
            if actual_count != expected_count:
                raise ContractError(
                    "Source/canonical visible observation mismatch for "
                    f"{source}: source={expected_count}, canonical={actual_count}"
                )
    adt_stats = report.source_counts.get("adt")
    if (
        adt_stats is not None
        and adt_stats["visibility_evidence_observations"] > 0
        and adt_stats["visible_observations"] == 0
    ):
        raise ContractError(
            "ADT contains frame-level visibility evidence but canonical "
            "visibility is empty"
        )
    if require_fixtures:
        present = {(s["source_dataset"], s["scene_id"]) for s in scenes}
        fixture_sources = (
            set(expected_sources)
            if expected_sources is not None
            else {source for source, _ in present}
        )
        missing = [
            (source, scene_id)
            for source, scene_ids in T0_FIXTURES.items()
            if source in fixture_sources
            for scene_id in scene_ids
            if (source, scene_id) not in present
        ]
        if missing:
            raise ContractError(f"Missing fixed T0 fixtures: {missing}")
    return report


def validate_source_media_contract(
    source: str, media_kind: Any, vsi_media: Any
) -> None:
    """Fail closed on source/media swaps before decoding or training."""
    if not isinstance(vsi_media, str) or not vsi_media:
        raise ContractError("vsi_media must be a nonempty relative POSIX path")
    media = vsi_media
    if (
        media != media.strip()
        or "\\" in media
        or any(character in media for character in ("\x00", "?", "#"))
        or media.startswith("/")
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", media)
    ):
        raise ContractError(f"Invalid canonical relative POSIX media path: {media!r}")
    raw_parts = media.split("/")
    if len(raw_parts) < 2 or any(part in {"", ".", ".."} for part in raw_parts):
        raise ContractError(f"Invalid canonical relative POSIX media path: {media!r}")
    path = PurePosixPath(media)
    if path.is_absolute() or tuple(path.parts) != tuple(raw_parts):
        raise ContractError(f"Invalid canonical relative POSIX media path: {media!r}")
    suffix = path.suffix.lower()
    if source == "adt":
        if media_kind != "video" or raw_parts[0] != "adt" or suffix != ".mp4":
            raise ContractError(
                "ADT requires media_kind=video, adt/ prefix, and an MP4 path"
            )
        return
    if source == "hypersim":
        if (
            media_kind != "image"
            or raw_parts[0] != "hypersim"
            or suffix not in {".jpg", ".jpeg", ".png"}
        ):
            raise ContractError(
                "Hypersim requires media_kind=image, hypersim/ prefix, and an image path"
            )
        return
    raise ContractError(f"Unsupported source/media contract: {source!r}")


def build_manifest_rows(
    qa_rows: Iterable[Mapping[str, Any]],
    frame_lookup: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Add stable exact-frame and actual-visible-object bindings to QA rows."""
    for qa in qa_rows:
        frames = [
            frame_lookup[(qa["source_dataset"], key)]
            for key in qa["actual_frame_keys"]
        ]
        visible_ids = sorted(
            {
                item["object_id"]
                for frame in frames
                for item in frame["visible_nodes"]
                if item["visible"] and item["field_mask"]["visibility"]
            }
        )
        row = dict(qa)
        row["actual_visible_object_ids"] = visible_ids
        row["empty_gt"] = not visible_ids
        row["frame_binding_sha256"] = frame_binding_sha256(row)
        yield row
