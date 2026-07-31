"""Versioned, source-neutral data contract for Part A A1-O.

The contract deliberately contains only JSON-compatible values.  Missing
supervision is represented by ``None`` plus an explicit false field mask; a
numeric zero is always a real measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "parta_canonical_v1"
GUIDE_EXACT_SAMPLING_POLICY = "guide_exact_raw_mp4_v1"
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
) -> dict[str, Any]:
    """Return the unique stable payload for exact GUIDE frame binding.

    Floats use their exact IEEE-754 hexadecimal representation so every
    producer and consumer hashes one unambiguous value.
    """
    fps = float(fps)
    base_interval = float(base_interval)
    if not math.isfinite(fps) or not math.isfinite(base_interval):
        raise ContractError("Sampling FPS/base interval must be finite")
    return {
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
    missing = [key for key, value in field_map.items() if value is None]
    if missing:
        raise ContractError(f"Missing GUIDE sampling fields: {missing}")
    expected = guide_sampling_binding_sha256(
        source_dataset=row.get("source_dataset", "adt"),
        scene_id=row["scene_id"],
        vsi_media=row["vsi_media"],
        sampling_policy=policy,
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
    recomputed_indices = guide_frame_indices(
        int(field_map["total_frames"]),
        float(field_map["fps"]),
        base_interval=float(field_map["base_interval"]),
        min_frames=int(field_map["min_frames"]),
        max_frames=int(field_map["max_frames"]),
    )
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
    return {
        "object_id": _source_object_id(source, scene_id, raw["object_id"]),
        "source_object_id": str(raw["object_id"]),
        "category": category,
        "source_category": source_category,
        "center_world_m": center,
        "extent_m": extent,
        "rotation_world_from_object": rotation,
        "motion_type": raw.get("motion_type"),
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
    for observation in raw.get("visible_nodes", ()):
        evidence_present, supervision_valid = source_visibility_contract(
            source, observation
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
        })
        validate_guide_sampling_binding(row)
    return row


@dataclass
class ValidationReport:
    scenes: int = 0
    frames: int = 0
    qa: int = 0
    visible_observations: int = 0
    visibility_evidence_observations: int = 0
    overflow_scenes: int = 0
    truncated_objects: int = 0
    source_counts: dict[str, dict[str, int]] | None = None
    scene_object_counts: dict[str, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": "parta_validation_report_v1", **vars(self)}


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
    report = ValidationReport(source_counts={}, scene_object_counts={})
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
            report.overflow_scenes += 1
            report.truncated_objects += len(ids) - max_slots
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
        if key not in scene_map:
            raise ContractError(f"Frame references missing scene {key}")
        frame_key = (frame["source_dataset"], frame["frame_key"])
        if frame_key in frame_map:
            raise ContractError(f"Duplicate frame {frame_key}")
        node_ids = {node["object_id"] for node in scene_map[key]["nodes"]}
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
    for qa in qa_rows:
        scene_key = (qa["source_dataset"], qa["scene_id"])
        if scene_key not in scene_map:
            raise ContractError(f"QA references missing scene {scene_key}")
        if qa["source_dataset"] == "adt":
            validate_guide_sampling_binding(qa)
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
