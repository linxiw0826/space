"""Fail-closed canonical data path for the Part A T0/A1-O side branch.

This module intentionally does not reuse the generic Qwen training dataset:
that path may resample videos and drops the exact canonical state metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from parta_data_contract import (
    CANONICAL_CATEGORIES,
    ContractError,
    T0_FIXTURES,
    read_jsonl,
    validate_records,
    validate_source_media_contract,
)

from .state_loss import StateTargets

ALLOWED_EXACT_QA_FILENAMES = frozenset({"qa_manifest_exact_verified.jsonl"})


@dataclass(frozen=True)
class PartASample:
    scene: Mapping[str, Any]
    frames: tuple[Mapping[str, Any], ...]
    qa: Mapping[str, Any]


@dataclass(frozen=True)
class TargetSelectionAudit:
    actual_input_visible_object_count: int
    selected_object_ids: tuple[str, ...]
    truncated_object_ids: tuple[str, ...]
    selection_scope: str = "actual_input_per_qa"
    selection_policy: str = "visible_frame_count_desc_then_object_id_v1"


class PartACanonicalDataset:
    """Join canonical scene/frame/QA rows and expose only frozen fixtures.

    Every source is validated independently before rows are joined. Missing
    fixtures, duplicate IDs, and any key/index mismatch are fatal; automatic
    fixture substitution is forbidden by D-58.
    """

    def __init__(
        self,
        source_roots: Mapping[str, str | Path],
        *,
        qa_filename: str = "qa_manifest_exact_verified.jsonl",
        fixture_only: bool = True,
    ) -> None:
        expected = set(T0_FIXTURES)
        if set(source_roots) != expected:
            raise ContractError(
                f"T0-A requires exactly ADT and Hypersim roots, got {sorted(source_roots)}"
            )
        _validate_qa_filename(qa_filename)
        samples: list[PartASample] = []
        available_fixtures: set[tuple[str, str]] = set()
        seen_qa_ids: set[str] = set()
        for source in sorted(source_roots):
            root = Path(source_roots[source]).resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"canonical {source} root is not a directory: {root}")
            paths = {
                "scene": root / "scene_states.jsonl",
                "frame": root / "frame_states.jsonl",
                "qa": root / qa_filename,
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"canonical {source} files missing: {missing}")
            scenes = list(read_jsonl(paths["scene"]))
            frames = list(read_jsonl(paths["frame"]))
            qa_rows = list(read_jsonl(paths["qa"]))
            for record_kind, rows in (
                ("scene", scenes),
                ("frame", frames),
                ("qa", qa_rows),
            ):
                _assert_exact_source(source, record_kind, rows)
            validate_records(
                scenes,
                frames,
                qa_rows,
                require_fixtures=True,
                expected_sources=(source,),
            )
            scene_map = {(row["source_dataset"], row["scene_id"]): row for row in scenes}
            frame_map = {(row["source_dataset"], row["frame_key"]): row for row in frames}
            fixture_ids = set(T0_FIXTURES[source])
            for row in qa_rows:
                if row["qa_id"] in seen_qa_ids:
                    raise ContractError(f"duplicate cross-source qa_id: {row['qa_id']}")
                seen_qa_ids.add(row["qa_id"])
                if fixture_only and row["scene_id"] not in fixture_ids:
                    continue
                key = (source, row["scene_id"])
                joined_frames = tuple(frame_map[(source, frame_key)] for frame_key in row["actual_frame_keys"])
                samples.append(PartASample(scene_map[key], joined_frames, row))
                available_fixtures.add(key)
        required = {
            (source, scene_id)
            for source, scene_ids in T0_FIXTURES.items()
            for scene_id in scene_ids
        }
        missing_fixtures = sorted(required - available_fixtures)
        if missing_fixtures:
            raise ContractError(
                "fixed fixtures have no QA rows (substitution forbidden): "
                f"{missing_fixtures}"
            )
        self.samples = tuple(sorted(samples, key=lambda item: item.qa["qa_id"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PartASample:
        return self.samples[index]


def _assert_exact_source(
    expected_source: str,
    record_kind: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    wrong = [
        row.get("source_dataset")
        for row in rows
        if row.get("source_dataset") != expected_source
    ]
    if wrong:
        raise ContractError(
            f"canonical {expected_source} root contains mixed/swapped "
            f"{record_kind} sources: {sorted({str(value) for value in wrong})}"
        )


class ExactMediaLoader:
    """Load only the media and exact indices declared by the canonical row."""

    def __init__(
        self,
        media_root: str | Path,
        *,
        video_reader_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.media_root = Path(media_root).resolve()
        if not self.media_root.is_dir():
            raise FileNotFoundError(f"media root is not a directory: {self.media_root}")
        self.video_reader_factory = video_reader_factory

    def load(self, sample: PartASample) -> tuple[Image.Image, ...]:
        qa = sample.qa
        validate_source_media_contract(
            qa.get("source_dataset"), qa.get("media_kind"), qa.get("vsi_media")
        )
        source = qa["source_dataset"]
        media_parts = qa["vsi_media"].split("/")
        source_root = (self.media_root / source).resolve()
        try:
            source_root.relative_to(self.media_root)
        except ValueError as error:
            raise ContractError(f"invalid source-specific media root: {source_root}") from error
        path = source_root.joinpath(*media_parts[1:]).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ContractError(
                f"media path escapes source-specific root: {qa['vsi_media']}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"exact media missing for {qa['qa_id']}: {path}")
        indices = [int(value) for value in qa["actual_frame_indices"]]
        if qa["media_kind"] == "image":
            if len(indices) != 1:
                raise ContractError("image sample must have exactly one bound frame")
            with Image.open(path) as image:
                return (image.convert("RGB").copy(),)
        if qa["media_kind"] != "video":
            raise ContractError(f"unsupported media_kind={qa['media_kind']!r}")
        if not 16 <= len(indices) <= 32 or len(indices) != len(set(indices)):
            raise ContractError("video sample must have 16-32 unique exact indices")
        reader_factory = self.video_reader_factory
        if reader_factory is None:
            try:
                from decord import VideoReader
            except ImportError as error:
                raise RuntimeError("decord is required for exact ADT MP4 decoding") from error
            reader_factory = VideoReader
        reader = reader_factory(str(path))
        if min(indices) < 0 or max(indices) >= len(reader):
            raise ContractError(
                f"bound frame outside video for {qa['qa_id']}: length={len(reader)}, indices={indices}"
            )
        array = reader.get_batch(indices).asnumpy()
        if array.shape[0] != len(indices):
            raise ContractError("video reader did not return every exact frame")
        return tuple(Image.fromarray(frame).convert("RGB") for frame in array)


def build_state_targets(
    sample: PartASample,
    *,
    max_objects: int = 384,
) -> tuple[StateTargets, TargetSelectionAudit]:
    """Build question-independent GT from actual encoded views only."""
    if max_objects != 384:
        raise ValueError("D-58 freezes the object slot budget to 384")
    qa = sample.qa
    validate_source_media_contract(
        qa.get("source_dataset"), qa.get("media_kind"), qa.get("vsi_media")
    )
    nodes = {node["object_id"]: node for node in sample.scene["nodes"]}
    frame_observations: list[dict[str, Mapping[str, Any]]] = []
    visible_counts: dict[str, int] = {}
    for frame in sample.frames:
        observation_ids = [row["object_id"] for row in frame["visible_nodes"]]
        if len(observation_ids) != len(set(observation_ids)):
            raise ContractError(
                f"duplicate visible_nodes object_id in frame {frame['frame_key']}"
            )
        observations = {
            row["object_id"]: row
            for row in frame["visible_nodes"]
            if row.get("evidence_present") is True
        }
        frame_observations.append(observations)
        for object_id, observation in observations.items():
            if observation.get("visible") is True and observation["field_mask"].get("visibility") is True:
                visible_counts[object_id] = visible_counts.get(object_id, 0) + 1
    declared = set(qa.get("actual_visible_object_ids", ()))
    if declared != set(visible_counts):
        raise ContractError(
            f"runtime visible union differs from manifest for {qa['qa_id']}"
        )
    ranked = sorted(visible_counts, key=lambda object_id: (-visible_counts[object_id], object_id))
    declared_selected = qa.get("selected_object_ids")
    declared_truncated = qa.get("truncated_object_ids")
    if declared_selected is not None or declared_truncated is not None:
        if not isinstance(declared_selected, list) or not isinstance(declared_truncated, list):
            raise ContractError("selected/truncated object IDs must be declared together as lists")
        selected = tuple(str(value) for value in declared_selected)
        truncated = tuple(str(value) for value in declared_truncated)
        expected_selected = tuple(ranked[:max_objects])
        expected_truncated = tuple(ranked[max_objects:])
        if selected != expected_selected or truncated != expected_truncated:
            raise ContractError(f"invalid manifest object selection for {qa['qa_id']}")
    else:
        selected = tuple(ranked[:max_objects])
        truncated = tuple(ranked[max_objects:])
    if any(object_id not in nodes for object_id in selected):
        raise ContractError(f"selected object absent from scene for {qa['qa_id']}")
    category_index = {name: index for index, name in enumerate(CANONICAL_CATEGORIES)}
    categories, centers, extents = [], [], []
    category_valid, center_valid, extent_valid = [], [], []
    visibility, visibility_valid = [], []
    for object_id in selected:
        node = nodes[object_id]
        masks = node["field_mask"]
        category = node.get("category") or "__unknown__"
        categories.append(category_index.get(category, 0))
        category_valid.append(bool(masks.get("category")))
        centers.append(node.get("center_world_m") or (0.0, 0.0, 0.0))
        extents.append(node.get("extent_m") or (0.0, 0.0, 0.0))
        center_valid.append(bool(masks.get("center")))
        extent_valid.append(bool(masks.get("extent")))
        object_visibility, object_validity = [], []
        for observations in frame_observations:
            observation = observations.get(object_id)
            valid = observation is not None and observation["field_mask"].get("visibility") is True
            object_validity.append(valid)
            object_visibility.append(bool(valid and observation.get("visible") is True))
        visibility.append(object_visibility)
        visibility_valid.append(object_validity)
    count, frames = len(selected), len(sample.frames)
    centers_tensor = torch.tensor(centers, dtype=torch.float32).reshape(count, 3)
    extents_tensor = torch.tensor(extents, dtype=torch.float32).reshape(count, 3)
    geometry_mask = torch.tensor(center_valid, dtype=torch.bool) & torch.tensor(extent_valid, dtype=torch.bool)
    if geometry_mask.any():
        lower = centers_tensor[geometry_mask] - 0.5 * extents_tensor[geometry_mask]
        upper = centers_tensor[geometry_mask] + 0.5 * extents_tensor[geometry_mask]
        scene_scale = torch.linalg.vector_norm(upper.max(0).values - lower.min(0).values).clamp_min(1e-3)
    else:
        scene_scale = torch.tensor(1.0)
    targets = StateTargets(
        categories=torch.tensor(categories, dtype=torch.long),
        centers_world_m=centers_tensor,
        extents_m=extents_tensor,
        visibility=torch.tensor(visibility, dtype=torch.float32).reshape(count, frames),
        category_valid=torch.tensor(category_valid, dtype=torch.bool),
        center_valid=torch.tensor(center_valid, dtype=torch.bool),
        extent_valid=torch.tensor(extent_valid, dtype=torch.bool),
        visibility_valid=torch.tensor(visibility_valid, dtype=torch.bool).reshape(count, frames),
        scene_scale_m=scene_scale,
        source_dataset=qa["source_dataset"],
        scene_id=qa["scene_id"],
    )
    targets.validate(max_frames=32)
    return targets, TargetSelectionAudit(len(ranked), selected, truncated)


def _validate_qa_filename(qa_filename: str) -> None:
    if (
        not isinstance(qa_filename, str)
        or qa_filename not in ALLOWED_EXACT_QA_FILENAMES
        or Path(qa_filename).name != qa_filename
        or "\\" in qa_filename
    ):
        raise ContractError(
            "qa_filename must be the frozen safe basename "
            f"{sorted(ALLOWED_EXACT_QA_FILENAMES)}"
        )
