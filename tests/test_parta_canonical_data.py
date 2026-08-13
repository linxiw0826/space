import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.canonical_data import (
    ExactMediaLoader,
    PartACanonicalDataset,
    PartASample,
    _assert_exact_source,
    build_state_targets,
)
from parta_data_contract import ContractError


def _sample(media_kind="image", frame_count=1, object_count=2):
    source = "hypersim" if media_kind == "image" else "adt"
    nodes = []
    for index in range(object_count):
        nodes.append(
            {
                "object_id": f"{source}:s:o{index:03d}",
                "category": "chair",
                "center_world_m": [float(index), 0.0, 0.0],
                "extent_m": [1.0, 1.0, 1.0],
                "field_mask": {
                    "category": True,
                    "center": True,
                    "extent": True,
                    "orientation": False,
                    "motion": False,
                },
            }
        )
    frames = []
    for frame_index in range(frame_count):
        observations = []
        for index in range(object_count):
            if (index + frame_index) % 2 == 0 or frame_count == 1:
                observations.append(
                    {
                        "object_id": f"{source}:s:o{index:03d}",
                        "evidence_present": True,
                        "visible": True,
                        "field_mask": {"visibility": True},
                    }
                )
        frames.append(
            {
                "source_dataset": source,
                "scene_id": "s",
                "frame_key": str(frame_index),
                "frame_index": frame_index,
                "visible_nodes": observations,
            }
        )
    visible = sorted(
        {
            row["object_id"]
            for frame in frames
            for row in frame["visible_nodes"]
        }
    )
    qa = {
        "qa_id": "hypersim:q",
        "source_dataset": source,
        "scene_id": "s",
        "vsi_media": (
            "hypersim/sample.png" if media_kind == "image" else "adt/sample.mp4"
        ),
        "media_kind": media_kind,
        "actual_frame_indices": list(range(frame_count)),
        "actual_frame_keys": [str(index) for index in range(frame_count)],
        "actual_visible_object_ids": visible,
    }
    return PartASample(
        scene={"source_dataset": source, "scene_id": "s", "nodes": nodes},
        frames=tuple(frames),
        qa=qa,
    )


class _FakeBatch:
    def __init__(self, array):
        self.array = array

    def asnumpy(self):
        return self.array


class _FakeReader:
    def __init__(self, _path):
        self.frames = np.stack(
            [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(40)]
        )

    def __len__(self):
        return len(self.frames)

    def get_batch(self, indices):
        return _FakeBatch(self.frames[indices])


def test_exact_media_loader_image_and_video_preserve_declared_order(tmp_path):
    (tmp_path / "hypersim").mkdir()
    (tmp_path / "adt").mkdir()
    Image.new("RGB", (3, 2), color=(1, 2, 3)).save(
        tmp_path / "hypersim/sample.png"
    )
    image_frames = ExactMediaLoader(tmp_path).load(_sample())
    assert len(image_frames) == 1
    assert image_frames[0].getpixel((0, 0)) == (1, 2, 3)

    (tmp_path / "adt/sample.mp4").touch()
    video_sample = _sample(media_kind="video", frame_count=16)
    video_sample.qa["source_dataset"] = "adt"
    video_sample.qa["actual_frame_indices"][:] = [15 - index for index in range(16)]
    frames = ExactMediaLoader(
        tmp_path, video_reader_factory=_FakeReader
    ).load(video_sample)
    assert [frame.getpixel((0, 0))[0] for frame in frames] == list(range(15, -1, -1))


def test_exact_video_loader_rejects_out_of_range_without_fallback(tmp_path):
    (tmp_path / "adt").mkdir()
    (tmp_path / "adt/sample.mp4").touch()
    sample = _sample(media_kind="video", frame_count=16)
    sample.qa["source_dataset"] = "adt"
    sample.qa["actual_frame_indices"][-1] = 99
    with pytest.raises(ContractError, match="outside video"):
        ExactMediaLoader(tmp_path, video_reader_factory=_FakeReader).load(sample)


def test_targets_use_actual_visible_union_and_question_independent_overflow():
    sample = _sample(media_kind="video", frame_count=16, object_count=385)
    targets, audit = build_state_targets(sample)
    assert targets.num_objects == 384
    assert audit.actual_input_visible_object_count == 385
    assert audit.selection_scope == "actual_input_per_qa"
    assert len(audit.truncated_object_ids) == 1
    assert audit.selected_object_ids == tuple(sorted(audit.selected_object_ids))
    assert targets.visibility.shape == (384, 16)
    assert targets.scene_scale_m.item() > 0

    changed_question = dict(sample.qa)
    changed_question["conversations"] = [{"from": "human", "value": "different"}]
    _, second_audit = build_state_targets(
        PartASample(sample.scene, sample.frames, changed_question)
    )
    assert second_audit == audit


def test_targets_fail_if_runtime_visibility_differs_from_bound_manifest():
    sample = _sample()
    sample.qa["actual_visible_object_ids"] = []
    with pytest.raises(ContractError, match="visible union"):
        build_state_targets(sample)


def test_targets_respect_valid_manifest_selection_and_reject_bad_selection():
    sample = _sample(media_kind="video", frame_count=16, object_count=385)
    ranked = sorted(sample.qa["actual_visible_object_ids"])
    sample.qa["selected_object_ids"] = ranked[:384]
    sample.qa["truncated_object_ids"] = ranked[384:]
    _, audit = build_state_targets(sample)
    assert audit.selected_object_ids == tuple(ranked[:384])
    assert audit.truncated_object_ids == tuple(ranked[384:])

    sample.qa["selected_object_ids"] = ranked[1:]
    sample.qa["truncated_object_ids"] = ranked[:1]
    with pytest.raises(ContractError, match="invalid manifest object selection"):
        build_state_targets(sample)


@pytest.mark.parametrize(
    ("source", "media_kind", "media", "message"),
    [
        ("adt", "image", "adt/x.mp4", "ADT requires"),
        ("adt", "video", "hypersim/x.mp4", "ADT requires"),
        ("adt", "video", "adt/x.jpg", "ADT requires"),
        ("hypersim", "video", "hypersim/x.jpg", "Hypersim requires"),
        ("hypersim", "image", "adt/x.jpg", "Hypersim requires"),
        ("hypersim", "image", "hypersim/x.mp4", "Hypersim requires"),
    ],
)
def test_source_media_contract_rejects_cross_source_attacks(
    tmp_path, source, media_kind, media, message
):
    sample = _sample()
    sample.qa.update(
        {"source_dataset": source, "media_kind": media_kind, "vsi_media": media}
    )
    with pytest.raises(ContractError, match=message):
        build_state_targets(sample)


def test_target_builder_rejects_duplicate_visible_node_ids():
    sample = _sample()
    sample.frames[0]["visible_nodes"].append(
        dict(sample.frames[0]["visible_nodes"][0])
    )
    with pytest.raises(ContractError, match="duplicate visible_nodes"):
        build_state_targets(sample)


@pytest.mark.parametrize("kind", ["scene", "frame", "qa"])
def test_canonical_root_rejects_mixed_or_swapped_source_before_filtering(kind):
    rows = [
        {"source_dataset": "adt", "scene_id": "fixture"},
        {"source_dataset": "hypersim", "scene_id": "non_fixture_attack"},
    ]
    with pytest.raises(ContractError, match="mixed/swapped"):
        _assert_exact_source("adt", kind, rows)


@pytest.mark.parametrize(
    "qa_filename",
    [
        "/tmp/qa_manifest_exact_verified.jsonl",
        "../qa_manifest_exact_verified.jsonl",
        "nested/qa_manifest_exact_verified.jsonl",
        "qa_manifest.jsonl",
        "qa_manifest_exact_verified.jsonl\\attack",
    ],
)
def test_dataset_rejects_unsafe_or_unfrozen_qa_filename(tmp_path, qa_filename):
    (tmp_path / "adt").mkdir()
    (tmp_path / "hypersim").mkdir()
    with pytest.raises(ContractError, match="frozen safe basename"):
        PartACanonicalDataset(
            {"adt": tmp_path / "adt", "hypersim": tmp_path / "hypersim"},
            qa_filename=qa_filename,
        )


def test_generic_canonical_dataset_accepts_known_source_without_t0_fixtures(
    tmp_path, monkeypatch
):
    root = tmp_path / "scannetppv2"
    root.mkdir()
    for filename in (
        "scene_states.jsonl",
        "frame_states.jsonl",
        "qa_manifest_exact_verified.jsonl",
    ):
        (root / filename).touch()

    rows = {
        "scene_states.jsonl": [
            {"source_dataset": "scannetppv2", "scene_id": "scene"}
        ],
        "frame_states.jsonl": [
            {
                "source_dataset": "scannetppv2",
                "scene_id": "scene",
                "frame_key": "frame",
            }
        ],
        "qa_manifest_exact_verified.jsonl": [
            {
                "source_dataset": "scannetppv2",
                "scene_id": "scene",
                "qa_id": "scannetppv2:q",
                "actual_frame_keys": ["frame"],
            }
        ],
    }
    validation_calls = []
    monkeypatch.setattr(
        "parta.canonical_data.read_jsonl", lambda path: iter(rows[path.name])
    )
    monkeypatch.setattr(
        "parta.canonical_data.validate_records",
        lambda *args, **kwargs: validation_calls.append(kwargs),
    )

    dataset = PartACanonicalDataset(
        {"scannetppv2": root}, fixture_only=False, require_fixtures=False
    )

    assert len(dataset) == 1
    assert dataset[0].qa["qa_id"] == "scannetppv2:q"
    assert validation_calls == [
        {"require_fixtures": False, "expected_sources": ("scannetppv2",)}
    ]


def test_fixture_only_still_requires_strict_fixture_validation():
    with pytest.raises(
        ValueError, match="fixture_only=True requires require_fixtures=True"
    ):
        PartACanonicalDataset(
            {"scannetppv2": "/unused"},
            fixture_only=True,
            require_fixtures=False,
        )


def test_strict_fixture_mode_rejects_non_t0_source_before_reading():
    with pytest.raises(ContractError, match="requires exactly ADT and Hypersim roots"):
        PartACanonicalDataset(
            {"scannetppv2": "/unused"},
            fixture_only=False,
            require_fixtures=True,
        )


@pytest.mark.parametrize(
    "media",
    [
        "/hypersim/x.jpg",
        "hypersim/../x.jpg",
        "hypersim/./x.jpg",
        "hypersim//x.jpg",
        "hypersim\\x.jpg",
        "file:hypersim/x.jpg",
        "https://hypersim/x.jpg",
        "hypersim/x.jpg?query=1",
        "hypersim/x.jpg#fragment",
        "",
    ],
)
def test_loader_rejects_noncanonical_relative_posix_media_paths(tmp_path, media):
    (tmp_path / "hypersim").mkdir()
    sample = _sample()
    sample.qa["vsi_media"] = media
    with pytest.raises(ContractError, match="POSIX|Hypersim requires"):
        ExactMediaLoader(tmp_path).load(sample)


def test_loader_rejects_symlink_escape_from_source_specific_root(tmp_path):
    (tmp_path / "hypersim").mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (1, 1)).save(outside)
    (tmp_path / "hypersim/escape.jpg").symlink_to(outside)
    sample = _sample()
    sample.qa["vsi_media"] = "hypersim/escape.jpg"
    with pytest.raises(ContractError, match="source-specific root"):
        ExactMediaLoader(tmp_path).load(sample)
