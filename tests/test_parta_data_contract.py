import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.adt_gt_supported_clip import build_support_certificate
from src.parta_data_contract import (
    ContractError,
    GUIDE_EXACT_SAMPLING_POLICY,
    T0_FIXTURES,
    adapt_frame,
    adapt_qa,
    adapt_scene,
    build_manifest_rows,
    canonical_category,
    coverage_bin,
    duration_coverage_ratio,
    frame_binding_sha256,
    guide_frame_indices,
    guide_sampling_binding_sha256,
    source_visibility_contract,
    validate_qa_evidence_contract,
    validate_records,
)

PROJECT = Path(__file__).resolve().parents[1]
FINALIZER_PATH = (
    PROJECT / "scripts/preprocess/finalize_adt_parta_training_data.py"
)
FINALIZER_SPEC = importlib.util.spec_from_file_location(
    "finalize_adt_parta_training_data", FINALIZER_PATH
)
FINALIZER = importlib.util.module_from_spec(FINALIZER_SPEC)
assert FINALIZER_SPEC.loader is not None
FINALIZER_SPEC.loader.exec_module(FINALIZER)


def raw_records(scene_id="Apartment_release_clean_seq131_M1292"):
    scene = {
        "schema_version": "adt_scene_state_v1",
        "scene_id": scene_id,
        "nodes": [{
            "object_id": "7",
            "category": "chair",
            "center_world_m": [1.0, 2.0, 3.0],
            "extent_m": [0.5, 0.6, 0.7],
            "rotation_world_from_object": None,
            "motion_type": "static",
        }],
    }
    frames = []
    for index in range(16):
        frames.append({
            "schema_version": "adt_frame_state_v1",
            "scene_id": scene_id,
            "frame_key": f"{scene_id}/{index}",
            "frame_index": index,
            "device_timestamp_ns": index * 1_000_000_000,
            "trajectory_timestamp_error_ns": 0,
            "calibration_timestamp_error_ns": 0,
            "vsi_media": "adt/a.mp4",
            "rotation_world_from_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_world_from_camera_m": [0, 0, 0],
            "visible_nodes": [{
                "object_id": "7",
                "object_geometry_valid": True,
                "center_camera_m": [1, 2, 3],
                "camera_distance_m": 4,
            }],
        })
    qa = {
        "schema_version": "adt_qa_train_v1",
        "vsi_row_index": 9,
        "scene_id": scene_id,
        "candidate_frame_keys": [row["frame_key"] for row in frames],
        "candidate_frame_indices": list(range(16)),
        "vsi_media": "adt/a.mp4",
        "question_type": "relative_direction_object",
        "conversations": [],
        "qa_evidence_scope": "scene_associated_unlocalized",
        "evidence_frame_indices": None,
        "qa_visual_support_verified": False,
        "duration_coverage_ratio": 1.0,
        "coverage_bin": "high",
        "loss_masks": {"scene_geometry": True},
        "sampling_policy": GUIDE_EXACT_SAMPLING_POLICY,
        "total_frames": 16,
        "fps": 1.0,
        "base_interval": 1.0,
        "min_frames": 16,
        "max_frames": 32,
        "clip_provenance": {
            "whole_video_total_frames": 16,
            "whole_video_fps_hex": float(1.0).hex(),
            "clip_start_raw_frame": 0,
            "clip_end_raw_frame": 15,
            "clip_start_device_timestamp_ns": 0,
            "clip_end_device_timestamp_ns": 15_000_000_000,
            "support_runs": [{
                "start_raw_frame": 0,
                "end_raw_frame": 15,
                "frame_count": 16,
                "start_device_timestamp_ns": 0,
                "end_device_timestamp_ns": 15_000_000_000,
            }],
            "tie_policy": "longest_run_then_earliest_start_v1",
            "hard_support_fields": ["trajectory", "calibration"],
            "max_trajectory_error_ns": 5_000_000,
            "max_calibration_error_ns": 50_000_000,
            "local_frame_indices": list(range(16)),
            "selected_device_timestamps_ns": [
                index * 1_000_000_000 for index in range(16)
            ],
        },
    }
    qa["clip_provenance"]["whole_video_start_device_timestamp_ns"] = 0
    qa["clip_provenance"]["whole_video_end_device_timestamp_ns"] = (
        15_000_000_000
    )
    qa["clip_provenance"]["clip_frame_count"] = 16
    qa["clip_provenance"]["support_certificate"] = (
        build_support_certificate(
            scene_id=scene_id,
            vsi_media="adt/a.mp4",
            frame_timestamps=[
                index * 1_000_000_000 for index in range(16)
            ],
            fps=1.0,
            support_mask=[True] * 16,
            max_trajectory_error_ns=5_000_000,
            max_calibration_error_ns=50_000_000,
        )
    )
    qa["sampling_binding_sha256"] = guide_sampling_binding_sha256(
        source_dataset="adt",
        scene_id=scene_id,
        vsi_media=qa["vsi_media"],
        frame_keys=qa["candidate_frame_keys"],
        frame_indices=qa["candidate_frame_indices"],
        total_frames=qa["total_frames"],
        fps=qa["fps"],
        base_interval=qa["base_interval"],
        min_frames=qa["min_frames"],
        max_frames=qa["max_frames"],
        sampling_policy=qa["sampling_policy"],
        clip_provenance=qa["clip_provenance"],
    )
    return scene, frames, qa


def canonical():
    raw_scene, raw_frames, raw_qa = raw_records()
    scene = adapt_scene("adt", raw_scene)
    frames = [adapt_frame("adt", row) for row in raw_frames]
    qa = adapt_qa("adt", raw_qa)
    lookup = {("adt", row["frame_key"]): row for row in frames}
    manifest = list(build_manifest_rows([qa], lookup))
    return [scene], frames, manifest


def test_adapt_and_validate_reject_duplicate_visible_node_ids():
    _, raw_frames, _ = raw_records()
    duplicated_raw = copy.deepcopy(raw_frames[0])
    duplicated_raw["visible_nodes"].append(
        copy.deepcopy(duplicated_raw["visible_nodes"][0])
    )
    with pytest.raises(ContractError, match="Duplicate visible_nodes"):
        adapt_frame("adt", duplicated_raw)

    scenes, frames, qa = canonical()
    frames[0]["visible_nodes"].append(
        copy.deepcopy(frames[0]["visible_nodes"][0])
    )
    with pytest.raises(ContractError, match="Duplicate visible_nodes"):
        validate_records(scenes, frames, qa)


@pytest.mark.parametrize(
    ("media_kind", "vsi_media"),
    [
        ("image", "adt/a.mp4"),
        ("video", "hypersim/a.mp4"),
        ("video", "adt/a.jpg"),
    ],
)
def test_canonical_validator_rejects_adt_media_contract_attacks(
    media_kind, vsi_media
):
    scenes, frames, qa = canonical()
    qa[0]["media_kind"] = media_kind
    qa[0]["vsi_media"] = vsi_media
    with pytest.raises(ContractError, match="ADT requires"):
        validate_records(scenes, frames, qa)


@pytest.mark.parametrize(
    "vsi_media",
    [
        "/hypersim/a.png",
        "hypersim/../a.png",
        "hypersim/./a.png",
        "hypersim//a.png",
        "hypersim\\a.png",
        "https://hypersim/a.png",
        "",
    ],
)
def test_adapt_qa_rejects_noncanonical_media_paths(vsi_media):
    raw = {
        "schema_version": "hypersim_qa_train_v1",
        "vsi_row_index": 1,
        "scene_id": "ai_001_001",
        "frame_key": "ai_001_001/0",
        "frame_index": 0,
        "vsi_media": vsi_media,
        "question_type": "absolute_count",
        "conversations": [],
        "loss_masks": {"scene_geometry": True},
    }
    with pytest.raises(ContractError, match="POSIX"):
        adapt_qa("hypersim", raw)


@pytest.mark.parametrize(
    "vsi_media",
    ["/adt/a.mp4", "adt/../a.mp4", "adt//a.mp4", "adt\\a.mp4"],
)
def test_canonical_validator_rejects_noncanonical_frame_media(vsi_media):
    scenes, frames, qa = canonical()
    frames[0]["vsi_media"] = vsi_media
    with pytest.raises(ContractError, match="POSIX"):
        validate_records(scenes, frames, qa)


def rehash_canonical_qa(row):
    row["source_sampling_binding_sha256"] = (
        guide_sampling_binding_sha256(
            source_dataset=row["source_dataset"],
            scene_id=row["scene_id"],
            vsi_media=row["vsi_media"],
            frame_keys=row["actual_frame_keys"],
            frame_indices=row["actual_frame_indices"],
            total_frames=row["video_total_frames"],
            fps=row["video_fps"],
            base_interval=row["sampling_base_interval"],
            min_frames=row["sampling_min_frames"],
            max_frames=row["sampling_max_frames"],
            sampling_policy=row["sampling_policy"],
            clip_provenance=row["clip_provenance"],
        )
    )
    row["frame_binding_sha256"] = frame_binding_sha256(row)


def test_valid_contract_null_and_mask():
    scenes, frames, qa = canonical()
    node = scenes[0]["nodes"][0]
    assert node["rotation_world_from_object"] is None
    assert node["field_mask"]["orientation"] is False
    report = validate_records(scenes, frames, qa)
    assert report.qa == 1
    assert report.scene_capacity_overflow_scenes == 0
    assert report.scene_capacity_excess_objects == 0
    assert report.scene_capacity_scope == "whole_scene_nodes_vs_k384"
    assert report.as_dict()["schema_version"] == "parta_validation_report_v2"
    assert "overflow_scenes" not in report.as_dict()
    assert "truncated_objects" not in report.as_dict()
    assert report.qa_coverage_counts["adt"][
        "relative_direction_object"
    ]["high"] == 1
    assert frames[0]["visible_nodes"][0]["visible"] is True
    assert frames[0]["visible_nodes"][0]["evidence_present"] is True


def test_scene_capacity_report_uses_unambiguous_whole_scene_fields():
    scenes, frames, qa = canonical()
    template = scenes[0]["nodes"][0]
    scenes[0]["nodes"] = []
    for index in range(385):
        node = copy.deepcopy(template)
        node["object_id"] = f"adt:capacity:{index}"
        scenes[0]["nodes"].append(node)
    frames[0]["visible_nodes"][0]["object_id"] = "adt:capacity:0"
    for row in frames[1:]:
        row["visible_nodes"][0]["object_id"] = "adt:capacity:0"
    qa[0]["actual_visible_object_ids"] = ["adt:capacity:0"]
    report = validate_records(scenes, frames, qa)
    assert report.scene_capacity_overflow_scenes == 1
    assert report.scene_capacity_excess_objects == 1
    assert report.scene_capacity_scope == "whole_scene_nodes_vs_k384"


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "low"),
        (0.499999, "low"),
        (0.5, "medium"),
        (0.749999, "medium"),
        (0.75, "high"),
        (1.0, "high"),
    ],
)
def test_coverage_bin_frozen_boundaries(ratio, expected):
    assert coverage_bin(ratio) == expected


@pytest.mark.parametrize(
    ("clip_end", "expected_ratio", "expected_bin"),
    [(500, 0.5, "medium"), (750, 0.75, "high")],
)
def test_duration_coverage_exact_boundaries(
    clip_end, expected_ratio, expected_bin
):
    provenance = {
        "whole_video_start_device_timestamp_ns": 0,
        "whole_video_end_device_timestamp_ns": 1000,
        "clip_start_device_timestamp_ns": 0,
        "clip_end_device_timestamp_ns": clip_end,
    }
    ratio = duration_coverage_ratio(provenance)
    assert ratio == expected_ratio
    assert coverage_bin(ratio) == expected_bin


def test_unlocalized_cannot_masquerade_as_verified():
    _, _, qa = canonical()
    row = qa[0]
    row["qa_visual_support_verified"] = True
    with pytest.raises(
        ContractError, match="scene_associated_unlocalized|Unlocalized"
    ):
        validate_qa_evidence_contract(row)


def test_adt_adapter_rejects_frame_verified_even_with_valid_subset():
    _, _, raw = raw_records()
    raw.update({
        "qa_evidence_scope": "frame_verified",
        "qa_visual_support_verified": True,
        "evidence_frame_indices": [0],
    })
    with pytest.raises(ContractError, match="ADT source QA"):
        adapt_qa("adt", raw)


@pytest.mark.parametrize("evidence", [[], [999999]])
def test_frame_verified_evidence_must_be_nonempty_actual_subset(evidence):
    _, _, qa = canonical()
    row = qa[0]
    row.update({
        "source_dataset": "hypersim",
        "qa_evidence_scope": "frame_verified",
        "qa_visual_support_verified": True,
        "evidence_frame_indices": evidence,
    })
    with pytest.raises(ContractError, match="nonempty|actual-frame subset"):
        validate_qa_evidence_contract(row)


def test_question_text_does_not_change_selection_or_binding():
    _, _, qa = canonical()
    original = copy.deepcopy(qa[0])
    changed = copy.deepcopy(original)
    changed["conversations"] = [{"from": "human", "value": "different"}]
    assert frame_binding_sha256(original) == frame_binding_sha256(changed)
    assert (
        original["source_sampling_binding_sha256"]
        == changed["source_sampling_binding_sha256"]
    )


def test_reference_error_fails():
    scenes, frames, qa = canonical()
    qa[0]["actual_frame_keys"][0] = "missing"
    rehash_canonical_qa(qa[0])
    with pytest.raises(ContractError, match="missing frame"):
        validate_records(scenes, frames, qa)


def test_declared_visible_set_must_match_actual_frames():
    scenes, frames, qa = canonical()
    qa[0]["actual_visible_object_ids"] = ["adt:not-seven"]
    with pytest.raises(ContractError, match="visible GT mismatch"):
        validate_records(scenes, frames, qa)


def test_unseen_gt_requires_explicit_empty_gt():
    scenes, frames, qa = canonical()
    for frame in frames:
        frame["visible_nodes"] = []
    qa[0]["empty_gt"] = False
    with pytest.raises(ContractError, match="No actual-input-visible"):
        validate_records(scenes, frames, qa)


def test_bad_video_frame_count_fails():
    scenes, frames, qa = canonical()
    qa[0]["actual_frame_keys"] = qa[0]["actual_frame_keys"][:8]
    qa[0]["actual_frame_indices"] = qa[0]["actual_frame_indices"][:8]
    rehash_canonical_qa(qa[0])
    with pytest.raises(ContractError, match="non-GUIDE raw frame IDs"):
        validate_records(scenes, frames, qa)


def test_fixed_fixture_missing_fails():
    scenes, frames, qa = canonical()
    assert len(T0_FIXTURES["adt"]) == 3
    with pytest.raises(ContractError, match="Missing fixed T0"):
        validate_records(scenes, frames, qa, require_fixtures=True)


def test_sampling_is_deterministic_and_bounded():
    first = guide_frame_indices(900, 30.0)
    second = guide_frame_indices(900, 30.0)
    assert first == second
    assert len(first) == 30
    assert first[0] == 0 and first[-1] == 899


def test_guide_exact_sampling_contract_examples():
    assert guide_frame_indices(480, 30.0) == [
        0, 31, 63, 95, 127, 159, 191, 223,
        255, 287, 319, 351, 383, 415, 447, 479,
    ]
    assert len(guide_frame_indices(100, 30.0)) == 16
    long = guide_frame_indices(3000, 30.0)
    assert len(long) == 32
    assert long[0] == 0 and long[-1] == 2999


def test_exact_raw_ids_fail_without_substitution():
    timestamps = list(range(0, 480_000_000, 1_000_000))
    selected = guide_frame_indices(480, 30.0)
    trajectory = list(timestamps)
    calibration = list(timestamps)
    trajectory.remove(timestamps[selected[5]])
    with pytest.raises(ValueError, match=str(selected[5])):
        FINALIZER.validate_exact_indices(
            timestamps,
            selected,
            trajectory,
            calibration,
            max_trajectory_error_ns=100_000,
            max_calibration_error_ns=100_000,
        )
    # The neighboring frame would pass but must never replace the selected ID.
    assert selected[5] - 1 in range(len(timestamps))


def test_exact_raw_ids_preserve_order_and_identity():
    timestamps = list(range(0, 480_000_000, 1_000_000))
    selected = guide_frame_indices(480, 30.0)
    diagnostics = FINALIZER.validate_exact_indices(
        timestamps,
        selected,
        timestamps,
        timestamps,
        max_trajectory_error_ns=5_000_000,
        max_calibration_error_ns=50_000_000,
    )
    assert [row["frame_index"] for row in diagnostics] == selected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_policy", None),
        ("sampling_policy", "legacy_valid_frame_linspace_v1"),
        ("sampling_policy", "unknown_policy"),
        ("actual_frame_indices", list(range(1, 17))),
        ("source_sampling_binding_sha256", "0" * 64),
        ("video_total_frames", 17),
        ("video_fps", 2.0),
        ("sampling_base_interval", 2.0),
        ("sampling_min_frames", 15),
        ("sampling_max_frames", 31),
    ],
)
def test_adt_sampling_contract_tampering_fails(field, value):
    scenes, frames, qa = canonical()
    qa[0][field] = value
    with pytest.raises(
        ContractError,
        match="sampling_policy|sampling binding",
    ):
        validate_records(scenes, frames, qa)


def test_tampered_final_frame_binding_fails():
    scenes, frames, qa = canonical()
    qa[0]["frame_binding_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="final frame binding SHA256"):
        validate_records(scenes, frames, qa)


def test_rehashed_non_guide_binding_still_fails():
    scenes, frames, qa = canonical()
    qa[0]["actual_frame_indices"][0:2] = reversed(
        qa[0]["actual_frame_indices"][0:2]
    )
    qa[0]["actual_frame_keys"][0:2] = reversed(
        qa[0]["actual_frame_keys"][0:2]
    )
    rehash_canonical_qa(qa[0])
    with pytest.raises(ContractError, match="non-GUIDE raw frame IDs"):
        validate_records(scenes, frames, qa)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
                lambda p: p.__setitem__("clip_start_raw_frame", 1),
                "frame count|too short|tie policy|clip-local",
        ),
        (
            lambda p: p.__setitem__(
                "clip_start_device_timestamp_ns", 123
            ),
            "timestamp/run mismatch",
        ),
        (
            lambda p: p["support_runs"][0].__setitem__("frame_count", 15),
            "support-run frame count",
        ),
        (
            lambda p: p.__setitem__("max_trajectory_error_ns", 5_000_001),
                "certificate/provenance mismatch|thresholds differ",
        ),
        (
            lambda p: p["local_frame_indices"].__setitem__(1, 2),
            "clip-local GUIDE positions",
        ),
    ],
)
def test_rehashed_d59_clip_provenance_tampering_fails(mutation, match):
    scenes, frames, qa = canonical()
    mutation(qa[0]["clip_provenance"])
    rehash_canonical_qa(qa[0])
    with pytest.raises(ContractError, match=match):
        validate_records(scenes, frames, qa)


def test_mask_value_mismatch_fails():
    scenes, frames, qa = canonical()
    broken = copy.deepcopy(scenes)
    broken[0]["nodes"][0]["field_mask"]["center"] = False
    with pytest.raises(ContractError, match="Mask/value mismatch"):
        validate_records(broken, frames, qa)


def test_object_ids_are_namespaced_by_scene():
    scene_a, _, _ = raw_records("scene_a")
    scene_b, _, _ = raw_records("scene_b")
    a = adapt_scene("adt", scene_a)["nodes"][0]["object_id"]
    b = adapt_scene("adt", scene_b)["nodes"][0]["object_id"]
    assert a == "adt:scene_a:7"
    assert b == "adt:scene_b:7"
    assert a != b


def test_exact_key_index_mismatch_fails():
    scenes, frames, qa = canonical()
    frames[3]["frame_index"] = 999
    with pytest.raises(ContractError, match="key/index mismatch"):
        validate_records(scenes, frames, qa)


def test_hypersim_geometry_invalid_is_not_supervision():
    scene = adapt_scene("hypersim", {
        "schema_version": "hypersim_scene_state_v1",
        "scene_id": "ai_001_001",
        "nodes": [{
            "object_id": 1,
            "object_name": "chair",
            "bbox_center_m": [1, 2, 3],
            "bbox_extent_m": [1, 1, 1],
            "bbox_orientation_raw": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }],
    })
    frame = adapt_frame("hypersim", {
        "schema_version": "hypersim_frame_state_v1",
        "scene_id": "ai_001_001",
        "frame_key": "ai_001_001/cam_00/0001",
        "frame_id": 1,
        "vsi_media": "hypersim/a.png",
        "rotation_world_from_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "translation_world_from_camera_m": [0, 0, 0],
        "visible_nodes": [{
            "object_id": 1,
            "pixel_count": 100,
            "geometry_valid": False,
            "center_camera_m": [1, 2, 3],
            "camera_distance_m": 4,
        }],
    })
    qa = adapt_qa("hypersim", {
        "schema_version": "hypersim_qa_train_v1",
        "vsi_row_index": 1,
        "scene_id": "ai_001_001",
        "frame_key": frame["frame_key"],
        "frame_index": 1,
        "vsi_media": "hypersim/a.png",
        "question_type": "x",
        "conversations": [],
        "loss_masks": {"scene_geometry": True},
    })
    lookup = {("hypersim", frame["frame_key"]): frame}
    manifest = list(build_manifest_rows([qa], lookup))
    assert frame["visible_nodes"][0]["visible"] is False
    assert frame["visible_nodes"][0]["field_mask"]["visibility"] is False
    assert manifest[0]["actual_visible_object_ids"] == []
    assert manifest[0]["empty_gt"] is True
    validate_records([scene], [frame], manifest)


def test_adt_direct_visibility_does_not_require_hypersim_pixel_fields():
    _, raw_frames, _ = raw_records()
    observation = raw_frames[0]["visible_nodes"][0]
    assert "pixel_count" not in observation
    assert "geometry_valid" not in observation
    frame = adapt_frame("adt", raw_frames[0])
    adapted = frame["visible_nodes"][0]
    assert adapted["evidence_present"] is True
    assert adapted["visible"] is True
    assert adapted["field_mask"]["visibility"] is True


def test_adt_explicit_invisible_evidence_is_preserved_but_not_supervised():
    _, raw_frames, _ = raw_records()
    raw_frames[0]["visible_nodes"][0]["visible"] = False
    frame = adapt_frame("adt", raw_frames[0])
    adapted = frame["visible_nodes"][0]
    assert adapted["evidence_present"] is True
    assert adapted["visible"] is False
    assert adapted["field_mask"]["visibility"] is False


def test_adt_visibility_source_count_mismatch_fails_closed():
    scenes, frames, qa = canonical()
    with pytest.raises(ContractError, match="observation mismatch"):
        validate_records(
            scenes,
            frames,
            qa,
            expected_visible_observations={"adt": 17},
        )


def test_adt_evidence_with_empty_canonical_visibility_fails():
    scenes, frames, qa = canonical()
    for frame in frames:
        for observation in frame["visible_nodes"]:
            observation["visible"] = False
            observation["field_mask"]["visibility"] = False
    qa = list(build_manifest_rows(
        qa,
        {("adt", frame["frame_key"]): frame for frame in frames},
    ))
    with pytest.raises(ContractError, match="canonical visibility is empty"):
        validate_records(scenes, frames, qa)


def test_hypersim_visibility_requires_geometry_and_pixel_count():
    assert source_visibility_contract(
        "hypersim",
        {"geometry_valid": True, "pixel_count": 16},
    ) == (True, True)
    assert source_visibility_contract(
        "hypersim",
        {"geometry_valid": True, "pixel_count": 15},
    ) == (True, False)
    with pytest.raises(ContractError, match="numeric pixel_count"):
        source_visibility_contract(
            "hypersim",
            {"geometry_valid": True},
        )


def test_coordinate_transform_and_fixed_category_policy():
    scene, _, _ = raw_records()
    scene["nodes"][0]["center_world_m"] = [1, 2, 3]
    scene["nodes"][0]["velocity_world_mps"] = [0, 1, 0]
    adapted = adapt_scene("adt", scene)
    assert adapted["coordinate_frame"].endswith("xright_yup_zback_m_v1")
    assert adapted["nodes"][0]["center_world_m"] == [1.0, 3.0, -2.0]
    assert adapted["nodes"][0]["velocity_world_mps"] == [0.0, 0.0, -1.0]
    assert canonical_category("couch")[0] == "sofa"
    assert canonical_category("unreleased-custom-prototype")[0] == "__unknown__"
    hypersim = copy.deepcopy(scene)
    hypersim["schema_version"] = "hypersim_scene_state_v1"
    assert adapt_scene("hypersim", hypersim)["nodes"][0][
        "velocity_world_mps"
    ] == [0.0, 1.0, 0.0]


def test_expected_sources_are_hard_required():
    scenes, frames, qa = canonical()
    with pytest.raises(ContractError, match="Missing expected sources"):
        validate_records(
            scenes,
            frames,
            qa,
            expected_sources=["adt", "hypersim"],
        )


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_adt_anchor(source_dir: Path, certificate):
    certificate_path = source_dir / "adt_support_certificates.jsonl"
    _write_jsonl(certificate_path, [certificate])
    digest = hashlib.sha256(certificate_path.read_bytes()).hexdigest()
    (source_dir / "adt_alignment_report.json").write_text(json.dumps({
        "support_certificate_registry": {
            "path": certificate_path.name,
            "sha256": digest,
            "trust_stage": "finalizer_output_v1",
        }
    }))


def test_canonical_cli_rejects_adt_frame_verified_attack(tmp_path):
    scene, frames, qa = raw_records()
    qa.update({
        "qa_evidence_scope": "frame_verified",
        "qa_visual_support_verified": True,
        "evidence_frame_indices": [0],
    })
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "canonical"
    source_dir.mkdir()
    _write_jsonl(source_dir / "adt_scene_states.jsonl", [scene])
    _write_jsonl(source_dir / "adt_frame_states.jsonl", frames)
    _write_jsonl(source_dir / "adt_qa_train.jsonl", [qa])
    _write_adt_anchor(
        source_dir, qa["clip_provenance"]["support_certificate"]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/preprocess/build_parta_canonical_data.py"),
            "--source", "adt",
            "--input-dir", str(source_dir),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ADT source QA" in result.stderr


def test_matched_arm_assertion_passes_and_fails_on_mismatch(tmp_path):
    scenes, frames, qa = canonical()
    second = copy.deepcopy(qa[0])
    second["qa_id"] = "adt:10"
    second["vsi_row_index"] = 10
    second["conversations"] = [
        {"from": "human", "value": "second question"},
        {"from": "gpt", "value": "second answer"},
    ]
    qa.append(second)
    a0 = tmp_path / "a0.jsonl"
    a1 = tmp_path / "a1.jsonl"
    scene_path = tmp_path / "scenes.jsonl"
    frame_path = tmp_path / "frames.jsonl"
    report = tmp_path / "matched.json"
    _write_jsonl(scene_path, scenes)
    _write_jsonl(frame_path, frames)
    _write_jsonl(a0, qa)
    _write_jsonl(a1, copy.deepcopy(qa))
    command = [
        sys.executable,
        str(PROJECT / "scripts/preprocess/assert_parta_matched_arms.py"),
        "--a0-manifest", str(a0),
        "--a1-manifest", str(a1),
        "--scenes", str(scene_path),
        "--frames", str(frame_path),
        "--output", str(report),
    ]
    subprocess.run(command, check=True)
    payload = json.loads(report.read_text())
    assert payload["status"] == "pass"
    assert payload["qa_loss"]["a0"]["status"] == "not_provided"
    assert len(payload["artifacts"]["a0"]["file_sha256"]) == 64

    mismatched = copy.deepcopy(qa)
    mismatched[0]["coverage_bin"] = "medium"
    _write_jsonl(a1, mismatched)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "coverage bin" in result.stderr

    answer_changed = copy.deepcopy(qa)
    answer_changed[0]["conversations"] = [
        {"from": "gpt", "value": "tampered answer"}
    ]
    _write_jsonl(a1, answer_changed)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "causal QA payload differs" in result.stderr

    _write_jsonl(a1, list(reversed(qa)))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "ordered qa_id sequence differs" in result.stderr


def test_atomic_archive_failure_leaves_no_formal_product(tmp_path, monkeypatch):
    output = tmp_path / "formal.tar.gz"
    output.write_bytes(b"stale")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n")

    def fail_open(*args, **kwargs):
        raise RuntimeError("injected tar failure")

    monkeypatch.setattr(FINALIZER.tarfile, "open", fail_open)
    with pytest.raises(RuntimeError, match="injected"):
        FINALIZER.write_archive_atomic(output, [source])
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_dynamic_object_pose_gap_masks_geometry_but_static_does_not():
    dynamic = {"motion_type": "dynamic"}
    static = {"motion_type": "static"}
    assert not FINALIZER.object_geometry_is_valid(dynamic, 5_000_001, 5_000_000)
    assert FINALIZER.object_geometry_is_valid(dynamic, 5_000_000, 5_000_000)
    assert FINALIZER.object_geometry_is_valid(static, 999_000_000, 5_000_000)


def test_dynamic_scene_node_pose_gap_keeps_identity_but_masks_geometry():
    raw = {
        "schema_version": "adt_scene_state_v1",
        "scene_id": "dynamic_scene",
        "nodes": [{
            "object_id": "tracked-7",
            "category": "chair",
            "motion_type": "dynamic",
            "geometry_valid": False,
            "reference_pose_timestamp_error_ns": 5_000_001,
            "center_world_m": None,
            "extent_m": None,
            "rotation_world_from_object": None,
        }],
    }
    scene = adapt_scene("adt", raw)
    node = scene["nodes"][0]
    assert node["source_object_id"] == "tracked-7"
    assert node["object_id"].endswith("tracked-7")
    assert node["center_world_m"] is None
    assert node["extent_m"] is None
    assert node["rotation_world_from_object"] is None
    assert node["field_mask"]["center"] is False
    assert node["field_mask"]["extent"] is False
    assert node["field_mask"]["orientation"] is False


def test_canonical_and_exact_manifest_cli_e2e(tmp_path):
    raw_scene, raw_frames, raw_qa = raw_records()
    exact_indices = guide_frame_indices(480, 30)
    raw_frames = [
        {
            **raw_frames[position],
            "frame_key": f"{raw_scene['scene_id']}/{raw_index}",
            "frame_index": raw_index,
            "device_timestamp_ns": raw_index * 1_000_000,
        }
        for position, raw_index in enumerate(exact_indices)
    ]
    raw_qa["candidate_frame_keys"] = [
        row["frame_key"] for row in raw_frames
    ]
    raw_qa["candidate_frame_indices"] = exact_indices
    raw_qa["total_frames"] = 480
    raw_qa["fps"] = 30.0
    raw_qa["clip_provenance"] = {
        **raw_qa["clip_provenance"],
        "whole_video_total_frames": 480,
        "whole_video_fps_hex": float(30.0).hex(),
        "whole_video_end_device_timestamp_ns": 479_000_000,
        "clip_start_raw_frame": 0,
        "clip_end_raw_frame": 479,
        "clip_end_device_timestamp_ns": 479_000_000,
        "clip_frame_count": 480,
        "support_runs": [{
            "start_raw_frame": 0,
            "end_raw_frame": 479,
            "frame_count": 480,
            "start_device_timestamp_ns": 0,
            "end_device_timestamp_ns": 479_000_000,
        }],
        "local_frame_indices": exact_indices,
        "selected_device_timestamps_ns": [
            index * 1_000_000 for index in exact_indices
        ],
        "support_certificate": build_support_certificate(
            scene_id=raw_scene["scene_id"],
            vsi_media=raw_qa["vsi_media"],
            frame_timestamps=[
                index * 1_000_000 for index in range(480)
            ],
            fps=30.0,
            support_mask=[True] * 480,
            max_trajectory_error_ns=5_000_000,
            max_calibration_error_ns=50_000_000,
        ),
    }
    raw_qa["sampling_binding_sha256"] = guide_sampling_binding_sha256(
        source_dataset="adt",
        scene_id=raw_qa["scene_id"],
        vsi_media=raw_qa["vsi_media"],
        frame_keys=raw_qa["candidate_frame_keys"],
        frame_indices=raw_qa["candidate_frame_indices"],
        total_frames=raw_qa["total_frames"],
        fps=raw_qa["fps"],
        base_interval=raw_qa["base_interval"],
        min_frames=raw_qa["min_frames"],
        max_frames=raw_qa["max_frames"],
        sampling_policy=raw_qa["sampling_policy"],
        clip_provenance=raw_qa["clip_provenance"],
    )
    source_dir = tmp_path / "source"
    canonical_dir = tmp_path / "canonical"
    source_dir.mkdir()
    _write_jsonl(source_dir / "adt_scene_states.jsonl", [raw_scene])
    _write_jsonl(source_dir / "adt_frame_states.jsonl", raw_frames)
    _write_jsonl(source_dir / "adt_qa_train.jsonl", [raw_qa])
    _write_adt_anchor(
        source_dir, raw_qa["clip_provenance"]["support_certificate"]
    )
    project = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_canonical_data.py"),
            "--source", "adt",
            "--input-dir", str(source_dir),
            "--output-dir", str(canonical_dir),
            "--expected-source", "adt",
        ],
        check=True,
    )
    metadata = tmp_path / "metadata.jsonl"
    _write_jsonl(metadata, [{
        "source_dataset": "adt",
        "scene_id": raw_scene["scene_id"],
        "vsi_media": "adt/a.mp4",
        "total_frames": 480,
        "fps": 30,
    }])
    output = tmp_path / "exact.jsonl"
    report = tmp_path / "exact_report.json"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_exact_frame_manifest.py"),
            "--scenes", str(canonical_dir / "scene_states.jsonl"),
            "--frames", str(canonical_dir / "frame_states.jsonl"),
            "--qa", str(canonical_dir / "qa_manifest.jsonl"),
            "--video-metadata", str(metadata),
            "--output", str(output),
            "--report-output", str(report),
            "--expected-source", "adt",
        ],
        check=True,
    )
    row = json.loads(output.read_text().splitlines()[0])
    assert len(row["actual_frame_keys"]) == 16
    assert row["actual_frame_indices"] == exact_indices
    assert len(row["frame_binding_sha256"]) == 64
    assert json.loads(report.read_text())["qa"] == 1

    second_output = tmp_path / "exact_second.jsonl"
    second_report = tmp_path / "exact_second_report.json"
    subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_exact_frame_manifest.py"),
            "--scenes", str(canonical_dir / "scene_states.jsonl"),
            "--frames", str(canonical_dir / "frame_states.jsonl"),
            "--qa", str(output),
            "--video-metadata", str(metadata),
            "--output", str(second_output),
            "--report-output", str(second_report),
            "--expected-source", "adt",
        ],
        check=True,
    )
    assert second_output.read_bytes() == output.read_bytes()

    # Attack: rewrite the support bitset, all row/certificate hashes, and try
    # to supply the attacker's matching digest at invocation. The trusted
    # digest is anchored in canonical validation_report.json; exact has no
    # caller-controlled digest option, so this must fail closed.
    anchored_certificate = canonical_dir / "adt_support_certificates.jsonl"
    attacked_certificate = copy.deepcopy(
        raw_qa["clip_provenance"]["support_certificate"]
    )
    attacked_certificate["support_mask_bitset_hex"] = (
        "fe" + attacked_certificate["support_mask_bitset_hex"][2:]
    )
    from src.adt_gt_supported_clip import support_certificate_sha256
    attacked_certificate["certificate_sha256"] = (
        support_certificate_sha256(attacked_certificate)
    )
    _write_jsonl(anchored_certificate, [attacked_certificate])
    attacked_row = copy.deepcopy(row)
    attacked_row["clip_provenance"]["support_certificate"] = (
        attacked_certificate
    )
    rehash_canonical_qa(attacked_row)
    attacked_qa = tmp_path / "attacked_qa.jsonl"
    _write_jsonl(attacked_qa, [attacked_row])
    attacker_digest = hashlib.sha256(
        anchored_certificate.read_bytes()
    ).hexdigest()
    attack_result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_exact_frame_manifest.py"),
            "--scenes", str(canonical_dir / "scene_states.jsonl"),
            "--frames", str(canonical_dir / "frame_states.jsonl"),
            "--qa", str(attacked_qa),
            "--video-metadata", str(metadata),
            "--output", str(tmp_path / "attacked_out.jsonl"),
            "--report-output", str(tmp_path / "attacked_report.json"),
            "--support-certificates-sha256", attacker_digest,
        ],
        capture_output=True,
        text=True,
    )
    assert attack_result.returncode != 0
    assert (
        "unrecognized arguments" in attack_result.stderr
        or "anchor SHA256 mismatch" in attack_result.stderr
    )
    # Restore the trusted certificate for the remaining tamper checks.
    _write_jsonl(
        anchored_certificate,
        [raw_qa["clip_provenance"]["support_certificate"]],
    )

    for field, value, expected_errors in [
        (
                "video_total_frames",
                481,
                (
                    "D-59 whole-video total-frame mismatch",
                    "total-frame provenance mismatch",
                ),
        ),
        (
                "video_fps",
                29.0,
                ("D-59 whole-video FPS mismatch", "FPS provenance mismatch"),
        ),
        ("sampling_base_interval", 2.0, ("base_interval mismatch",)),
        ("sampling_min_frames", 15, ("min_frames mismatch",)),
        ("sampling_max_frames", 31, ("max_frames mismatch",)),
    ]:
        tampered = copy.deepcopy(row)
        tampered[field] = value
        rehash_canonical_qa(tampered)
        tampered_path = tmp_path / f"tampered_{field}.jsonl"
        _write_jsonl(tampered_path, [tampered])
        result = subprocess.run(
            [
                sys.executable,
                str(
                    project
                    / "scripts/preprocess/build_parta_exact_frame_manifest.py"
                ),
                "--scenes", str(canonical_dir / "scene_states.jsonl"),
                "--frames", str(canonical_dir / "frame_states.jsonl"),
                "--qa", str(tampered_path),
                "--video-metadata", str(metadata),
                "--output", str(tmp_path / f"out_{field}.jsonl"),
                "--report-output", str(tmp_path / f"report_{field}.json"),
                "--expected-source", "adt",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert any(error in result.stderr for error in expected_errors)

    reordered = copy.deepcopy(row)
    reordered["actual_frame_indices"][0:2] = reversed(
        reordered["actual_frame_indices"][0:2]
    )
    reordered["actual_frame_keys"][0:2] = reversed(
        reordered["actual_frame_keys"][0:2]
    )
    rehash_canonical_qa(reordered)
    reordered_path = tmp_path / "tampered_reordered_ids.jsonl"
    _write_jsonl(reordered_path, [reordered])
    result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_exact_frame_manifest.py"),
            "--scenes", str(canonical_dir / "scene_states.jsonl"),
            "--frames", str(canonical_dir / "frame_states.jsonl"),
            "--qa", str(reordered_path),
            "--video-metadata", str(metadata),
            "--output", str(tmp_path / "out_reordered.jsonl"),
            "--report-output", str(tmp_path / "report_reordered.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "non-GUIDE raw frame IDs" in result.stderr
        or "differ from recomputed GUIDE IDs" in result.stderr
    )


def test_exact_manifest_cli_missing_raw_frame_state_fails(tmp_path):
    raw_scene, raw_frames, raw_qa = raw_records()
    raw_qa["total_frames"] = 480
    raw_qa["fps"] = 30.0
    raw_qa["clip_provenance"] = {
        **raw_qa["clip_provenance"],
        "whole_video_total_frames": 480,
        "whole_video_fps_hex": float(30.0).hex(),
        "whole_video_end_device_timestamp_ns": 479_000_000,
        "clip_start_raw_frame": 0,
        "clip_end_raw_frame": 479,
        "clip_end_device_timestamp_ns": 479_000_000,
        "clip_frame_count": 480,
        "support_runs": [{
            "start_raw_frame": 0,
            "end_raw_frame": 479,
            "frame_count": 480,
            "start_device_timestamp_ns": 0,
            "end_device_timestamp_ns": 479_000_000,
        }],
        "local_frame_indices": guide_frame_indices(480, 30.0),
        "selected_device_timestamps_ns": [
            index * 1_000_000
            for index in guide_frame_indices(480, 30.0)
        ],
        "support_certificate": build_support_certificate(
            scene_id=raw_scene["scene_id"],
            vsi_media=raw_qa["vsi_media"],
            frame_timestamps=[
                index * 1_000_000 for index in range(480)
            ],
            fps=30.0,
            support_mask=[True] * 480,
            max_trajectory_error_ns=5_000_000,
            max_calibration_error_ns=50_000_000,
        ),
    }
    raw_qa["candidate_frame_indices"] = guide_frame_indices(480, 30.0)
    raw_qa["candidate_frame_keys"] = [
        f"{raw_scene['scene_id']}/{index}"
        for index in raw_qa["candidate_frame_indices"]
    ]
    raw_qa["sampling_binding_sha256"] = guide_sampling_binding_sha256(
        source_dataset="adt",
        scene_id=raw_qa["scene_id"],
        vsi_media=raw_qa["vsi_media"],
        frame_keys=raw_qa["candidate_frame_keys"],
        frame_indices=raw_qa["candidate_frame_indices"],
        total_frames=480,
        fps=30.0,
        base_interval=1.0,
        min_frames=16,
        max_frames=32,
        sampling_policy=raw_qa["sampling_policy"],
        clip_provenance=raw_qa["clip_provenance"],
    )
    source_dir = tmp_path / "source"
    canonical_dir = tmp_path / "canonical"
    source_dir.mkdir()
    _write_jsonl(source_dir / "adt_scene_states.jsonl", [raw_scene])
    _write_jsonl(source_dir / "adt_frame_states.jsonl", raw_frames)
    _write_jsonl(source_dir / "adt_qa_train.jsonl", [raw_qa])
    _write_adt_anchor(
        source_dir, raw_qa["clip_provenance"]["support_certificate"]
    )
    project = Path(__file__).resolve().parents[1]
    canonical_result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/preprocess/build_parta_canonical_data.py"),
            "--source", "adt",
            "--input-dir", str(source_dir),
            "--output-dir", str(canonical_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert canonical_result.returncode != 0
    assert (
        "missing frame" in canonical_result.stderr.lower()
        or "KeyError" in canonical_result.stderr
    )
