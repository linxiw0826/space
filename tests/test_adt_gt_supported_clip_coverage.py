import csv
import importlib.util
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT
    / "scripts/preprocess/audit_adt_gt_supported_clip_coverage.py"
)
SPEC = importlib.util.spec_from_file_location("adt_clip_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def timestamps(count):
    return [index * 1_000_000_000 for index in range(count)]


def analyze(
    count=40,
    trajectory=None,
    calibration=None,
    fps=1.0,
    qa_count=7,
):
    frames = timestamps(count)
    return AUDIT.analyze_arrays(
        scene_id="scene",
        frame_timestamps=frames,
        fps=fps,
        trajectory_timestamps=(
            frames if trajectory is None else trajectory
        ),
        calibration_timestamps=(
            frames if calibration is None else calibration
        ),
        qa_count=qa_count,
        max_trajectory_error_ns=1,
        max_calibration_error_ns=1,
    )


def test_head_and_tail_without_gt_are_cropped():
    row = analyze(40, trajectory=timestamps(36)[3:36])
    assert row["clip_start_raw_frame"] == 3
    assert row["clip_end_raw_frame"] == 35
    assert row["selected_raw_frame_ids"][0] == 3
    assert row["selected_raw_frame_ids"][-1] == 35


def test_internal_gap_uses_longest_run():
    frames = timestamps(50)
    trajectory = frames[:20] + frames[25:]
    row = analyze(50, trajectory=trajectory)
    assert row["all_run_count"] == 2
    assert (row["clip_start_raw_frame"], row["clip_end_raw_frame"]) == (25, 49)


def test_vfr_frame_and_duration_coverage_are_distinct():
    frames = [0, 1, 2, 3, 100, 200, 300, 400, 500, 600,
              700, 800, 900, 1000, 1100, 1200, 1300, 1400,
              1500, 1600, 1700, 1800, 1900, 2000]
    trajectory = frames[4:]
    row = AUDIT.analyze_arrays(
        scene_id="vfr",
        frame_timestamps=frames,
        fps=16.0,
        trajectory_timestamps=trajectory,
        calibration_timestamps=frames,
        qa_count=1,
        max_trajectory_error_ns=0,
        max_calibration_error_ns=0,
    )
    assert row["frame_coverage_ratio"] == 20 / 24
    assert row["duration_coverage_ratio"] == pytest.approx(
        (2000 - 100) / 2000
    )
    assert row["frame_coverage_ratio"] != row["duration_coverage_ratio"]


def test_equal_length_tie_breaks_to_earliest_run():
    frames = timestamps(40)
    trajectory = frames[:18] + frames[22:]
    row = analyze(40, trajectory=trajectory)
    assert row["all_run_lengths"] == [18, 18]
    assert (row["clip_start_raw_frame"], row["clip_end_raw_frame"]) == (0, 17)


def test_run_shorter_than_sixteen_rejected():
    with pytest.raises(ValueError, match="required exact frame count"):
        analyze(15)


def test_guide_sixteen_and_thirty_two_boundaries():
    short = analyze(16)
    long = analyze(100, fps=1.0)
    assert short["selected_frame_count"] == 16
    assert short["selected_raw_frame_ids"] == list(range(16))
    assert long["selected_frame_count"] == 32
    assert long["selected_raw_frame_ids"][0] == 0
    assert long["selected_raw_frame_ids"][-1] == 99


def test_selected_ids_are_all_supported_without_substitution():
    frames = timestamps(40)
    trajectory = frames[3:37]
    row = analyze(40, trajectory=trajectory)
    assert row["trajectory_failures_within_selected"] == 0
    assert row["calibration_failures_within_selected"] == 0
    assert all(3 <= value <= 36 for value in row["selected_raw_frame_ids"])


def test_qa_count_and_summary_retention():
    usable = analyze(16, qa_count=11)
    failed = AUDIT.failed_row("bad", 5, ValueError("missing"))
    summary = AUDIT.build_summary(
        [usable, failed], requested=2, fixture_ids=[]
    )
    assert usable["qa_count"] == 11
    assert summary["requested_qa_rows"] == 16
    assert summary["usable_qa_rows"] == 11
    assert summary["qa_retention_rate"] == 11 / 16


def test_missing_files_are_recorded_per_scene():
    row = AUDIT.failed_row(
        "missing-scene", 3, FileNotFoundError("Missing video")
    )
    assert row["usable"] is False
    assert row["qa_count"] == 3
    assert "Missing video" in row["rejection_reason"]


def test_fixture_status_hard_fail_contract():
    row = analyze(16)
    row["scene_id"] = "fixture-a"
    summary = AUDIT.build_summary(
        [row], requested=1, fixture_ids=["fixture-a", "fixture-b"]
    )
    assert summary["t0_fixtures"]["fixture-a"]["usable"] is True
    assert summary["t0_fixtures"]["fixture-b"]["usable"] is False


def test_json_csv_scene_values_are_consistent(tmp_path):
    row = analyze(16)
    report = {"schema_version": AUDIT.SCHEMA_VERSION, "scenes": [row]}
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    AUDIT.atomic_write_text(json_path, json.dumps(report))
    AUDIT.write_csv(csv_path, [row])
    loaded = json.loads(json_path.read_text())["scenes"][0]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["scene_id"] == loaded["scene_id"]
    assert int(csv_row["selected_frame_count"]) == loaded["selected_frame_count"]
    assert json.loads(csv_row["selected_raw_frame_ids"]) == (
        loaded["selected_raw_frame_ids"]
    )


def test_empty_observation_is_not_a_hard_support_field(tmp_path):
    archive_path = tmp_path / "gt.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("aria_trajectory.csv", "tracking_timestamp_us\n")
        archive.writestr("instances.json", '{"7": {"category": "chair"}}')
        archive.writestr("3d_bounding_box.csv", "object_uid\n7\n")
        archive.writestr(
            "scene_objects.csv",
            "object_uid,timestamp[ns]\n7,-1\n8,100\n",
        )
        archive.writestr(
            "2d_bounding_box.csv",
            "stream_id,timestamp[ns]\n214-1,100\n1201-1,200\n",
        )
    with zipfile.ZipFile(archive_path) as archive:
        capabilities = AUDIT.inspect_direct_gt(archive)
    assert capabilities["per_frame_hard_support_fields"] == [
        "trajectory",
        "calibration",
    ]
    assert capabilities["rgb_box_annotation_timestamps"] == 1
    assert capabilities["dynamic_object_pose_timestamps"] == 1
    assert capabilities["joined_direct_object_count"] == 1
    assert capabilities["empty_rgb_box_policy"].startswith(
        "valid_empty_observation"
    )


def test_empty_direct_object_join_is_rejected(tmp_path):
    archive_path = tmp_path / "bad_gt.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("aria_trajectory.csv", "tracking_timestamp_us\n")
        archive.writestr("instances.json", "{}")
        archive.writestr("3d_bounding_box.csv", "object_uid\n")
        archive.writestr("scene_objects.csv", "object_uid,timestamp[ns]\n")
        archive.writestr(
            "2d_bounding_box.csv", "stream_id,timestamp[ns]\n"
        )
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(
            AUDIT.SceneAuditError, match="No object ID joins"
        ) as caught:
            AUDIT.inspect_direct_gt(archive)
    assert caught.value.code == "direct_gt_object_join_empty"


def test_file_artifact_has_streaming_hash_and_size(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    artifact = AUDIT.file_artifact(path)
    assert artifact["size_bytes"] == 3
    assert artifact["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def valid_args(tmp_path):
    jsonl = tmp_path / "vsi.jsonl"
    sequences = tmp_path / "sequences.txt"
    jsonl.write_text("")
    sequences.write_text("scene\n")
    groundtruth = tmp_path / "gt"
    calibration = tmp_path / "calibration"
    videos = tmp_path / "videos"
    groundtruth.mkdir()
    calibration.mkdir()
    videos.mkdir()
    return Namespace(
        jsonl=jsonl,
        sequences=sequences,
        groundtruth_root=groundtruth,
        calibration_root=calibration,
        video_root=videos,
        output_json=tmp_path / "report.json",
        output_csv=tmp_path / "report.csv",
        base_interval=1.0,
        min_frames=16,
        max_frames=32,
        max_trajectory_error_ns=5_000_000,
        max_calibration_error_ns=50_000_000,
        sequence_limit=None,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("base_interval", 0.0, "base-interval"),
        ("min_frames", 0, "min-frames"),
        ("max_trajectory_error_ns", -1, "trajectory"),
        ("max_calibration_error_ns", -1, "calibration"),
        ("sequence_limit", 0, "sequence-limit"),
    ],
)
def test_invalid_global_parameters_fail_fast(tmp_path, field, value, match):
    args = valid_args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        AUDIT.validate_global_args(args)


def test_output_paths_must_differ(tmp_path):
    args = valid_args(tmp_path)
    args.output_csv = args.output_json
    with pytest.raises(ValueError, match="must differ"):
        AUDIT.validate_global_args(args)


@pytest.mark.parametrize("protected_name", ["jsonl", "sequences"])
def test_output_cannot_overwrite_primary_input(tmp_path, protected_name):
    args = valid_args(tmp_path)
    args.output_json = getattr(args, protected_name)
    with pytest.raises(ValueError, match="collides with source"):
        AUDIT.validate_global_args(args)


@pytest.mark.parametrize(
    "root_name",
    ["video_root", "groundtruth_root", "calibration_root"],
)
def test_output_cannot_be_inside_source_root(tmp_path, root_name):
    args = valid_args(tmp_path)
    args.output_json = getattr(args, root_name) / "report.json"
    with pytest.raises(ValueError, match="protected source root"):
        AUDIT.validate_global_args(args)


@pytest.mark.parametrize("name", ["actual.zip", "actual.mp4"])
def test_output_cannot_equal_resolved_actual_artifact(tmp_path, name):
    artifact = tmp_path / name
    artifact.write_bytes(b"source")
    with pytest.raises(ValueError, match="collides with source"):
        AUDIT.validate_output_collisions(
            (artifact,),
            (artifact,),
            (),
        )


def test_failed_scene_artifacts_preserve_missing_and_existing(tmp_path):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not-an-mp4")
    missing_gt = tmp_path / "missing_gt.zip"
    calibration = tmp_path / "calibration.zip"
    calibration.write_bytes(b"calibration")
    artifacts = {
        "video": AUDIT.candidate_artifact(
            [video], expected_path=video
        ),
        "groundtruth": AUDIT.candidate_artifact(
            [], expected_path=missing_gt
        ),
        "calibration": AUDIT.candidate_artifact(
            [calibration], expected_path=calibration
        ),
    }
    row = AUDIT.failed_row(
        "bad", 2,
        AUDIT.SceneAuditError(
            "video_metadata_invalid", "bad MP4"
        ),
    )
    row["input_artifacts"] = artifacts
    summary = AUDIT.build_summary([row], 1, [])
    assert artifacts["video"]["status"] == "ok"
    assert artifacts["groundtruth"] == {
        "status": "missing",
        "path": str(missing_gt.resolve()),
        "size_bytes": None,
        "sha256": None,
    }
    assert artifacts["calibration"]["status"] == "ok"
    assert summary["source_artifact_status_counts"]["video"] == {"ok": 1}
    assert summary["source_artifact_status_counts"]["groundtruth"] == {
        "missing": 1
    }
    assert summary["scene_input_artifacts"]["bad"] == artifacts


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("video_metadata_invalid", "bad MP4"),
        ("direct_gt_object_join_empty", "empty join"),
        ("gt_supported_run_too_short", "fewer than 16"),
        ("missing_video", "missing file"),
    ],
)
def test_failed_scene_codes_keep_artifact_provenance(tmp_path, code, detail):
    source = tmp_path / f"{code}.bin"
    source.write_bytes(code.encode())
    row = AUDIT.failed_row(
        code, 1, AUDIT.SceneAuditError(code, detail)
    )
    row["input_artifacts"] = {
        "video": AUDIT.artifact_record(source),
        "groundtruth": AUDIT.artifact_record(tmp_path / "missing_gt"),
        "calibration": AUDIT.artifact_record(tmp_path / "missing_cal"),
    }
    assert row["rejection_code"] == code
    assert row["input_artifacts"]["video"]["sha256"] is not None
    assert row["input_artifacts"]["groundtruth"]["status"] == "missing"


def test_hash_error_has_stable_status_and_other_artifacts_survive(
    tmp_path, monkeypatch
):
    bad = tmp_path / "bad.bin"
    good = tmp_path / "good.bin"
    bad.write_bytes(b"bad")
    good.write_bytes(b"good")
    original = AUDIT.file_sha256

    def fail_selected(path):
        if path == bad:
            raise OSError("simulated hash failure")
        return original(path)

    monkeypatch.setattr(AUDIT, "file_sha256", fail_selected)
    bad_record = AUDIT.artifact_record(bad)
    good_record = AUDIT.artifact_record(good)
    assert bad_record["status"] == "hash_error"
    assert bad_record["sha256"] is None
    assert "simulated hash failure" in bad_record["error"]
    assert good_record["status"] == "ok"
    row = AUDIT.failed_row(
        "hash-bad", 1,
        AUDIT.SceneAuditError("artifact_hash_error", bad_record["error"]),
    )
    assert row["rejection_code"] == "artifact_hash_error"
