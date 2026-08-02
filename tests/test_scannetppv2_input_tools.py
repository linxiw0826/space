import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.scannetppv2_labels import normalize_scannetppv2_label

PROJECT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT / "scripts" / "preprocess" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = load_script("audit_scannetppv2_vsi_inputs")
PROBE = load_script("probe_vsi_video_metadata")
CONFIG = load_script("build_scannetppv2_download_config")
PILOT = load_script("audit_scannetppv2_pilot")
BUNDLE = load_script("build_scannetppv2_pilot_frame_bundle")
FULL = load_script("audit_scannetppv2_full_contract")


def test_minimal_download_assets_cover_geometry_identity_and_camera_metadata():
    assert CONFIG.MINIMAL_ASSETS == (
        "scan_mesh_path",
        "scan_mesh_segs_path",
        "scan_anno_json_path",
        "iphone_pose_intrinsic_imu_path",
        "iphone_exif_path",
    )
    assert "iphone_video_path" not in CONFIG.MINIMAL_ASSETS
    assert "iphone_depth_path" not in CONFIG.MINIMAL_ASSETS


def test_scannetpp_media_and_source_inventory(tmp_path: Path):
    jsonl = tmp_path / "vsi.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"video": "scannetppv2/scene_b.mp4"}),
                json.dumps({"video": "adt/sequence.mp4"}),
                json.dumps({"video": "scannetppv2/scene_a.mp4"}),
                json.dumps({"video": "scannetppv2/scene_a.mp4"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert PROBE.source_media(jsonl, "scannetppv2") == [
        "scannetppv2/scene_a.mp4",
        "scannetppv2/scene_b.mp4",
    ]
    assert AUDIT.parse_scannetpp_media("scannetppv2/scene_a.mp4") == "scene_a"
    assert AUDIT.parse_scannetpp_media("scannetppv2/nested/scene_a.mp4") is None


def test_resolve_media_rejects_escape_and_wrong_extension(tmp_path: Path):
    root = tmp_path / "media"
    root.mkdir()
    assert PROBE.resolve_media(
        root, "scannetppv2/scene.mp4", "scannetppv2"
    ) == root / "scannetppv2" / "scene.mp4"
    with pytest.raises(ValueError):
        PROBE.resolve_media(root, "scannetppv2/../secret.mp4", "scannetppv2")
    with pytest.raises(ValueError):
        PROBE.resolve_media(root, "scannetppv2/scene.mov", "scannetppv2")


def test_parse_rate_contract():
    assert PROBE.parse_rate("30000/1001") == pytest.approx(29.97002997)
    assert PROBE.parse_rate("0/0") is None
    assert PROBE.parse_rate(None) is None


def test_ply_vertex_count_and_label_aliases(tmp_path: Path):
    path = tmp_path / "mesh.ply"
    path.write_bytes(
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 12\n"
        b"property float x\n"
        b"element face 0\n"
        b"end_header\n"
    )
    assert PILOT.ply_vertex_count(path) == (12, "binary_little_endian")
    assert PILOT.normalized_label("ceiling lamp") == "ceiling light"
    assert PILOT.normalized_label("office chair") == "chair"
    assert PILOT.normalized_label("trash bin") == "trash can"
    assert PILOT.normalized_label("mouse") == "computer mouse"


def test_projection_metrics_prefers_matching_camera_pose():
    frames = [
        {
            "chair": {
                "inst_ids": np.asarray([0], dtype=np.int64),
                "inst_num_pixels": np.asarray([100], dtype=np.int64),
                "num_pixels": 100,
            }
        }
    ]
    poses = {
        "frame_000000": {
            "aligned_pose": np.eye(4).tolist(),
            "intrinsic": [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]],
        }
    }
    result = PILOT.projection_metrics(
        frames=frames,
        poses=poses,
        centers={0: np.asarray([0.0, 0.0, 2.0])},
        source_index=lambda index: index,
        width=10,
        height=10,
    )
    assert result["in_front_rate"] == 1.0
    assert result["center_in_image_rate"] == 1.0


def test_pilot_bundle_indices_and_metadata_lookup():
    assert BUNDLE.selected_indices(4934) == [0, 1233, 2466, 3699, 4933]
    report = {
        "videos": [
            {
                "media": "scannetppv2/39f36da05b.mp4",
                "status": "ok",
                "frame_count": 4934,
            }
        ]
    }
    assert BUNDLE.media_record(report, "39f36da05b")["frame_count"] == 4934
    with pytest.raises(ValueError):
        BUNDLE.media_record(report, "missing")


def test_frame_metainfo_detects_bad_scene_and_cross_category_collision(
    tmp_path: Path,
):
    metainfo = {
        "good": [
            {
                "chair": {
                    "num_pixels": 7,
                    "inst_ids": np.asarray([3], dtype=np.int64),
                    "inst_num_pixels": np.asarray([7], dtype=np.int64),
                },
                "table": {
                    "num_pixels": 5,
                    "inst_ids": np.asarray([3], dtype=np.int64),
                    "inst_num_pixels": np.asarray([5], dtype=np.int64),
                },
            }
        ],
        "empty": [],
    }
    path = tmp_path / "frames.npy"
    np.save(path, metainfo, allow_pickle=True)
    result = AUDIT.audit_frame_metainfo(path)
    assert result["schema_errors"] == {
        "duplicate_instance_across_categories": 1,
        "frames_not_nonempty_list": 1,
    }
    assert result["invalid_scenes"] == [
        {
            "scene_id": "empty",
            "reason": "frames_not_nonempty_list",
            "value_type": "list",
            "frame_count": 0,
        }
    ]


def test_jsonl_counts_qa_by_scene(tmp_path: Path):
    path = tmp_path / "vsi.jsonl"
    rows = [
        {
            "video": "scannetppv2/a.mp4",
            "question_type": "count",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ],
        },
        {
            "video": "scannetppv2/a.mp4",
            "question_type": "distance",
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = AUDIT.audit_jsonl(path)
    assert result["scannetpp_rows"] == 2
    assert result["scene_qa_counts"] == {"a": 2}


def test_full_contract_scene_audit(tmp_path: Path):
    scene = tmp_path / "data" / "scene"
    (scene / "scans").mkdir(parents=True)
    (scene / "iphone").mkdir()
    (scene / "scans" / "mesh_aligned_0.05.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 2\nend_header\n"
    )
    (scene / "scans" / "segments.json").write_text(
        json.dumps({"segIndices": [1, 2]}), encoding="utf-8"
    )
    group = {
        "index": 0,
        "id": 10,
        "objectId": 10,
        "label": "chair",
        "segments": [1],
        "obb": {
            "centroid": [0, 0, 2],
            "axesLengths": [1, 1, 1],
            "normalizedAxes": np.eye(3).reshape(-1).tolist(),
        },
    }
    (scene / "scans" / "segments_anno.json").write_text(
        json.dumps({"segGroups": [group]}), encoding="utf-8"
    )
    poses = {
        f"frame_{index:06d}": {
            "intrinsic": np.eye(3).tolist(),
            "aligned_pose": np.eye(4).tolist(),
        }
        for index in range(2)
    }
    (scene / "iphone" / "pose_intrinsic_imu.json").write_text(
        json.dumps(poses), encoding="utf-8"
    )
    exif = {
        str(index): {"PixelXDimension": 1920, "PixelYDimension": 1440}
        for index in range(2)
    }
    (scene / "iphone" / "exif.json").write_text(
        json.dumps(exif), encoding="utf-8"
    )
    frames = [{"chair": {"inst_ids": [0]}}]
    video = {
        "status": "ok",
        "frame_count": 2,
        "avg_fps": 60.0,
        "width": 640,
        "height": 480,
    }
    result = FULL.audit_scene(
        "scene", scene, FULL.summarize_vsi_frames(frames), video
    )
    assert result["status"] == "passed"
    assert result["vsi_observed_instances"] == 1
    assert result["pose_frames"] == 2
    assert result["metainfo_frames"] == 1
    assert result["image_scale_x"] == pytest.approx(1 / 3)

    empty_result = FULL.audit_scene(
        "scene", scene, FULL.summarize_vsi_frames([{}]), video
    )
    assert empty_result["vsi_observed_instances"] == 0
    assert empty_result["identity_join_evidence"] == (
        "not_applicable_no_vsi_instance_observations"
    )


def test_full_contract_accepts_native_multilabel_and_vsi_alias(tmp_path: Path):
    scene = tmp_path / "data" / "scene"
    (scene / "scans").mkdir(parents=True)
    (scene / "iphone").mkdir()
    (scene / "scans" / "mesh_aligned_0.05.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 2\nend_header\n"
    )
    (scene / "scans" / "segments.json").write_text(
        json.dumps({"segIndices": [1, 2]}), encoding="utf-8"
    )
    obb = {
        "centroid": [0, 0, 2],
        "axesLengths": [1, 1, 1],
        "normalizedAxes": np.eye(3).reshape(-1).tolist(),
    }
    groups = [
        {"index": 0, "id": 10, "objectId": 10, "label": "desk", "segments": [1], "obb": obb},
        {"index": 1, "id": 11, "objectId": 11, "label": "cup", "segments": [1, 2], "obb": obb},
    ]
    (scene / "scans" / "segments_anno.json").write_text(
        json.dumps({"segGroups": groups}), encoding="utf-8"
    )
    poses = {
        f"frame_{index:06d}": {
            "intrinsic": np.eye(3).tolist(),
            "aligned_pose": np.eye(4).tolist(),
        }
        for index in range(2)
    }
    (scene / "iphone" / "pose_intrinsic_imu.json").write_text(json.dumps(poses))
    (scene / "iphone" / "exif.json").write_text(json.dumps({
        str(index): {"PixelXDimension": 1920, "PixelYDimension": 1440}
        for index in range(2)
    }))
    result = FULL.audit_scene(
        "scene",
        scene,
        FULL.summarize_vsi_frames([{"table": {"inst_ids": [0]}}]),
        {"status": "ok", "frame_count": 2, "avg_fps": 60.0, "width": 640, "height": 480},
    )
    assert result["multilabel_segments"] == 1
    assert result["multilabel_max_owners"] == 2
    assert result["single_label_policy"] == "official_first3_then_smallest_instance_v1"


@pytest.mark.parametrize(
    ("official", "vsi"),
    [("desk", "table"), ("mug", "cup"), ("shoe", "shoes"), ("chair", "chair")],
)
def test_scannetppv2_label_normalization(official: str, vsi: str):
    assert normalize_scannetppv2_label(official) == vsi
