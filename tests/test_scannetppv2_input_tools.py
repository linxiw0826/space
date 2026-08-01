import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

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
