import json
import struct
from pathlib import Path

import numpy as np

from src.scannetppv2_rasterizer import (
    Rasterizer,
    compile_rasterizer,
    instance_pixel_counts,
    load_mesh_instances,
)
from src.scannetppv2_support import (
    build_support_certificate,
    validate_support_certificate,
)
from src.parta_data_contract import ContractError
import pytest


PROJECT = Path(__file__).resolve().parents[1]


def test_native_triangle_rasterizer(tmp_path: Path):
    mesh_path = tmp_path / "mesh.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 3\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\n"
        "property uchar blue\nelement face 1\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    payload = b"".join([
        struct.pack("<fffBBB", -0.5, -0.5, 2.0, 0, 0, 0),
        struct.pack("<fffBBB", 0.5, -0.5, 2.0, 0, 0, 0),
        struct.pack("<fffBBB", 0.0, 0.5, 2.0, 0, 0, 0),
        struct.pack("<Biii", 3, 0, 1, 2),
    ])
    mesh_path.write_bytes(header + payload)
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps({"segIndices": [0, 1, 2]}))
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"segGroups": [{
        "index": 0, "segments": [0, 1, 2], "label": "triangle"
    }]}))

    library = tmp_path / "rasterizer.so"
    compile_rasterizer(
        PROJECT / "src/scannetppv2_rasterizer.cpp", library
    )
    mesh = load_mesh_instances(mesh_path, segments, annotation)
    intrinsic = np.asarray([[10, 0, 5], [0, 10, 5], [0, 0, 1]], dtype=float)
    labels = Rasterizer(library).render(
        mesh, np.eye(4), intrinsic, width=10, height=10
    )
    counts = instance_pixel_counts(labels)
    assert labels.shape == (10, 10)
    assert counts[0] > 0
    assert set(np.unique(labels)) <= {-1, 0}


def test_render_support_certificate_is_tamper_evident():
    certificate = build_support_certificate(
        scene_id="scene",
        vsi_media="scannetppv2/scene.mp4",
        sampling_binding_sha256="a" * 64,
        video_total_frames=60,
        video_fps=60.0,
        video_width=640,
        video_height=480,
        source_assets={"mesh": {"sha256": "b" * 64, "size_bytes": 1}},
        rasterizer_source_sha256="c" * 64,
        rasterizer_library_sha256="d" * 64,
        frames=[{
            "frame_index": index,
            "frame_key": f"scene/frame_{index:06d}",
            "pose_sha256": "e" * 64,
            "intrinsic_sha256": "f" * 64,
            "instance_mask_sha256": "1" * 64,
            "visible_instance_pixel_counts": {"0": 20},
        } for index in range(16)],
    )
    validate_support_certificate(certificate)
    certificate["frames"][0]["visible_instance_pixel_counts"]["0"] = 21
    with pytest.raises(ContractError, match="digest mismatch"):
        validate_support_certificate(certificate)
