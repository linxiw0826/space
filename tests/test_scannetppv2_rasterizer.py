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
