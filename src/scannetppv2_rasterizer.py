"""Dependency-light exact triangle rasterizer for ScanNet++ instance masks."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeshInstances:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_labels: np.ndarray
    annotation_groups: list[dict]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_rasterizer(source: Path, output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++", "-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared",
        "-fopenmp", str(source), "-o", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"rasterizer compilation failed: {completed.stderr}")
    return {
        "source_sha256": sha256_file(source),
        "library_sha256": sha256_file(output),
        "compiler_command": " ".join(command),
    }


def _ply_header(handle) -> tuple[int, int]:
    vertex_count = face_count = None
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("truncated PLY header")
        text = line.decode("ascii").strip()
        if text == "format binary_little_endian 1.0":
            pass
        elif text.startswith("format "):
            raise ValueError(f"unsupported PLY format: {text}")
        elif text.startswith("element vertex "):
            vertex_count = int(text.split()[2])
        elif text.startswith("element face "):
            face_count = int(text.split()[2])
        elif text == "end_header":
            break
    if vertex_count is None or face_count is None:
        raise ValueError("PLY header lacks vertex/face counts")
    return vertex_count, face_count


def load_mesh_instances(
    mesh_path: Path, segments_path: Path, annotation_path: Path
) -> MeshInstances:
    vertex_dtype = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    with mesh_path.open("rb") as handle:
        vertex_count, face_count = _ply_header(handle)
        vertices_raw = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        faces_raw = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if handle.read(1):
            raise ValueError("PLY has trailing or unsupported face payload")
    if len(vertices_raw) != vertex_count or len(faces_raw) != face_count:
        raise ValueError("PLY payload is truncated")
    if not np.all(faces_raw["count"] == 3):
        raise ValueError("ScanNet++ rasterizer requires triangular faces")
    vertices = np.ascontiguousarray(vertices_raw["xyz"], dtype=np.float32)
    faces = np.ascontiguousarray(faces_raw["indices"], dtype=np.int32)

    segments = np.asarray(
        json.loads(segments_path.read_text(encoding="utf-8"))["segIndices"],
        dtype=np.int64,
    )
    if len(segments) != vertex_count or segments.min() < 0:
        raise ValueError("segment IDs do not match nonnegative mesh vertices")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    groups = annotation["segGroups"]
    if [int(group["index"]) for group in groups] != list(range(len(groups))):
        raise ValueError("segGroups.index is not zero-based list position")
    lookup = np.full(int(segments.max()) + 1, -1, dtype=np.int32)
    for group in groups:
        ids = np.asarray(group["segments"], dtype=np.int64)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(lookup)):
            raise ValueError("annotation references out-of-range segment")
        if len(ids) and np.any(lookup[ids] != -1):
            raise ValueError("annotation segment has multiple owners")
        lookup[ids] = int(group["index"])
    vertex_labels = np.ascontiguousarray(lookup[segments], dtype=np.int32)
    return MeshInstances(vertices, faces, vertex_labels, groups)


class Rasterizer:
    def __init__(self, library: Path):
        self.library_path = library
        self.library_sha256 = sha256_file(library)
        self._library = ctypes.CDLL(str(library))
        function = self._library.rasterize_scannetppv2
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        function.restype = ctypes.c_int
        self._function = function

    def render(
        self,
        mesh: MeshInstances,
        camera_from_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        camera = np.ascontiguousarray(camera_from_world, dtype=np.float64)
        k = np.ascontiguousarray(intrinsic, dtype=np.float64)
        if camera.shape != (4, 4) or k.shape != (3, 3):
            raise ValueError("expected camera 4x4 and intrinsic 3x3")
        output = np.empty((height, width), dtype=np.int32)
        status = self._function(
            mesh.vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(mesh.vertices),
            mesh.faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            len(mesh.faces),
            mesh.vertex_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            camera.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            width, height,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        if status != 0:
            raise RuntimeError(f"native rasterizer returned status {status}")
        return output


def instance_pixel_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
