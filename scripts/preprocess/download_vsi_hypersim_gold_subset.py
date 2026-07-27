#!/usr/bin/env python3
"""Download the minimal source-GT Hypersim subset used by VSI-590K.

The official Hypersim release stores one remotely seekable ZIP per scene. This
tool reads ZIP central directories over HTTP range requests and extracts only:
camera poses, scene scale/instance metadata, semantic-instance bounding boxes,
and depth/semantic-instance labels for the exact VSI frames.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


URL_TEMPLATE = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/"
    "hypersim/v1/scenes/{scene}.zip"
)
ITEM_PATTERN = re.compile(
    r"^(?P<scene>ai_\d{3}_\d{3})/images/"
    r"scene_(?P<camera>cam_\d{2})_final_preview/"
    r"frame\.(?P<frame>\d{4})\.tonemap$"
)
SCENE_FILES = (
    "_detail/metadata_cameras.csv",
    "_detail/metadata_node_strings.csv",
    "_detail/metadata_nodes.csv",
    "_detail/metadata_scene.csv",
    "_detail/mesh/"
    "metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5",
    "_detail/mesh/"
    "metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5",
    "_detail/mesh/"
    "metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5",
)
CAMERA_FILES = (
    "camera_keyframe_frame_indices.hdf5",
    "camera_keyframe_orientations.hdf5",
    "camera_keyframe_positions.hdf5",
    "metadata_camera.csv",
)
FRAME_MODALITIES = (
    "depth_meters",
    "semantic",
    "semantic_instance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser.parse_args()


class RemoteZipFile:
    def __init__(
        self,
        url: str,
        session: requests.Session,
        timeout: int,
        max_retries: int,
    ):
        self.url = url
        self.session = session
        self.timeout = timeout
        self.max_retries = max_retries
        response = self._request("HEAD")
        self.size = int(response.headers["content-length"])
        self.offset = 0

    def _request(self, method: str, **kwargs) -> requests.Response:
        error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method,
                    self.url,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as current:
                error = current
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"Request failed for {self.url}: {error}")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset = min(self.offset + offset, self.size)
        elif whence == 2:
            self.offset = max(0, self.size + offset)
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self.offset

    def read(self, length: int | None = None) -> bytes:
        available = self.size - self.offset
        length = available if length is None else min(length, available)
        if length <= 0:
            return b""
        end = self.offset + length - 1
        response = self._request(
            "GET", headers={"Range": f"bytes={self.offset}-{end}"}
        )
        data = response.content
        self.offset += len(data)
        return data


def load_items(path: Path) -> dict[str, dict[str, set[str]]]:
    scenes: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = ITEM_PATTERN.match(line.strip())
        if not match:
            raise ValueError(f"Invalid item at line {line_number}: {line}")
        scenes[match["scene"]][match["camera"]].add(match["frame"])
    return scenes


def required_members(
    scene: str, cameras: dict[str, set[str]]
) -> set[str]:
    prefix = f"{scene}/"
    members = {prefix + relative for relative in SCENE_FILES}
    for camera, frames in cameras.items():
        camera_root = f"{prefix}_detail/{camera}/"
        members.update(camera_root + name for name in CAMERA_FILES)
        geometry_root = f"{prefix}images/scene_{camera}_geometry_hdf5/"
        for frame in frames:
            members.update(
                f"{geometry_root}frame.{frame}.{modality}.hdf5"
                for modality in FRAME_MODALITIES
            )
    return members


def extract_scene(
    scene: str,
    cameras: dict[str, set[str]],
    args: argparse.Namespace,
) -> dict:
    url = URL_TEMPLATE.format(scene=scene)
    required = required_members(scene, cameras)
    extracted = 0
    skipped = 0
    total_bytes = 0
    missing: list[str] = []
    session = requests.Session()
    session.headers["Accept-Encoding"] = "identity"

    remote = RemoteZipFile(
        url, session, args.timeout, args.max_retries
    )
    with zipfile.ZipFile(remote) as archive:
        available = set(archive.namelist())
        missing = sorted(required - available)
        for member in sorted(required & available):
            destination = args.output_dir / member
            if destination.is_file() and not args.overwrite:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source:
                data = source.read()
            destination.write_bytes(data)
            extracted += 1
            total_bytes += len(data)

    return {
        "scene": scene,
        "url": url,
        "cameras": len(cameras),
        "target_frames": sum(len(frames) for frames in cameras.values()),
        "required_members": len(required),
        "extracted_members": extracted,
        "skipped_members": skipped,
        "extracted_bytes": total_bytes,
        "missing_members": missing,
    }


def main() -> None:
    args = parse_args()
    scenes = load_items(args.items)
    selected = sorted(scenes)
    if args.scene:
        requested = set(args.scene)
        selected = [scene for scene in selected if scene in requested]
        absent = sorted(requested - set(selected))
        if absent:
            raise ValueError(f"Requested scenes absent from items: {absent}")
    if args.scene_limit is not None:
        selected = selected[: args.scene_limit]

    reports = []

    def run(scene: str) -> dict:
        try:
            report = extract_scene(scene, scenes[scene], args)
            report["status"] = (
                "complete" if not report["missing_members"] else "incomplete"
            )
            return report
        except Exception as error:  # retain per-scene failure for resume
            return {
                "scene": scene,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, scene): scene for scene in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            report = future.result()
            reports.append(report)
            reports.sort(key=lambda item: item["scene"])
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            args.report_output.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "vsi_hypersim_gold_download_v1"
                        ),
                        "requested_scenes": len(selected),
                        "completed_reports": len(reports),
                        "reports": reports,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[{index}/{len(selected)}] "
                f"{report['scene']} {report['status']}",
                flush=True,
            )
            print(json.dumps(report), flush=True)

    counts = defaultdict(int)
    for report in reports:
        counts[report["status"]] += 1
    print(
        json.dumps(
            {
                "requested_scenes": len(selected),
                "status_counts": dict(counts),
                "report_output": str(args.report_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
