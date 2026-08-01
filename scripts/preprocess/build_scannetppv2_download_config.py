#!/usr/bin/env python3
"""Build credential-free, reproducible ScanNet++ V2 subset configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


MINIMAL_ASSETS = (
    "scan_mesh_path",
    "scan_mesh_segs_path",
    "scan_anno_json_path",
    # VSI ScanNet++ media are long videos rather than discrete DSLR images.
    # Keep only the lightweight iPhone camera/frame metadata needed to prove
    # the stream and pose mapping; the source RGB/depth payloads stay remote.
    "iphone_pose_intrinsic_imu_path",
    "iphone_exif_path",
)
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be nonempty and unique")
    invalid = [scene for scene in scenes if SCENE_ID_RE.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"invalid scene IDs: {invalid}")
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--pilot-scene")
    parser.add_argument("--exclude-scene", action="append", default=[])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("official source config must contain a YAML mapping")
    source_token = cfg.get("token", "<YOUR_TOKEN_HERE>")
    if source_token != "<YOUR_TOKEN_HERE>":
        raise ValueError("refusing to copy a persisted token into generated config")

    requested = read_scenes(args.scenes)
    excluded = set(args.exclude_scene)
    unknown_exclusions = sorted(excluded - set(requested))
    if unknown_exclusions:
        raise ValueError(f"excluded scenes absent from requested list: {unknown_exclusions}")
    selected = [scene for scene in requested if scene not in excluded]
    if args.pilot_scene is not None:
        if args.pilot_scene not in selected:
            raise ValueError(f"pilot scene not selected: {args.pilot_scene}")
        selected = [args.pilot_scene]

    cfg["token"] = "<YOUR_TOKEN_HERE>"
    cfg["data_root"] = str(args.data_root.resolve())
    cfg["metadata_only"] = False
    cfg["dry_run"] = False
    cfg["verbose"] = False
    cfg["download_scenes"] = selected
    cfg.pop("download_splits", None)
    cfg["download_assets"] = list(MINIMAL_ASSETS)
    cfg.pop("download_options", None)
    cfg["default_assets"] = list(MINIMAL_ASSETS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(cfg, sort_keys=False, width=1000), encoding="utf-8"
    )
    args.output.chmod(0o600)
    manifest = {
        "schema_version": "scannetppv2_download_selection_v1",
        "official_config": str(args.source_config.resolve()),
        "official_config_sha256": sha256_file(args.source_config),
        "generated_config": str(args.output.resolve()),
        "generated_config_sha256": sha256_file(args.output),
        "data_root": str(args.data_root.resolve()),
        "requested_scene_count": len(requested),
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "excluded_scenes": sorted(excluded),
        "assets": list(MINIMAL_ASSETS),
        "credential_contract": "SCANNETPP_TOKEN_environment_only",
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
