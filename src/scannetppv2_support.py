"""Tamper-evident support certificates for ScanNet++ rendered supervision."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.parta_data_contract import ContractError, content_sha256


SCHEMA_VERSION = "scannetppv2_render_support_certificate_v1"
REQUIRED_SOURCE_ASSETS = {"mesh", "segments", "annotation", "pose", "exif"}


def certificate_payload(certificate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: certificate[key]
        for key in (
            "schema_version",
            "scene_id",
            "vsi_media",
            "sampling_policy",
            "sampling_binding_sha256",
            "video_total_frames",
            "video_fps_hex",
            "video_width",
            "video_height",
            "source_assets",
            "video_metadata_sha256",
            "rasterizer_source_sha256",
            "rasterizer_library_sha256",
            "frames",
        )
    }


def build_support_certificate(
    *,
    scene_id: str,
    vsi_media: str,
    sampling_binding_sha256: str,
    video_total_frames: int,
    video_fps: float,
    video_width: int,
    video_height: int,
    source_assets: Mapping[str, Mapping[str, Any]],
    video_metadata_sha256: str,
    rasterizer_source_sha256: str,
    rasterizer_library_sha256: str,
    frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    certificate = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "vsi_media": vsi_media,
        "sampling_policy": "guide_exact_raw_mp4_v1",
        "sampling_binding_sha256": sampling_binding_sha256,
        "video_total_frames": int(video_total_frames),
        "video_fps_hex": float(video_fps).hex(),
        "video_width": int(video_width),
        "video_height": int(video_height),
        "source_assets": {key: dict(value) for key, value in source_assets.items()},
        "video_metadata_sha256": video_metadata_sha256,
        "rasterizer_source_sha256": rasterizer_source_sha256,
        "rasterizer_library_sha256": rasterizer_library_sha256,
        "frames": [dict(frame) for frame in frames],
    }
    certificate["certificate_sha256"] = content_sha256(
        certificate_payload(certificate)
    )
    validate_support_certificate(certificate)
    return certificate


def validate_support_certificate(certificate: Mapping[str, Any]) -> None:
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("Unsupported ScanNet++ support certificate schema")
    if certificate.get("sampling_policy") != "guide_exact_raw_mp4_v1":
        raise ContractError("Unsupported ScanNet++ certificate sampling policy")
    if int(certificate.get("video_total_frames", 0)) <= 0:
        raise ContractError("ScanNet++ certificate video frame count must be positive")
    if int(certificate.get("video_width", 0)) <= 0 or int(certificate.get("video_height", 0)) <= 0:
        raise ContractError("ScanNet++ certificate video dimensions must be positive")
    hashes = [
        certificate.get("sampling_binding_sha256"),
        certificate.get("video_metadata_sha256"),
        certificate.get("rasterizer_source_sha256"),
        certificate.get("rasterizer_library_sha256"),
    ]
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ContractError("ScanNet++ certificate contains invalid SHA256 fields")
    assets = certificate.get("source_assets")
    if not isinstance(assets, dict) or set(assets) != REQUIRED_SOURCE_ASSETS:
        raise ContractError("ScanNet++ certificate source asset set is incomplete")
    for asset in assets.values():
        if (
            not isinstance(asset, dict)
            or int(asset.get("size_bytes", 0)) <= 0
            or not isinstance(asset.get("sha256"), str)
            or len(asset["sha256"]) != 64
        ):
            raise ContractError("ScanNet++ certificate source asset record is invalid")
    frames = certificate.get("frames")
    if not isinstance(frames, list) or not 16 <= len(frames) <= 32:
        raise ContractError("ScanNet++ certificate must bind 16-32 frames")
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != sorted(set(indices)):
        raise ContractError("ScanNet++ certificate frame indices are not unique/sorted")
    required_frame = {
        "frame_index", "frame_key", "pose_sha256", "intrinsic_sha256",
        "instance_mask_sha256", "visible_instance_pixel_counts",
    }
    for frame in frames:
        if not required_frame <= set(frame):
            raise ContractError("ScanNet++ certificate frame fields are incomplete")
        counts = frame["visible_instance_pixel_counts"]
        if not isinstance(counts, dict) or any(int(value) <= 0 for value in counts.values()):
            raise ContractError("ScanNet++ certificate pixel counts must be positive")
        if not 0 <= int(frame["frame_index"]) < int(certificate["video_total_frames"]):
            raise ContractError("ScanNet++ certificate frame index is out of range")
        if any(
            not isinstance(frame[field], str) or len(frame[field]) != 64
            for field in ("pose_sha256", "intrinsic_sha256", "instance_mask_sha256")
        ):
            raise ContractError("ScanNet++ certificate frame hash is invalid")
    expected = content_sha256(certificate_payload(certificate))
    if certificate.get("certificate_sha256") != expected:
        raise ContractError("ScanNet++ support certificate digest mismatch")
