"""Fail-closed recomputation of canonical validation evidence.

This module deliberately validates the immutable canonical JSONL inputs rather
than upgrading fields from a historical report.  Historical reports are used
only as provenance when a caller explicitly replaces one.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from adt_gt_supported_clip import validate_support_certificate as validate_adt_certificate
from parta_data_contract import ContractError, validate_records
from scannetppv2_support import validate_support_certificate as validate_scannetpp_certificate

from .provenance import sha256_file
from .unified_data import (
    FROZEN_SOURCE_INVENTORY,
    FROZEN_SOURCE_REGISTRY,
    SUPPORT_CERTIFICATE_FILENAMES,
    SUPPORT_CERTIFICATE_REQUIRED,
)

CANONICAL_DATA_FILENAMES = (
    "scene_states.jsonl",
    "frame_states.jsonl",
    "qa_manifest_exact_verified.jsonl",
)


def _file_record(path: Path, payload: bytes | None = None) -> dict[str, Any]:
    if payload is None:
        payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError(f"{path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ContractError(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(row)
    return rows, payload


def _producer_record(project_root: Path, producer: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()
    return {
        "path": str(producer.resolve()),
        "sha256": sha256_file(producer),
        "git_revision": revision,
    }


def _validate_certificate_registry(
    source: str,
    root: Path,
    scenes: list[Mapping[str, Any]],
    frames: list[Mapping[str, Any]],
    qa_rows: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    filename = SUPPORT_CERTIFICATE_FILENAMES[source]
    path = root / filename
    required = source in SUPPORT_CERTIFICATE_REQUIRED
    if not path.is_file():
        if required:
            raise ContractError(f"missing required support certificate registry: {path}")
        return None

    certificates, certificate_payload = _read_jsonl_snapshot(path)
    validator = (
        validate_adt_certificate
        if source == "adt"
        else validate_scannetpp_certificate
        if source == "scannetppv2"
        else None
    )
    if validator is not None:
        for certificate in certificates:
            validator(certificate)

    expected_scenes = {str(scene["scene_id"]) for scene in scenes}
    certificate_scenes = [str(item.get("scene_id")) for item in certificates]
    if len(certificate_scenes) != len(set(certificate_scenes)):
        raise ContractError(f"duplicate {source} support certificate scene")
    if set(certificate_scenes) != expected_scenes:
        raise ContractError(
            f"{source} support certificate scene registry differs from canonical scenes"
        )

    if required:
        certificate_by_binding = {
            (str(item["scene_id"]), str(item["vsi_media"])): item
            for item in certificates
        }
        frame_by_key = {
            (str(item["source_dataset"]), str(item["frame_key"])): item
            for item in frames
        }
        for row in qa_rows:
            scene_id = str(row.get("scene_id"))
            media = str(row.get("vsi_media"))
            certificate = certificate_by_binding.get((scene_id, media))
            if certificate is None:
                raise ContractError(
                    f"{source} QA scene/media is absent from certificate registry: "
                    f"{row.get('qa_id')}"
                )
            if source == "adt":
                provenance = row.get("clip_provenance")
                embedded = (
                    provenance.get("support_certificate")
                    if isinstance(provenance, Mapping)
                    else None
                )
                if embedded != certificate:
                    raise ContractError(
                        f"ADT embedded/external support certificate mismatch: "
                        f"{row.get('qa_id')}"
                    )
                # validate_records subsequently revalidates this same embedded
                # certificate, maximal run, scene/media and clip provenance.
                continue

            if row.get("source_sampling_binding_sha256") != certificate.get(
                "sampling_binding_sha256"
            ):
                raise ContractError(
                    f"ScanNet++ certificate sampling binding mismatch: {row.get('qa_id')}"
                )
            certified_indices = [
                int(item["frame_index"]) for item in certificate["frames"]
            ]
            certified_keys = [str(item["frame_key"]) for item in certificate["frames"]]
            actual_indices = [int(value) for value in row.get("actual_frame_indices", ())]
            actual_keys = [str(value) for value in row.get("actual_frame_keys", ())]
            if actual_indices != certified_indices or actual_keys != certified_keys:
                raise ContractError(
                    f"ScanNet++ certified frames differ from canonical QA: {row.get('qa_id')}"
                )
            for frame_key, frame_index in zip(actual_keys, actual_indices):
                canonical_frame = frame_by_key.get((source, frame_key))
                if (
                    canonical_frame is None
                    or str(canonical_frame.get("scene_id")) != scene_id
                    or int(canonical_frame.get("frame_index", -1)) != frame_index
                ):
                    raise ContractError(
                        f"ScanNet++ certified frame differs from canonical frame table: "
                        f"{row.get('qa_id')}"
                    )
    return {
        **_file_record(path, certificate_payload),
        "certificate_count": len(certificates),
    }


def recompute_validation_report(
    *,
    source: str,
    root: str | Path,
    project_root: str | Path,
    producer: str | Path,
) -> dict[str, Any]:
    """Read and fully validate one canonical root, returning v2 evidence."""
    if source not in FROZEN_SOURCE_REGISTRY:
        raise ContractError(f"unsupported canonical source: {source}")
    root = Path(root).resolve()
    inputs: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for filename in CANONICAL_DATA_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical validation input: {path}")
        parsed, snapshot = _read_jsonl_snapshot(path)
        inputs[filename] = _file_record(path, snapshot)
        rows[filename] = parsed

    scenes = rows["scene_states.jsonl"]
    frames = rows["frame_states.jsonl"]
    qa_rows = rows["qa_manifest_exact_verified.jsonl"]
    certificate = _validate_certificate_registry(
        source, root, scenes, frames, qa_rows
    )
    if certificate is not None:
        inputs[SUPPORT_CERTIFICATE_FILENAMES[source]] = certificate

    report = validate_records(
        scenes,
        frames,
        qa_rows,
        require_fixtures=False,
        expected_sources=(source,),
    ).as_dict()
    expected = dict(FROZEN_SOURCE_INVENTORY[source])
    observed = {"qa": report["qa"], "scenes": report["scenes"]}
    source_observed = {
        "qa": report["source_counts"][source]["qa"],
        "scenes": report["source_counts"][source]["scenes"],
    }
    if observed != expected or source_observed != expected:
        raise ContractError(
            f"canonical inventory differs from expected contract: "
            f"expected={expected}, observed={observed}, source={source_observed}"
        )
    report.update(
        {
            "status": "complete_passed",
            "validation_mode": "full_canonical_recomputation_v1",
            "canonical_inputs": inputs,
            "producer": _producer_record(Path(project_root), Path(producer)),
        }
    )
    if certificate is not None:
        report["trusted_support_certificate_registry"] = {
            "path": SUPPORT_CERTIFICATE_FILENAMES[source],
            "size_bytes": certificate["size_bytes"],
            "sha256": certificate["sha256"],
            "certificate_count": certificate["certificate_count"],
            "anchored_by": "full_canonical_recomputation_v1",
        }
    for record in inputs.values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ContractError(
                f"canonical validation input changed during recomputation: {record['path']}"
            )
    return report


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as handle:
            shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def publish_validation_report(
    payload: Mapping[str, Any],
    output: str | Path,
    *,
    replace_existing: bool = False,
) -> Path | None:
    """Atomically publish evidence and preserve any explicitly replaced report."""
    output = Path(output)
    previous: dict[str, Any] | None = None
    backup: Path | None = None
    if output.exists():
        if not replace_existing:
            raise FileExistsError(
                f"validation output exists; pass --replace-existing explicitly: {output}"
            )
        digest = sha256_file(output)
        backup = output.with_name(f"{output.stem}.previous-{digest[:16]}{output.suffix}")
        if backup.exists() and sha256_file(backup) != digest:
            raise ContractError(f"historical report backup collision: {backup}")
        if not backup.exists():
            _atomic_copy(output, backup)
        previous = {
            **_file_record(backup),
            "original_path": str(output.resolve()),
        }

    final_payload = dict(payload)
    if previous is not None:
        final_payload["previous_validation_report"] = previous

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(final_payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return backup
