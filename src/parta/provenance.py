"""Small D-55 provenance primitives used by Part A runners."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_STATUSES = frozenset({"running", "complete", "failed"})


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def checkpoint_shard_digest(paths: Sequence[str | Path]) -> str:
    """Stable digest over shard names, sizes and content hashes."""
    records = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda path: path.name):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        records.append(
            {
                "name": raw_path.name,
                "size_bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            }
        )
    if not records:
        raise ValueError("checkpoint digest requires at least one shard")
    return stable_sha256(records)


@dataclass(frozen=True)
class ResolvedRunContract:
    """Minimum D-55 identity required before a run can be called complete."""

    run_id: str
    experiment: str
    seed: int
    resolved_config: Mapping[str, Any]
    manifest_sha256: str
    initialization_sha256: str
    code_revision: str
    exact_frame_binding_sha256: str
    output_dir: str
    status: str = "running"
    checkpoint_sha256: str | None = None

    def validate(self) -> None:
        if not self.run_id or not self.experiment or not self.code_revision:
            raise ValueError("run_id, experiment and code_revision are required")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {self.status}")
        for name, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("initialization_sha256", self.initialization_sha256),
            ("exact_frame_binding_sha256", self.exact_frame_binding_sha256),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if self.status == "complete" and not self.checkpoint_sha256:
            raise ValueError("complete run requires checkpoint_sha256")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA256")
        expected = {
            "video_min_frames": 16,
            "video_max_frames": 32,
        }
        for key, value in expected.items():
            if key in self.resolved_config and self.resolved_config[key] != value:
                raise ValueError(f"resolved config violates D-56: {key} must equal {value}")
        if "num_slots" in self.resolved_config and self.resolved_config["num_slots"] != 384:
            raise ValueError("resolved config violates D-58: num_slots must equal 384")

    @property
    def resolved_config_sha256(self) -> str:
        return stable_sha256(self.resolved_config)

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["resolved_config_sha256"] = self.resolved_config_sha256
        return stable_sha256(payload)

    @property
    def run_identity_sha256(self) -> str:
        """Immutable identity excluding status and eventual checkpoint digest."""
        return stable_sha256(
            {
                "run_id": self.run_id,
                "experiment": self.experiment,
                "seed": self.seed,
                "resolved_config_sha256": self.resolved_config_sha256,
                "manifest_sha256": self.manifest_sha256,
                "initialization_sha256": self.initialization_sha256,
                "code_revision": self.code_revision,
                "exact_frame_binding_sha256": self.exact_frame_binding_sha256,
                "output_dir": self.output_dir,
            }
        )

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["resolved_config_sha256"] = self.resolved_config_sha256
        payload["run_fingerprint"] = self.fingerprint
        payload["run_identity_sha256"] = self.run_identity_sha256
        payload["schema_version"] = "parta_run_provenance_v1"
        return payload


def write_run_status(contract: ResolvedRunContract, path: str | Path) -> None:
    """Atomically persist a validated running/complete/failed run state."""
    destination = Path(path)
    payload = contract.to_payload()
    if destination.exists():
        with destination.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("run_identity_sha256") != payload["run_identity_sha256"]:
            raise ValueError("refusing to overwrite provenance for a different run identity")
        previous_status = previous.get("status")
        if previous_status != "running":
            raise ValueError(f"refusing to overwrite terminal run status {previous_status}")
        if contract.status == "running":
            raise ValueError("duplicate running status write is not a valid transition")
    atomic_json_dump(payload, destination)
