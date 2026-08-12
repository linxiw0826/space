"""A1-O-drop checkpoint filtering and load-audit helpers."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import torch
from torch import nn

from .provenance import sha256_file

STATE_HEAD_PREFIXES = ("parta_state_head.", "module.parta_state_head.")


@dataclass(frozen=True)
class HeadFreeLoadAudit:
    dropped_state_head_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    loaded_shared_keys: int
    expected_state_head_keys: tuple[str, ...] = ()
    minimum_dropped_state_head_keys: int = 1

    @property
    def passed(self) -> bool:
        expected_ok = (
            set(self.dropped_state_head_keys) == set(self.expected_state_head_keys)
            if self.expected_state_head_keys
            else len(self.dropped_state_head_keys) >= self.minimum_dropped_state_head_keys
        )
        return expected_ok and not self.missing_keys and not self.unexpected_keys


def is_state_head_key(key: str) -> bool:
    return key.startswith(STATE_HEAD_PREFIXES)


def filter_head_free_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Remove only the training-side state head from an A1-O checkpoint."""
    kept = {}
    dropped = []
    for key, value in state_dict.items():
        if is_state_head_key(key):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, tuple(sorted(dropped))


def load_head_free_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    expected_state_head_keys: tuple[str, ...] = (),
    minimum_dropped_state_head_keys: int = 1,
) -> HeadFreeLoadAudit:
    """Load common QA/backbone weights without ever constructing the head."""
    filtered, dropped = filter_head_free_state_dict(state_dict)
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = tuple(sorted(key for key in incompatible.missing_keys if not is_state_head_key(key)))
    unexpected = tuple(sorted(key for key in incompatible.unexpected_keys if not is_state_head_key(key)))
    audit = HeadFreeLoadAudit(
        dropped_state_head_keys=dropped,
        missing_keys=missing,
        unexpected_keys=unexpected,
        loaded_shared_keys=len(filtered) - len(unexpected),
        expected_state_head_keys=tuple(sorted(expected_state_head_keys)),
        minimum_dropped_state_head_keys=minimum_dropped_state_head_keys,
    )
    return audit


TRAINING_CHECKPOINT_SCHEMA = "parta_training_checkpoint_v1"


@dataclass(frozen=True)
class ResumeContract:
    """Immutable inputs that a resumed run must reproduce exactly."""

    arm: str
    manifest_sha256: str
    resolved_config_sha256: str
    matched_contract_sha256: str
    transaction_kind: str = "formal"
    promotable: bool = True

    def validate(self) -> None:
        if self.transaction_kind not in {"formal", "engineering"}:
            raise ValueError("unknown checkpoint transaction kind")
        if self.promotable != (self.transaction_kind == "formal"):
            raise ValueError("engineering checkpoints must be non-promotable")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    if required - set(state):
        raise ValueError(f"checkpoint RNG state is incomplete: {sorted(required - set(state))}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Durably publish one checkpoint without exposing partial bytes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    sampler_position: int,
    contract: ResumeContract,
    model_state: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
) -> None:
    contract.validate()
    if global_step < 0 or epoch < 0 or sampler_position < 0:
        raise ValueError("checkpoint counters must be non-negative")
    atomic_torch_save(
        {
            "schema_version": TRAINING_CHECKPOINT_SCHEMA,
            "contract": asdict(contract),
            "model": dict(model_state) if model_state is not None else model.state_dict(),
            "optimizer": dict(optimizer_state) if optimizer_state is not None else optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
            "sampler_position": sampler_position,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_contract: ResumeContract,
) -> dict[str, int]:
    """Restore a full optimizer transaction, failing closed on identity drift."""
    expected_contract.validate()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    if payload.get("contract") != asdict(expected_contract):
        raise ValueError("resume contract mismatch; refusing cross-run checkpoint reuse")
    incompatible = model.load_state_dict(payload["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("strict training checkpoint model restore failed")
    optimizer.load_state_dict(payload["optimizer"])
    saved_scheduler = payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("scheduler presence differs from checkpoint")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    restore_rng_state(payload["rng_state"])
    return {
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "sampler_position": int(payload["sampler_position"]),
    }


def load_fsdp_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_contract: ResumeContract,
) -> dict[str, int]:
    """Restore full FSDP model/optimizer state using the official conversions."""
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig, FullStateDictConfig,
        FullyShardedDataParallel, StateDictType,
    )

    if not isinstance(model, FullyShardedDataParallel):
        raise TypeError("load_fsdp_training_checkpoint requires an FSDP model")
    expected_contract.validate()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    if payload.get("contract") != asdict(expected_contract):
        raise ValueError("resume contract mismatch; refusing cross-run checkpoint reuse")
    with FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    ):
        incompatible = model.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("strict FSDP model restore failed")
        optimizer_state = fsdp_optimizer_state_to_load(
            model, optimizer, payload["optimizer"], fsdp_api=FullyShardedDataParallel
        )
    optimizer.load_state_dict(optimizer_state)
    saved_scheduler = payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("scheduler presence differs from FSDP checkpoint")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)
    restore_rng_state(payload["rng_state"])
    return {
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "sampler_position": int(payload["sampler_position"]),
    }


def fsdp_optimizer_state_to_load(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    full_optimizer_state: Mapping[str, Any],
    *,
    fsdp_api: Any,
) -> Mapping[str, Any]:
    """Injectable seam around the official FSDP optimizer-state conversion."""
    return fsdp_api.optim_state_dict_to_load(model, optimizer, full_optimizer_state)


def export_head_free_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
) -> HeadFreeLoadAudit:
    """Export A1-O-drop weights without constructing a state-head module."""
    payload = torch.load(Path(source_path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("training checkpoint schema mismatch")
    contract = payload.get("contract", {})
    if contract.get("arm") != "a1o":
        raise ValueError("head-free export requires an A1-O training checkpoint")
    if contract.get("transaction_kind") != "formal" or contract.get("promotable") is not True:
        raise ValueError("head-free export rejects engineering/non-promotable checkpoints")
    filtered, dropped = filter_head_free_state_dict(payload["model"])
    if not dropped:
        raise ValueError("A1-O checkpoint contains no state-head parameters")
    source_path = Path(source_path).resolve()
    source_contract = dict(contract)
    source_contract["source_checkpoint"] = {
        "path": str(source_path),
        "sha256": sha256_file(source_path),
        "role": "selected_validation",
        "global_step": int(payload.get("global_step", -1)),
    }
    export = {
        "schema_version": "parta_a1o_drop_checkpoint_v1",
        "source_contract": source_contract,
        "model": filtered,
        "dropped_state_head_keys": dropped,
        "qa_forward_contract": "a0_shared_forward_v1",
    }
    atomic_torch_save(export, destination_path)
    return HeadFreeLoadAudit(
        dropped_state_head_keys=dropped,
        missing_keys=(),
        unexpected_keys=(),
        loaded_shared_keys=len(filtered),
        expected_state_head_keys=dropped,
    )


def load_head_free_artifact(
    model: nn.Module, artifact_path: str | Path
) -> tuple[HeadFreeLoadAudit, dict[str, Any]]:
    """Strictly load an A1-O-drop artifact into a model built without a head."""
    if hasattr(model, "parta_state_head"):
        raise ValueError("head-free load requires a model constructed without state head")
    payload = torch.load(Path(artifact_path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "parta_a1o_drop_checkpoint_v1":
        raise ValueError("A1-O-drop artifact schema mismatch")
    state = payload.get("model")
    if not isinstance(state, Mapping) or any(is_state_head_key(key) for key in state):
        raise ValueError("A1-O-drop artifact contains invalid state-head keys")
    incompatible = model.load_state_dict(state, strict=False)
    audit = HeadFreeLoadAudit(
        dropped_state_head_keys=tuple(payload.get("dropped_state_head_keys", ())),
        missing_keys=tuple(sorted(incompatible.missing_keys)),
        unexpected_keys=tuple(sorted(incompatible.unexpected_keys)),
        loaded_shared_keys=len(state) - len(incompatible.unexpected_keys),
        expected_state_head_keys=tuple(payload.get("dropped_state_head_keys", ())),
    )
    if not audit.passed:
        raise ValueError(f"head-free shared GUIDE load failed: {audit}")
    tensor_registry = [
        {"name": key, "shape": list(value.shape), "dtype": str(value.dtype),
         "sha256": __import__("hashlib").sha256(
             value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
         ).hexdigest()}
        for key, value in sorted(state.items())
    ]
    report = {
        "schema_version": "parta_head_free_load_audit_v1",
        "qa_forward_contract": payload.get("qa_forward_contract"),
        "source_contract": payload.get("source_contract"),
        "dropped_state_head_keys": list(audit.dropped_state_head_keys),
        "missing_keys": list(audit.missing_keys),
        "unexpected_keys": list(audit.unexpected_keys),
        "loaded_shared_keys": audit.loaded_shared_keys,
        "shared_state_sha256": __import__("hashlib").sha256(
            repr(tensor_registry).encode()
        ).hexdigest(),
    }
    return audit, report
