"""Runtime-independent collection and hard gates for Part A T0-B."""

from __future__ import annotations

import math
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .provenance import atomic_json_dump
from .provenance import sha256_file


def validate_t0_a_initialization_transaction(
    *,
    report_path: str | Path,
    provenance_path: str | Path,
    run_status_path: str | Path,
    checkpoint_path: str | Path,
    current_code_revision: str,
    guide_artifact_sha256: str,
    vggt_artifact_sha256: str,
    current_manifest_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless formal T0-B starts from the passed T0-A transaction."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    status = json.loads(Path(run_status_path).read_text(encoding="utf-8"))
    checkpoint = Path(checkpoint_path).resolve()
    failures: list[str] = []
    if report.get("schema_version") != "parta_t0_report_v1" or report.get("status") != "complete_passed":
        failures.append("t0_a_report_not_passed")
    if status.get("status") != "complete" or status.get("experiment") != "parta-t0-a":
        failures.append("t0_a_run_not_complete")
    if provenance.get("status") != "complete_passed":
        failures.append("t0_a_provenance_not_passed")
    if provenance.get("a1_checkpoint_role") != "initialization_no_optimizer_updates":
        failures.append("checkpoint_role")
    if provenance.get("a1_checkpoint_optimizer_steps") != 0:
        failures.append("optimizer_steps")
    checkpoint_record = provenance.get("a1_checkpoint_artifact", {})
    shards = checkpoint_record.get("ordered_shards", []) if isinstance(checkpoint_record, Mapping) else []
    if (not checkpoint.is_file() or len(shards) != 1
            or shards[0].get("sha256") != sha256_file(checkpoint)):
        failures.append("checkpoint_hash")
    if provenance.get("a1_checkpoint_state_sha256") != status.get("checkpoint_sha256"):
        failures.append("checkpoint_state_hash")
    if provenance.get("parameter_sha256_before_backward") != provenance.get(
        "parameter_sha256_after_backward"
    ):
        failures.append("initialization_was_updated")
    if provenance.get("git_revision") != current_code_revision or status.get(
        "code_revision"
    ) != current_code_revision:
        failures.append("code_revision")
    if provenance.get("checkpoint_sha256") != guide_artifact_sha256:
        failures.append("guide_hash")
    if provenance.get("vggt_checkpoint_sha256") != vggt_artifact_sha256:
        failures.append("vggt_hash")
    t0_inputs = provenance.get("manifest_sha256", {})
    for source in ("adt", "hypersim"):
        current = current_manifest_inputs.get(source, {})
        files = current.get("files", {}) if isinstance(current, Mapping) else {}
        current_file_hashes = {
            item.get("sha256")
            for item in files.values()
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        } if isinstance(files, Mapping) else set()
        # T0-A predates the three-source unified registry and binds the exact
        # ADT/Hypersim QA manifest content hash.  The current registry nests
        # that hash under source -> files -> filename, so compare against the
        # complete signed file set instead of a removed top-level sha256.
        if t0_inputs.get(source) not in current_file_hashes:
            failures.append(f"manifest.{source}")
    exact = provenance.get("exact_frame_binding_sha256")
    if not isinstance(exact, str) or len(exact) != 64:
        failures.append("exact_frame_binding")
    if failures:
        raise ValueError(f"formal T0-B rejected T0-A transaction: {sorted(set(failures))}")
    return {
        "t0_a_report_sha256": sha256_file(report_path),
        "t0_a_provenance_sha256": sha256_file(provenance_path),
        "t0_a_run_status_sha256": sha256_file(run_status_path),
        "t0_a_checkpoint_sha256": sha256_file(checkpoint),
        "t0_a_checkpoint_state_sha256": provenance["a1_checkpoint_state_sha256"],
        "t0_a_checkpoint_role": provenance["a1_checkpoint_role"],
        "t0_a_checkpoint_optimizer_steps": 0,
        "t0_a_exact_frame_binding_sha256": exact,
        "t0_a_manifest_sha256": dict(t0_inputs),
        "code_revision": current_code_revision,
    }
from .t0 import GradientBatchRecord, summarize_gradient_calibration


@dataclass(frozen=True)
class T0BBatchObservation:
    batch_index: int
    qa_id: str
    source_dataset: str
    qa_loss: float
    state_loss: float
    qa_gradient_norm: float
    state_gradient_norm: float
    shared_gradient_parameter_count: int
    head_gradient_parameter_count: int
    enabled_components: tuple[str, ...]
    masked_components: tuple[str, ...]
    component_losses: Mapping[str, float]
    component_valid_counts: Mapping[str, int]
    matching_valid: bool
    matched_pairs: int
    gt_objects: int
    exact_frame_consistent: bool
    actual_frame_count: int

    @property
    def finite(self) -> bool:
        values = (
            self.qa_loss,
            self.state_loss,
            self.qa_gradient_norm,
            self.state_gradient_norm,
            *self.component_losses.values(),
        )
        return all(math.isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class T0BThresholds:
    minimum_batches: int = 20
    maximum_batches: int = 50
    minimum_valid_fraction: float = 0.95
    minimum_gradient_norm: float = 1e-12
    defaults_status: str = "pending_gate_config_after_d62_execution_evidence"

    def validate(self, requested_batches: int) -> None:
        if not self.minimum_batches <= requested_batches <= self.maximum_batches:
            raise ValueError(
                f"requested T0-B batches must be in "
                f"[{self.minimum_batches},{self.maximum_batches}]"
            )
        if not 0.0 < self.minimum_valid_fraction <= 1.0:
            raise ValueError("minimum_valid_fraction must be in (0,1]")
        if self.minimum_gradient_norm <= 0:
            raise ValueError("minimum_gradient_norm must be positive")


def active_target_components(target: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    enabled = ["existence"]
    masks = {
        "category": target.category_valid,
        "center": target.center_valid,
        "extent": target.extent_valid,
        "visibility": target.visibility_valid,
    }
    enabled.extend(name for name, mask in masks.items() if bool(mask.any()))
    masked = [name for name in masks if name not in enabled]
    return tuple(enabled), tuple(masked)


def parameter_gradient_norm(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter], *, retain_graph: bool
) -> tuple[float, int]:
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable or not loss.requires_grad:
        return 0.0, 0
    gradients = torch.autograd.grad(
        loss, trainable, retain_graph=retain_graph, allow_unused=True
    )
    squared = torch.zeros((), dtype=torch.float64)
    count = 0
    for gradient in gradients:
        if gradient is None:
            continue
        count += 1
        squared += gradient.detach().float().cpu().double().square().sum()
    return float(squared.sqrt().item()), count


def build_t0_b_report(
    observations: Sequence[T0BBatchObservation],
    *,
    requested_batches: int,
    thresholds: T0BThresholds,
    checkpoint_resume_passed: bool,
    runtime_status: str,
    resolved_config_sha256: str,
    manifest_sha256: str | None = None,
    manifest_report_sha256: str | None = None,
    exact_registry_sha256: str | None = None,
    expected_sources: Sequence[str] = ("adt", "hypersim", "scannetppv2"),
    expected_components: Sequence[str] = (
        "existence", "category", "center", "extent", "visibility"
    ),
) -> dict[str, Any]:
    """Build a machine-decidable report without promoting CPU mock evidence."""
    thresholds.validate(requested_batches)
    if runtime_status not in {"awaiting_gpu", "gpu_complete"}:
        raise ValueError("runtime_status must be awaiting_gpu or gpu_complete")
    records = [
        GradientBatchRecord(item.source_dataset, item.qa_gradient_norm, item.state_gradient_norm)
        for item in observations
    ]
    calibration: dict[str, Any] | None = None
    if len(records) == requested_batches:
        calibration = summarize_gradient_calibration(
            records,
            minimum_batches=thresholds.minimum_batches,
            maximum_batches=thresholds.maximum_batches,
            minimum_valid_fraction=thresholds.minimum_valid_fraction,
            minimum_norm=thresholds.minimum_gradient_norm,
        )
    source_counts = Counter(item.source_dataset for item in observations)
    component_enabled = defaultdict(int)
    component_masked = defaultdict(int)
    for item in observations:
        for name in item.enabled_components:
            component_enabled[name] += 1
        for name in item.masked_components:
            component_masked[name] += 1
    expected_source_set = set(expected_sources)
    source_set_exact = set(source_counts) == expected_source_set
    component_mask_consistent = True
    for item in observations:
        enabled = set(item.enabled_components)
        masked = set(item.masked_components)
        expected_component_set = set(expected_components)
        if enabled & masked or (enabled | masked) & expected_component_set != expected_component_set:
            component_mask_consistent = False
        for name in expected_components:
            valid_count = int(item.component_valid_counts.get(name, 0))
            should_enable = valid_count > 0
            if (name in enabled) != should_enable:
                component_mask_consistent = False
            value = float(item.component_losses.get(name, float("nan")))
            if should_enable and not math.isfinite(value):
                component_mask_consistent = False
            if not should_enable and value != 0.0:
                component_mask_consistent = False
    component_coverage = all(component_enabled[name] > 0 for name in expected_components)
    checks = {
        "requested_batch_count": len(observations) == requested_batches,
        "source_registry_exact": source_set_exact,
        "source_balanced": source_set_exact and bool(source_counts)
        and max(source_counts.values()) - min(source_counts.values()) <= 1,
        "loss_finite": bool(observations) and all(item.finite for item in observations),
        "shared_qa_gradients": bool(observations)
        and all(item.qa_gradient_norm > thresholds.minimum_gradient_norm for item in observations),
        "shared_state_gradients": bool(observations)
        and all(item.state_gradient_norm > thresholds.minimum_gradient_norm for item in observations),
        "head_gradients": bool(observations)
        and all(item.head_gradient_parameter_count > 0 for item in observations),
        "matching_valid": bool(observations) and all(item.matching_valid for item in observations),
        "exact_frame_consistency": bool(observations)
        and all(item.exact_frame_consistent for item in observations),
        "checkpoint_resume_equivalence": checkpoint_resume_passed,
        "component_mask_consistency": component_mask_consistent,
        "component_coverage": component_coverage,
        "gradient_calibration": bool(calibration and calibration["passed"]),
    }
    passed = runtime_status == "gpu_complete" and all(checks.values())
    return {
        "schema_version": "parta_t0_b_report_v1",
        "phase": "t0-b",
        "status": "complete_passed" if passed else (
            "awaiting_gpu" if runtime_status == "awaiting_gpu" else "complete_failed"
        ),
        "formal_gpu_evidence": runtime_status == "gpu_complete",
        "resolved_config_sha256": resolved_config_sha256,
        "manifest_sha256": manifest_sha256,
        "manifest_report_sha256": manifest_report_sha256,
        "exact_registry_sha256": exact_registry_sha256,
        "source_registry": sorted(expected_source_set),
        "requested_batches": requested_batches,
        "thresholds": asdict(thresholds),
        "checks": {name: {"passed": value} for name, value in checks.items()},
        "source_batch_counts": dict(sorted(source_counts.items())),
        "expected_sources": sorted(expected_source_set),
        "expected_components": list(expected_components),
        "component_enabled_batch_counts": dict(sorted(component_enabled.items())),
        "component_masked_batch_counts": dict(sorted(component_masked.items())),
        "gradient_calibration": calibration,
        "observations": [asdict(item) | {"finite": item.finite} for item in observations],
    }


def finalize_t0_b_report(payload: Mapping[str, Any], path: str) -> None:
    atomic_json_dump(dict(payload), path)
    if payload.get("status") == "complete_failed":
        failed = [name for name, item in payload["checks"].items() if not item["passed"]]
        raise AssertionError(f"T0-B hard gate failed: {failed}")


def nested_state_digest(value: Any) -> str:
    """Stable content digest for model/optimizer/scheduler/RNG state."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"\0")
            for child in item:
                update(child)
        elif hasattr(item, "dtype") and hasattr(item, "tobytes"):
            digest.update(b"array\0")
            digest.update(str(item.dtype).encode())
            digest.update(repr(tuple(item.shape)).encode())
            digest.update(item.tobytes())
        else:
            digest.update(type(item).__name__.encode() + b"\0" + repr(item).encode())

    update(value)
    return digest.hexdigest()
