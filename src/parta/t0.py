"""Machine-readable T0-A/T0-B checks for Part A.

This module contains no dataset or GPU assumptions. A later RunnerAgent binds
these checks to the five fixed canonical fixtures and the real Qwen model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .provenance import atomic_json_dump

FIXTURE_SCENE_IDS = (
    "Apartment_release_clean_seq131_M1292",
    "Apartment_release_clean_seq133_M1292",
    "Apartment_release_clean_seq134_M1292",
    "ai_001_001",
    "ai_001_002",
)

DTYPE_TOLERANCES = {
    "torch.float32": {"atol": 1e-6, "rtol": 1e-5},
    "torch.float16": {"atol": 1e-3, "rtol": 1e-3},
    "torch.bfloat16": {"atol": 5e-3, "rtol": 5e-3},
}

T0_A_REQUIRED_CHECKS = frozenset(
    {
        "fixed_fixtures",
        "visual_before_question",
        "exact_frame_contract",
        "shape_mask_frame_span",
        "slot_count_k384",
        "gt_permutation_invariance",
        "empty_gt",
        "all_masked_no_object",
        "finite",
        "qa_supervision_and_e01_trainability",
        "component_shared_gradients",
        "question_invariance",
        "save_resume_equivalence",
        "head_free_equivalence",
    }
)
T0_B_REQUIRED_CHECKS = frozenset({"gradient_calibration"})
REQUIRED_T0_CHECKS = T0_A_REQUIRED_CHECKS | T0_B_REQUIRED_CHECKS


@dataclass(frozen=True)
class TensorComparison:
    passed: bool
    dtype: str
    atol: float
    rtol: float
    max_abs_difference: float
    relative_l2_difference: float


@dataclass(frozen=True)
class GradientBatchRecord:
    source_dataset: str
    qa_gradient_norm: float
    state_gradient_norm: float


def assert_exact_frame_contract(
    frame_ids_a: torch.Tensor,
    frame_ids_b: torch.Tensor,
    masks_a: torch.Tensor,
    masks_b: torch.Tensor,
    spans_a: torch.Tensor,
    spans_b: torch.Tensor,
) -> None:
    for name, left, right in (
        ("frame_ids", frame_ids_a, frame_ids_b),
        ("visual_masks", masks_a, masks_b),
        ("frame_spans", spans_a, spans_b),
    ):
        if not torch.equal(left, right):
            raise AssertionError(f"matched A0/A1-O {name} differ")


def assert_fixed_fixtures(available_scene_ids: Iterable[str]) -> None:
    available = set(available_scene_ids)
    missing = [scene_id for scene_id in FIXTURE_SCENE_IDS if scene_id not in available]
    if missing:
        raise AssertionError(f"missing fixed T0 fixtures (automatic substitution forbidden): {missing}")


def assert_finite_tensors(named_tensors: Mapping[str, torch.Tensor]) -> None:
    nonfinite = [name for name, tensor in named_tensors.items() if not torch.isfinite(tensor).all()]
    if nonfinite:
        raise AssertionError(f"non-finite T0 tensors: {nonfinite}")


def compare_tensors(left: torch.Tensor, right: torch.Tensor) -> TensorComparison:
    if left.shape != right.shape:
        raise AssertionError(f"tensor shapes differ: {left.shape} != {right.shape}")
    dtype_key = str(left.dtype)
    if dtype_key != str(right.dtype):
        raise AssertionError(f"tensor dtypes differ: {left.dtype} != {right.dtype}")
    if dtype_key not in DTYPE_TOLERANCES:
        raise ValueError(f"no D-58 tolerance registered for {dtype_key}")
    tolerance = DTYPE_TOLERANCES[dtype_key]
    left_fp32 = left.detach().float()
    right_fp32 = right.detach().float()
    difference = left_fp32 - right_fp32
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    denominator = max(float(left_fp32.norm().item()), float(right_fp32.norm().item()), 1e-12)
    relative_l2 = float(difference.norm().item()) / denominator
    passed = bool(
        torch.allclose(left_fp32, right_fp32, atol=tolerance["atol"], rtol=tolerance["rtol"])
    )
    return TensorComparison(
        passed=passed,
        dtype=dtype_key,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
        max_abs_difference=max_abs,
        relative_l2_difference=relative_l2,
    )


def shared_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """FP32 accumulated norm over shared trainable parameters only."""
    squared = torch.zeros((), dtype=torch.float64)
    found = False
    for parameter in parameters:
        if not parameter.requires_grad or parameter.grad is None:
            continue
        found = True
        gradient = parameter.grad.detach().float().cpu()
        squared += gradient.double().square().sum()
    return float(squared.sqrt().item()) if found else 0.0


def component_shared_gradient_norms(
    losses: Mapping[str, torch.Tensor],
    shared_parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float]:
    """Measure each enabled state component without including head-only params."""
    parameters = tuple(parameter for parameter in shared_parameters if parameter.requires_grad)
    norms = {}
    for name, loss in losses.items():
        if not loss.requires_grad:
            norms[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = torch.zeros((), dtype=torch.float64)
        for gradient in gradients:
            if gradient is not None:
                squared += gradient.detach().float().cpu().double().square().sum()
        norms[name] = float(squared.sqrt().item())
    return norms


def assert_component_shared_gradient_norms(
    norms: Mapping[str, float],
    enabled_components: Iterable[str],
    minimum_norm: float = 1e-12,
) -> None:
    """Hard gate every enabled state term; returning diagnostics is insufficient."""
    enabled = tuple(enabled_components)
    missing = [name for name in enabled if name not in norms]
    invalid = [
        name
        for name in enabled
        if name in norms and (not np.isfinite(norms[name]) or norms[name] <= minimum_norm)
    ]
    if missing or invalid:
        raise AssertionError(
            f"state component shared-gradient gate failed: missing={missing}, "
            f"nonfinite_or_le_{minimum_norm}={invalid}"
        )


def summarize_gradient_calibration(
    records: Sequence[GradientBatchRecord],
    minimum_batches: int = 50,
    maximum_batches: int = 100,
    minimum_valid_fraction: float = 0.95,
    minimum_norm: float = 1e-12,
) -> dict[str, object]:
    if not minimum_batches <= len(records) <= maximum_batches:
        raise ValueError(f"T0-B requires {minimum_batches}-{maximum_batches} batches")
    if len(set(record.source_dataset for record in records)) < 2:
        raise ValueError("T0-B batches must be source-stratified")
    source_counts = {
        source: sum(record.source_dataset == source for record in records)
        for source in sorted(set(record.source_dataset for record in records))
    }
    if max(source_counts.values()) - min(source_counts.values()) > 1:
        raise ValueError(f"T0-B source strata are imbalanced: {source_counts}")

    qa = np.asarray([record.qa_gradient_norm for record in records], dtype=np.float64)
    state = np.asarray([record.state_gradient_norm for record in records], dtype=np.float64)
    valid = np.isfinite(qa) & np.isfinite(state) & (qa > minimum_norm) & (state > minimum_norm)
    valid_fraction = float(valid.mean())
    if valid.any():
        qa_valid = qa[valid]
        state_valid = state[valid]
        state_over_qa = state_valid / qa_valid
        qa_over_state = qa_valid / state_valid
        median_qa = float(np.median(qa_valid))
        median_state = float(np.median(state_valid))
        median_qa_over_state = float(np.median(qa_over_state))
        lambda_state = float(np.clip(0.1 * median_qa_over_state, 0.01, 0.1))
        stats = {
            "qa": _percentiles(qa_valid),
            "state": _percentiles(state_valid),
            "state_over_qa": _percentiles(state_over_qa),
            "qa_over_state": _percentiles(qa_over_state),
        }
    else:
        lambda_state = None
        stats = {"qa": None, "state": None, "state_over_qa": None, "qa_over_state": None}
    return {
        "passed": valid_fraction >= minimum_valid_fraction,
        "batch_count": len(records),
        "valid_batch_count": int(valid.sum()),
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": minimum_valid_fraction,
        "minimum_gradient_norm": minimum_norm,
        "source_batch_counts": source_counts,
        "statistics": stats,
        "lambda_state_candidate": lambda_state,
        "formula": "clip(0.1 * median_per_batch(g_QA / g_state), 0.01, 0.1)",
    }


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


class T0Report:
    """Collect named checks without hiding individual failures in one score."""

    def __init__(
        self,
        run_id: str,
        resolved_config_sha256: str,
        *,
        phase: str = "all",
    ):
        required_by_phase = {
            "t0-a": T0_A_REQUIRED_CHECKS,
            "t0-b": T0_B_REQUIRED_CHECKS,
            "all": REQUIRED_T0_CHECKS,
        }
        if phase not in required_by_phase:
            raise ValueError(f"unsupported T0 phase: {phase!r}")
        self.required_checks = required_by_phase[phase]
        self.payload: dict[str, object] = {
            "schema_version": "parta_t0_report_v1",
            "run_id": run_id,
            "resolved_config_sha256": resolved_config_sha256,
            "fixtures": list(FIXTURE_SCENE_IDS),
            "phase": phase,
            "checks": {},
            "status": "running",
        }

    def add_boolean(self, name: str, passed: bool, **metrics: object) -> None:
        self.payload["checks"][name] = {"passed": bool(passed), **metrics}

    def add_comparison(self, name: str, comparison: TensorComparison) -> None:
        self.payload["checks"][name] = asdict(comparison)

    def add_component_gradient_check(
        self,
        norms: Mapping[str, float],
        enabled_components: Iterable[str],
        minimum_norm: float = 1e-12,
    ) -> None:
        enabled = tuple(enabled_components)
        try:
            assert_component_shared_gradient_norms(norms, enabled, minimum_norm)
        except AssertionError:
            self.add_boolean(
                "component_shared_gradients",
                False,
                norms=dict(norms),
                enabled_components=list(enabled),
                minimum_norm=minimum_norm,
            )
            raise
        self.add_boolean(
            "component_shared_gradients",
            True,
            norms=dict(norms),
            enabled_components=list(enabled),
            minimum_norm=minimum_norm,
        )

    def finalize(self, path: str) -> Mapping[str, object]:
        checks = self.payload["checks"]
        missing = sorted(self.required_checks - set(checks))
        failed = sorted(name for name, item in checks.items() if not item.get("passed", False))
        if missing or failed:
            self.payload["status"] = "complete_failed"
            self.payload["missing_required_checks"] = missing
            self.payload["failed_checks"] = failed
            atomic_json_dump(self.payload, path)
            raise AssertionError(f"T0 hard gate failed: missing={missing}, failed={failed}")
        self.payload["status"] = "complete_passed"
        atomic_json_dump(self.payload, path)
        return self.payload
