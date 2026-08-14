"""Fail-closed orchestration contracts for the Part A GPU pretrain gates."""

from __future__ import annotations

import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import atomic_json_dump, sha256_file, stable_sha256
from .resource_profile_contract import validate_rank_failure_rows


FORMAL_SOURCE_REGISTRY = ("adt", "hypersim", "scannetppv2")
PHASES = ("t0_b", "engineering_coverage", "resource_profile")
ENGINEERING_COVERAGE_REQUIREMENTS = (
    "t0_a_final_commit",
    "three_source_t0_b",
    "a1o_fixed_train_subset_learnability",
    "matched_a0_a1o_real_runner_steps",
    "checkpoint_save_resume",
    "a1o_drop_head_free_val_load",
    "validator",
    "resource_preflight",
    "provenance",
    "engineering_state_non_promotable",
    "formal_startup_step0",
    "engineering_subset_not_extra_weighted",
)
TRUSTED_PRODUCER_NAME = "produce_pretrain_gate_report.py"
CANONICAL_PRODUCER = Path(__file__).resolve().parents[2] / "scripts/parta/produce_pretrain_gate_report.py"
# PENDING[D-62 execution evidence]: the reviewed producer digest must be frozen on the
# execution server after transfer.  Code-server edits must never self-attest a digest.
TRUSTED_PRODUCER_SHA256: str | None = None


def validate_formal_training_authorization(
    unified_gate_path: str | Path,
    frozen_config_path: str | Path,
    *,
    manifest_sha256: str,
    manifest_report_sha256: str,
    matched_contract_sha256: str,
    profile_report_sha256: str,
    guide_artifact_sha256: str,
    vggt_artifact_sha256: str,
    code_revision: str,
    resolved_training_config: Mapping[str, Any],
    expected_source_registry: Sequence[str] = FORMAL_SOURCE_REGISTRY,
    formal_arm: str | None = None,
) -> dict[str, Any]:
    """Validate the one finalize-only authorization consumed by formal training.

    Both files are immutable inputs.  The gate must bind the freeze artifact and
    the freeze artifact must bind every identity that can change experiment
    semantics.  No path-only trust or permissive subset comparison is allowed.
    """
    gate_path = Path(unified_gate_path).resolve()
    freeze_path = Path(frozen_config_path).resolve()
    if not gate_path.is_file() or not freeze_path.is_file():
        raise FileNotFoundError("formal training requires unified gate and frozen config artifacts")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if tuple(freeze.get("formal_source_registry", ())) != tuple(expected_source_registry):
        raise ValueError("frozen config does not bind the exact D-62 three-source registry")
    lifecycle = freeze.get("engineering_lifecycle", {})
    required_lifecycle = {
        "subset_is_train_internal": True,
        "subset_extra_weight": False,
        "model_promotable": False,
        "optimizer_promotable": False,
        "scheduler_promotable": False,
        "rng_promotable": False,
        "sampler_promotable": False,
        "formal_arms_start_at_step": 0,
        "formal_arms_share_initialization": True,
    }
    if any(lifecycle.get(key) != value for key, value in required_lifecycle.items()):
        raise ValueError("frozen config violates the D-62 engineering/formal lifecycle")
    startup = freeze.get("formal_startup_contract", {})
    if startup.get("schema_version") != "parta_formal_startup_v1":
        raise ValueError("freeze lacks a machine-decidable formal startup contract")
    arms = startup.get("arms", {})
    if set(arms) != {"a0", "a1o"}:
        raise ValueError("formal startup contract must contain exactly A0 and A1-O")
    initializations = {item.get("initialization_sha256") for item in arms.values()}
    if len(initializations) != 1 or any(item.get("start_step") != 0 for item in arms.values()):
        raise ValueError("formal arms must share one initialization and start at step 0")
    if formal_arm is not None and formal_arm not in arms:
        raise ValueError("runtime arm is absent from formal startup contract")
    required_gate = {
        "schema_version": "parta_unified_pretrain_gate_v1",
        "status": "complete_passed",
        "formal_gpu_evidence": True,
        "formal_config_frozen": True,
        "training_authorized_by_this_artifact": True,
        "freeze_artifact_sha256": sha256_file(freeze_path),
    }
    mismatches = [key for key, value in required_gate.items() if gate.get(key) != value]
    if mismatches:
        raise ValueError(f"unified gate is not a finalize-only training authorization: {mismatches}")
    if gate.get("frozen_config_artifact_sha256") != stable_sha256(freeze):
        raise ValueError("unified gate does not bind the supplied frozen config")
    resolved_path = Path(str(freeze.get("resolved_training_config_path", ""))).resolve()
    if resolved_path != Path(str(resolved_training_config.get("artifact_path", ""))).resolve():
        raise ValueError("runtime resolved-config artifact differs from frozen artifact")
    if not resolved_path.is_file() or sha256_file(resolved_path) != freeze.get(
        "resolved_training_config_sha256"
    ):
        raise ValueError("frozen resolved-training-config hash mismatch")
    frozen_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": manifest_sha256,
        "manifest_report_sha256": manifest_report_sha256,
        "matched_contract_sha256": matched_contract_sha256,
        "profile_report_sha256": profile_report_sha256,
        "guide_artifact_sha256": guide_artifact_sha256,
        "vggt_artifact_sha256": vggt_artifact_sha256,
        "code_revision": code_revision,
        "training_config": dict(resolved_training_config["training_config"]),
    }
    if frozen_resolved != expected:
        differing = sorted(key for key in set(frozen_resolved) | set(expected)
                           if frozen_resolved.get(key) != expected.get(key))
        raise ValueError(f"runtime differs from frozen formal config: {differing}")
    if freeze.get("manifest_sha256") != [manifest_sha256]:
        raise ValueError("freeze artifact does not bind the runtime manifest")
    if freeze.get("profile_report_sha256") != profile_report_sha256:
        raise ValueError("freeze artifact does not bind the selected profile")
    return {
        "unified_gate_sha256": sha256_file(gate_path),
        "frozen_config_sha256": sha256_file(freeze_path),
        "resolved_training_config_sha256": sha256_file(resolved_path),
        **expected,
        "formal_startup_contract_sha256": stable_sha256(startup),
    }


@dataclass(frozen=True)
class ProvisionalGateDefaults:
    overfit_examples: int = 64
    overfit_steps: int = 100
    overfit_minimum_loss_decrease_fraction: float = 0.20
    profile_frame_counts: tuple[int, ...] = (32,)
    defaults_status: str = "pending_gate_config_after_d62_execution_evidence"

    def validate(self) -> None:
        if self.overfit_examples < 1 or self.overfit_steps < 1:
            raise ValueError("overfit examples and steps must be positive")
        if not 0.0 < self.overfit_minimum_loss_decrease_fraction < 1.0:
            raise ValueError("overfit loss-decrease threshold must be in (0,1)")
        if self.profile_frame_counts != (32,):
            raise ValueError("profiling must cover only the frozen 32-frame worst-case point")


@dataclass(frozen=True)
class PhaseCommand:
    phase: str
    argv: tuple[str, ...]
    resolved_config: Mapping[str, Any]
    manifest_path: str
    provenance_path: str
    producer_path: str
    producer_trust_registry_path: str

    def validate(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase: {self.phase}")
        if not self.argv or any(not isinstance(item, str) or not item for item in self.argv):
            raise ValueError(f"{self.phase} argv must be a non-empty string list")
        for name in ("manifest_path", "provenance_path", "producer_path"):
            if not getattr(self, name):
                raise ValueError(f"{self.phase} {name} is required")
        producer = Path(self.producer_path).resolve()
        if producer != CANONICAL_PRODUCER.resolve():
            raise ValueError("phase must use the repository formal gate producer")
        trust_path = Path(self.producer_trust_registry_path).resolve()
        if not trust_path.is_file():
            raise ValueError(
                "PENDING[D-62 execution evidence]: missing execution-server producer trust registry"
            )
        trust = json.loads(trust_path.read_text(encoding="utf-8"))
        expected_trust = {
            "schema_version": "parta_gate_producer_trust_v1",
            "producer_name": TRUSTED_PRODUCER_NAME,
            "producer_path": str(producer),
            "review_status": "static_review_passed",
            "frozen_on_execution_server": True,
        }
        if any(trust.get(key) != value for key, value in expected_trust.items()):
            raise ValueError("invalid execution-server producer trust registry")
        digest = sha256_file(producer)
        if trust.get("producer_sha256") != digest:
            raise ValueError("formal gate producer differs from execution-server frozen hash")
        if TRUSTED_PRODUCER_SHA256 is not None and digest != TRUSTED_PRODUCER_SHA256:
            raise ValueError("formal gate producer differs from preregistered source hash")
        normalized = [str(Path(item).resolve()) if index == 1 else item for index, item in enumerate(self.argv)]
        if len(normalized) < 2 or normalized[1] != str(producer):
            raise ValueError("actual argv is not bound to producer_path")
        required = {"--phase", self.phase, "--contract", "{contract_path}", "--report", "{report_path}"}
        if not required.issubset(set(self.argv)):
            raise ValueError("formal producer argv lacks signed phase/contract/report arguments")

    @classmethod
    def from_json(cls, path: str | Path, expected_phase: str) -> "PhaseCommand":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        command = cls(
            phase=str(payload.get("phase", "")),
            argv=tuple(payload.get("argv", ())),
            resolved_config=dict(payload.get("resolved_config", {})),
            manifest_path=str(payload.get("manifest_path", "")),
            provenance_path=str(payload.get("provenance_path", "")),
            producer_path=str(payload.get("producer_path", "")),
            producer_trust_registry_path=str(payload.get("producer_trust_registry_path", "")),
        )
        command.validate()
        if command.phase != expected_phase:
            raise ValueError(f"expected {expected_phase} command, got {command.phase}")
        return command


def awaiting_gpu_plan(
    output_dir: str | Path,
    defaults: ProvisionalGateDefaults,
    commands: Mapping[str, PhaseCommand | None],
) -> dict[str, Any]:
    defaults.validate()
    phases = []
    for phase in PHASES:
        command = commands.get(phase)
        phases.append({
            "phase": phase,
            "status": "awaiting_gpu",
            "command": list(command.argv) if command else None,
            "report_path": None,
        })
    return {
        "schema_version": "parta_pretrain_gate_plan_v1",
        "status": "awaiting_gpu",
        "formal_gpu_evidence": False,
        "phase_order": list(PHASES),
        "defaults": asdict(defaults),
        "phases": phases,
        "output_dir": str(Path(output_dir).resolve()),
    }


def _true_check(report: Mapping[str, Any], name: str) -> bool:
    value = report.get("checks", {}).get(name)
    return bool(value.get("passed")) if isinstance(value, Mapping) else bool(value)


def validate_phase_report(
    phase: str, report: Mapping[str, Any], defaults: ProvisionalGateDefaults
) -> list[str]:
    """Return failures; an empty list means the GPU report satisfies its contract."""
    failures: list[str] = []
    result = report.get("result", report)
    if report.get("formal_gpu_evidence") is not True:
        failures.append("formal_gpu_evidence")
    if report.get("status") != "complete_passed":
        failures.append("status")
    if phase == "t0_b":
        required = (
            "requested_batch_count", "source_registry_exact", "source_balanced",
            "loss_finite", "shared_qa_gradients", "shared_state_gradients",
            "head_gradients", "matching_valid", "exact_frame_consistency",
            "checkpoint_resume_equivalence", "component_mask_consistency",
            "component_coverage", "gradient_calibration",
        )
        failures.extend(name for name in required if not _true_check(result, name))
    elif phase == "engineering_coverage":
        matrix = result.get("coverage_matrix", {})
        if result.get("schema_version") != "parta_engineering_coverage_v2":
            failures.append("engineering_coverage_schema")
        if set(matrix) != set(ENGINEERING_COVERAGE_REQUIREMENTS):
            failures.append("coverage_matrix_exact")
        failures.extend(
            name for name in ENGINEERING_COVERAGE_REQUIREMENTS
            if not isinstance(matrix.get(name), Mapping)
            or not isinstance(matrix[name].get("artifact_path"), str)
            or not isinstance(matrix[name].get("artifact_sha256"), str)
            or len(matrix[name]["artifact_sha256"]) != 64
            or not isinstance(matrix[name].get("semantic_summary"), Mapping)
        )
        subset = result.get("engineering_subset", {})
        reuse = subset.get("formal_train_reuse", {})
        promotion = subset.get("transaction_promotion", {})
        if (not isinstance(subset.get("path"), str)
                or not isinstance(subset.get("sha256"), str)
                or len(subset.get("sha256", "")) != 64
                or reuse.get("subset_rows_remain_in_train_manifest") is not True
                or reuse.get("source_balanced_weights_unchanged") is not True
                or reuse.get("extra_sampling_weight") is not False):
            failures.append("engineering_subset_contract")
        if (promotion.get("promotable_to_formal_training") is not False
                or set(promotion.get("discard", ()))
                   != {"model", "optimizer", "scheduler", "rng", "sampler"}
                or promotion.get("formal_restart_optimizer_step") != 0):
            failures.append("engineering_transaction_promotable")
    elif phase == "resource_profile":
        measurements = result.get("measurements")
        top_contract = result.get("normalized_execution_contract")
        top_contract_sha = result.get("normalized_execution_contract_sha256")
        if (not isinstance(top_contract, Mapping)
                or top_contract_sha != stable_sha256(top_contract)):
            failures.append("profile_top_level_execution_contract")
        if not isinstance(measurements, list):
            failures.append("measurements")
        else:
            observed = tuple(sorted({item.get("frame_count") for item in measurements}))
            if observed != defaults.profile_frame_counts:
                failures.append("profile_frame_counts")
            feasible = []
            for item in measurements:
                for name in (
                    "peak_memory_bytes", "total_memory_bytes", "step_time_seconds",
                    "throughput_samples_per_second", "batch_size",
                    "gradient_accumulation_steps", "forward_backward_steps", "oom",
                    "world_size", "per_rank_peak_memory_bytes",
                    "peak_reserved_memory_bytes", "distributed_strategy", "finite",
                    "per_rank_batch_size", "normalized_execution_contract",
                    "normalized_execution_contract_sha256",
                ):
                    if name not in item:
                        failures.append(f"measurement.{name}")
                always_numeric = (item.get("batch_size"), item.get("per_rank_batch_size"),
                                  item.get("gradient_accumulation_steps"), item.get("world_size"))
                measured_numeric = (item.get("peak_memory_bytes"),
                                    item.get("peak_reserved_memory_bytes"), item.get("step_time_seconds"),
                                    item.get("throughput_samples_per_second"))
                if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                       for value in always_numeric):
                    failures.append("measurement_nonpositive_or_nonfinite")
                if item.get("oom"):
                    if any(value is not None for value in measured_numeric) or not isinstance(
                        item.get("oom_evidence"), Mapping
                    ):
                        failures.append("oom_measurement_contract")
                    if not isinstance(item.get("per_rank_peak_memory_bytes"), list) \
                            or len(item["per_rank_peak_memory_bytes"]) != 4:
                        failures.append("oom_rank_evidence")
                    else:
                        try:
                            validate_rank_failure_rows(item["per_rank_peak_memory_bytes"])
                        except ValueError:
                            failures.append("oom_rank_evidence")
                elif any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                         for value in measured_numeric):
                    failures.append("measurement_nonpositive_or_nonfinite")
                if not item.get("oom") and (
                    not isinstance(item.get("forward_backward_steps"), int)
                    or item["forward_backward_steps"] < 1
                ):
                    failures.append("measurement_no_forward_backward")
                if item.get("world_size") != 4:
                    failures.append("profile_world_size")
                if item.get("distributed_strategy") not in {"ddp", "fsdp"}:
                    failures.append("profile_strategy")
                contract = item.get("normalized_execution_contract")
                if (not isinstance(contract, Mapping)
                        or item.get("normalized_execution_contract_sha256")
                           != stable_sha256(contract)
                        or contract != top_contract
                        or item.get("normalized_execution_contract_sha256") != top_contract_sha):
                    failures.append("profile_execution_contract")
                per_rank = item.get("per_rank_peak_memory_bytes")
                if not item.get("oom") and (
                    not isinstance(per_rank, list) or len(per_rank) != 4
                    or [rank.get("rank") for rank in per_rank] != [0, 1, 2, 3]
                    or any(not isinstance(rank.get("peak_allocated_bytes"), int)
                           or rank["peak_allocated_bytes"] <= 0
                           or not isinstance(rank.get("peak_reserved_bytes"), int)
                           or rank["peak_reserved_bytes"] <= 0
                           or not isinstance(rank.get("total_memory_bytes"), int)
                           or rank["total_memory_bytes"] <= 0 for rank in per_rank)
                    or any("NVIDIA H20" not in str(rank.get("device_name")) for rank in per_rank)
                ):
                    failures.append("per_rank_h20_cuda_peak")
                elif (not item.get("oom") and item.get("finite") is True
                      and all(rank["peak_allocated_bytes"] < rank["total_memory_bytes"] * 0.90
                              for rank in per_rank)):
                    feasible.append(item["distributed_strategy"])
                elif not item.get("oom") and item.get("finite") is not True:
                    failures.append("profile_nonfinite")
            if isinstance(measurements, list) and {
                (item.get("distributed_strategy"), item.get("frame_count")) for item in measurements
            } != {("ddp", 32), ("fsdp", 32)}:
                failures.append("profile_strategy_frame_matrix")
            if not feasible:
                failures.append("no_safe_feasible_configuration")
        recommendation = result.get("recommendation", {})
        if recommendation.get("status") != "provisional_not_frozen":
            failures.append("recommendation_not_provisional")
        if recommendation.get("selection_rule") != (
            "max_throughput_then_min_max_rank_allocated_then_strategy_lexical_v1"
        ):
            failures.append("recommendation_selection_rule")
        if isinstance(measurements, list):
            safe = {
                item.get("distributed_strategy") for item in measurements
                if not item.get("oom")
                and item.get("finite") is True
                and isinstance(item.get("per_rank_peak_memory_bytes"), list)
                and all(rank.get("peak_allocated_bytes", float("inf"))
                        < rank.get("total_memory_bytes", 0) * 0.90
                        for rank in item["per_rank_peak_memory_bytes"])
            }
            if recommendation.get("selected_strategy") not in safe:
                failures.append("recommendation_not_safe")
            safe_items = [item for item in measurements
                          if item.get("distributed_strategy") in safe]
            selected = min(
                safe_items,
                key=lambda item: (-item["throughput_samples_per_second"],
                                  item["peak_memory_bytes"], item["distributed_strategy"]),
            ) if safe_items else None
            if recommendation.get("selected_strategy") != (
                selected.get("distributed_strategy") if selected else None
            ):
                failures.append("recommendation_not_deterministic")
    else:
        raise ValueError(f"unknown phase: {phase}")
    return sorted(set(failures))


def run_phase(
    command: PhaseCommand,
    *,
    phase_dir: str | Path,
    defaults: ProvisionalGateDefaults,
) -> dict[str, Any]:
    """Run one phase and atomically publish its terminal status."""
    command.validate()
    directory = Path(phase_dir)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.time()
    started_ns = time.time_ns()
    run_id = uuid.uuid4().hex
    report_path = directory / "phase_report.json"
    contract_path = directory / "producer_contract.json"
    if report_path.exists():
        raise FileExistsError(f"private phase report already exists: {report_path}")
    manifest = Path(command.manifest_path)
    provenance = Path(command.provenance_path)
    producer = Path(command.producer_path)
    for artifact in (manifest, provenance, producer):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    expanded_argv = tuple(
        item.replace("{report_path}", str(report_path)).replace("{contract_path}", str(contract_path))
        for item in command.argv
    )
    command_sha = stable_sha256(list(expanded_argv))
    producer_contract = {
        "schema_version": "parta_gate_producer_contract_v1",
        "run_id": run_id,
        "phase": command.phase,
        "started_unix_ns": started_ns,
        "command_sha256": command_sha,
        "resolved_config_sha256": stable_sha256(command.resolved_config),
        "resolved_config": dict(command.resolved_config),
        "manifest_sha256": sha256_file(manifest),
        "manifest_path": str(manifest.resolve()),
        "provenance_sha256": sha256_file(provenance),
        "producer_sha256": sha256_file(producer),
    }
    atomic_json_dump(producer_contract, contract_path)
    running = {
        "schema_version": "parta_pretrain_phase_status_v1",
        "phase": command.phase,
        "status": "running",
        "argv": list(expanded_argv),
        "command_sha256": command_sha,
        "report_path": str(report_path),
        "producer_contract_sha256": sha256_file(contract_path),
        "manifest_sha256": producer_contract["manifest_sha256"],
        "started_unix_seconds": started,
    }
    atomic_json_dump(running, directory / "status.json")
    log_path = directory / "console.log"
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(expanded_argv, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failures = [f"exit_code={completed.returncode}"]
            report_sha = None
        else:
            if not report_path.is_file():
                failures = ["missing_report"]
                report_sha = None
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                identity = report.get("producer_contract")
                failures = [] if identity == producer_contract else ["producer_contract_mismatch"]
                if report_path.stat().st_mtime_ns < started_ns:
                    failures.append("stale_report_mtime")
                failures.extend(validate_phase_report(command.phase, report, defaults))
                report_sha = sha256_file(report_path)
        status = "complete_passed" if not failures else "complete_failed"
        terminal = {
            **running,
            "status": status,
            "exit_code": completed.returncode,
            "failures": failures,
            "report_sha256": report_sha,
            "finished_unix_seconds": time.time(),
        }
        atomic_json_dump(terminal, directory / "status.json")
        return terminal
    except BaseException as error:
        atomic_json_dump(
            {
                **running,
                "status": "complete_failed",
                "failures": [f"{type(error).__name__}: {error}"],
                "finished_unix_seconds": time.time(),
            },
            directory / "status.json",
        )
        raise


def unified_gate_report(
    phase_statuses: Sequence[Mapping[str, Any]], defaults: ProvisionalGateDefaults,
    *, frozen_config_artifact: Mapping[str, Any] | None = None,
    prior_unified_gate_sha256: str | None = None,
) -> dict[str, Any]:
    by_phase = {str(item.get("phase")): item for item in phase_statuses}
    missing = [phase for phase in PHASES if phase not in by_phase]
    failed = [
        phase for phase in PHASES
        if phase in by_phase and by_phase[phase].get("status") != "complete_passed"
    ]
    passed = not missing and not failed
    profile_report_sha = by_phase.get("resource_profile", {}).get("report_sha256")
    frozen = False
    if frozen_config_artifact:
        record_path = Path(str(frozen_config_artifact.get("user_gate_record_path", "")))
        record_valid = record_path.is_absolute() and record_path.is_file()
        if record_valid:
            record_valid = (
                sha256_file(record_path) == frozen_config_artifact.get("user_gate_record_sha256")
                and "Gate@CONFIG: APPROVE" in record_path.read_text(encoding="utf-8")
            )
        frozen = bool(
            frozen_config_artifact.get("schema_version") == "parta_formal_config_freeze_v1"
            and frozen_config_artifact.get("status") == "frozen"
            and frozen_config_artifact.get("generated_by") == "freeze_pretrain_config.py"
            and frozen_config_artifact.get("profile_report_sha256") == profile_report_sha
            and frozen_config_artifact.get("unified_gate_sha256") == prior_unified_gate_sha256
            and frozen_config_artifact.get("phase_status_sha256") == {
                phase: stable_sha256(dict(by_phase[phase])) for phase in PHASES
            }
            and frozen_config_artifact.get("manifest_sha256") == sorted({
                by_phase[phase].get("manifest_sha256") for phase in PHASES
            })
            and record_valid
            and isinstance(frozen_config_artifact.get("resolved_training_config_path"), str)
            and Path(frozen_config_artifact["resolved_training_config_path"]).is_file()
            and sha256_file(frozen_config_artifact["resolved_training_config_path"])
                == frozen_config_artifact.get("resolved_training_config_sha256")
        )
    return {
        "schema_version": "parta_unified_pretrain_gate_v1",
        "status": "complete_passed" if passed else "complete_failed",
        "formal_gpu_evidence": passed,
        "phase_order": list(PHASES),
        "missing_phases": missing,
        "failed_phases": failed,
        "phase_status_sha256": {
            phase: stable_sha256(dict(by_phase[phase])) for phase in PHASES if phase in by_phase
        },
        "defaults": asdict(defaults),
        "formal_config_frozen": frozen,
        "frozen_config_artifact_sha256": (
            stable_sha256(dict(frozen_config_artifact)) if frozen_config_artifact else None
        ),
        "training_authorized_by_this_artifact": passed and frozen,
    }
