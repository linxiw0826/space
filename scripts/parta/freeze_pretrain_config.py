#!/usr/bin/env python3
"""Create an auditable config-freeze artifact after an explicit user Gate record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from parta.gate_orchestration import FORMAL_SOURCE_REGISTRY, PHASES  # noqa: E402
from parta.checkpoint_selection import RULE as CHECKPOINT_SELECTION_RULE  # noqa: E402


def validate_profile_selected_config(profile_report: dict, resolved_training: dict) -> tuple[str, dict]:
    recommendation = profile_report.get("result", {}).get("recommendation", {})
    selected_strategy = recommendation.get("selected_strategy")
    if selected_strategy not in {"ddp", "fsdp"}:
        raise ValueError("profile report lacks a selected DDP/FSDP strategy")
    if (resolved_training.get("distributed_strategy") != selected_strategy
            or resolved_training.get("world_size") != 4):
        raise ValueError("formal config must use the four-rank profile-selected strategy")
    selected_measurement = next((item for item in profile_report.get("result", {}).get(
        "measurements", ()) if item.get("distributed_strategy") == selected_strategy), None)
    if not isinstance(selected_measurement, dict):
        raise ValueError("profile report lacks selected measurement")
    profile_contract = selected_measurement.get("normalized_execution_contract", {})
    expected_formal = {
        "learning_rate": float(profile_contract["learning_rate"]),
        "weight_decay": float(profile_contract["weight_decay"]),
        "lambda_state": float(profile_contract["lambda_state"]),
        "max_grad_norm": float(profile_contract["max_grad_norm"]),
        "gradient_accumulation_steps": int(profile_contract["gradient_accumulation_steps"]),
        "dtype": profile_contract["dtype"],
        "num_workers": int(profile_contract["num_workers"]),
        "gradient_checkpointing": bool(profile_contract["gradient_checkpointing"]),
        "per_rank_batch_size": 1,
        "effective_global_batch_size": int(profile_contract["effective_global_batch_size"]),
    }
    drift = {key: (resolved_training.get(key), value) for key, value in expected_formal.items()
             if resolved_training.get(key) != value}
    if drift:
        raise ValueError(f"formal config differs from selected profile measurement: {drift}")
    return selected_strategy, expected_formal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-gate", type=Path, required=True)
    parser.add_argument("--phase-status", action="append", type=Path, required=True)
    parser.add_argument("--profile-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolved-training-config", type=Path, required=True)
    parser.add_argument("--user-gate-record", type=Path, required=True)
    parser.add_argument("--formal-startup-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.unified_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "complete_passed" or gate.get("training_authorized_by_this_artifact"):
        raise ValueError("freeze requires a passed but not-yet-authorized unified gate")
    record = args.user_gate_record.resolve()
    if "Gate@CONFIG: APPROVE" not in record.read_text(encoding="utf-8"):
        raise ValueError("user Gate record lacks exact Gate@CONFIG: APPROVE marker")
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in args.phase_status]
    by_phase = {item["phase"]: item for item in statuses}
    required = PHASES
    if set(by_phase) != set(required) or any(by_phase[name].get("status") != "complete_passed" for name in required):
        raise ValueError("freeze requires the complete D-62 mandatory coverage matrix")
    if by_phase["resource_profile"].get("report_sha256") != sha256_file(args.profile_report):
        raise ValueError("profile report does not match its passed phase status")
    profile_report = json.loads(args.profile_report.read_text(encoding="utf-8"))
    resolved_training = json.loads(args.resolved_training_config.read_text(encoding="utf-8"))
    selected_strategy, expected_formal = validate_profile_selected_config(
        profile_report, resolved_training
    )
    recommendation = profile_report["result"]["recommendation"]
    selected_measurement = next((item for item in profile_report.get("result", {}).get(
        "measurements", ()) if item.get("distributed_strategy") == selected_strategy), None)
    if not isinstance(selected_measurement, dict):
        raise ValueError("profile report lacks selected measurement")
    manifest_sha = sha256_file(args.manifest)
    if any(item.get("manifest_sha256") != manifest_sha for item in statuses):
        raise ValueError("phase manifests differ from the config-freeze manifest")
    startup = json.loads(args.formal_startup_contract.read_text(encoding="utf-8"))
    arms = startup.get("arms", {})
    if (
        startup.get("schema_version") != "parta_formal_startup_v1"
        or set(arms) != {"a0", "a1o"}
        or any(item.get("start_step") != 0 for item in arms.values())
        or len({item.get("initialization_sha256") for item in arms.values()}) != 1
    ):
        raise ValueError("invalid A0/A1-O shared-initialization step-0 startup contract")
    payload = {
        "schema_version": "parta_formal_config_freeze_v1", "status": "frozen",
        "generated_by": "freeze_pretrain_config.py",
        "unified_gate_sha256": sha256_file(args.unified_gate),
        "phase_status_sha256": {name: stable_sha256(by_phase[name]) for name in required},
        "profile_report_sha256": sha256_file(args.profile_report),
        "profile_selected_strategy": selected_strategy,
        "profile_selection_rule": recommendation.get("selection_rule"),
        "profile_execution_contract_sha256": selected_measurement.get(
            "normalized_execution_contract_sha256"
        ),
        "profile_bound_training_fields": expected_formal,
        "manifest_sha256": [manifest_sha],
        "resolved_training_config_sha256": sha256_file(args.resolved_training_config),
        "resolved_training_config_path": str(args.resolved_training_config.resolve()),
        "user_gate_record_path": str(record),
        "user_gate_record_sha256": sha256_file(record),
        "formal_source_registry": list(FORMAL_SOURCE_REGISTRY),
        "engineering_lifecycle": {
            "subset_is_train_internal": True,
            "subset_extra_weight": False,
            "model_promotable": False,
            "optimizer_promotable": False,
            "scheduler_promotable": False,
            "rng_promotable": False,
            "sampler_promotable": False,
            "formal_arms_start_at_step": 0,
            "formal_arms_share_initialization": True,
        },
        "formal_startup_contract": startup,
        "formal_startup_contract_sha256": sha256_file(args.formal_startup_contract),
        "checkpoint_selection_contract": {
            "schema_version": "parta_checkpoint_selection_v1",
            "selection_rule": CHECKPOINT_SELECTION_RULE,
            "metric_source": "validation_only",
            "source_balancing": "equal_mean_over_adt_hypersim_scannetppv2",
            "tie_break": "earliest_step",
            "vsibench_used_for_selection": False,
            "required_reports": {"a0": "checkpoint_selection.json",
                                 "a1o": "checkpoint_selection.json"},
        },
    }
    atomic_json_dump(payload, args.output)


if __name__ == "__main__":
    main()
