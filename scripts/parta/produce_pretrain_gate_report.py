#!/usr/bin/env python3
"""Produce provenance-bound reports from concrete Part A GPU artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.gate_orchestration import (  # noqa: E402
    ENGINEERING_COVERAGE_REQUIREMENTS,
    FORMAL_SOURCE_REGISTRY,
    PHASES,
)
from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from parta.t0 import T0_A_REQUIRED_CHECKS  # noqa: E402
from parta.unified_data import FROZEN_SOURCE_INVENTORY, FROZEN_TOTAL_INVENTORY  # noqa: E402
from parta.resource_profile_contract import (LAMBDA_STATE,
    normalize_profile_matched_execution, normalize_profile_worker_argv,
    normalized_contract_sha256,
    validate_rank_failure_rows)  # noqa: E402
from parta.resource_profile_contract import validate_preexecution_profile  # noqa: E402
from parta.resource_profile_contract import validate_resolved_profile  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _training_evidence(root: Path, contract: dict, *, expected_arm: str) -> dict:
    status = _json(root / "run_status.json")
    completion = _json(root / "completion.json")
    resolved = _json(root / "resolved_config.json")
    steps = _jsonl(root / "train_steps.jsonl")
    if status.get("schema_version") != "parta_training_provenance_v1":
        raise ValueError("unexpected training provenance schema")
    if status.get("status") != "complete" or completion.get("status") != "complete":
        raise ValueError(f"training run is not complete: {root}")
    if status.get("arm") != expected_arm or resolved.get("arm") != expected_arm:
        raise ValueError(f"training arm mismatch: expected {expected_arm}")
    if not steps or not str(resolved.get("effective_device", "")).startswith("cuda"):
        raise ValueError(f"training run lacks real CUDA steps: {root}")
    if any(item.get("peak_cuda_memory_bytes") is None for item in steps):
        raise ValueError(f"training run lacks CUDA memory evidence: {root}")
    if completion.get("global_step") != len(steps) or [row.get("step") for row in steps] != list(
        range(1, len(steps) + 1)
    ):
        raise ValueError("optimizer-step evidence is not contiguous")
    checkpoint = Path(str(completion.get("checkpoint_path", ""))).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != completion.get("checkpoint_sha256"):
        raise ValueError("selected checkpoint content does not match completion provenance")
    selection_path = root / "checkpoint_selection.json"
    selection = _json(selection_path)
    if (completion.get("checkpoint_role") != "selected_validation"
            or selection.get("selected", {}).get("checkpoint_sha256") != sha256_file(checkpoint)
            or int(selection.get("selected", {}).get("step", -1))
               != int(completion.get("selected_step", -2))):
        raise ValueError("formal completion is not the validation-selected checkpoint")
    expected = contract["resolved_config"]
    artifacts = status.get("artifacts", {})
    if artifacts.get("manifest", {}).get("sha256") != contract["manifest_sha256"]:
        raise ValueError("training manifest differs from signed phase manifest")
    if artifacts.get("guide", {}).get("artifact_sha256") != expected.get("guide_artifact_sha256"):
        raise ValueError("training did not use the expected GUIDE artifact")
    if artifacts.get("vggt", {}).get("artifact_sha256") != expected.get("vggt_artifact_sha256"):
        raise ValueError("training did not use the expected VGGT artifact")
    exact_inputs = artifacts.get("exact_canonical_inputs", {})
    required_sources = set(expected.get("source_registry", ()))
    if required_sources != set(FORMAL_SOURCE_REGISTRY):
        raise ValueError("formal GPU gate requires the exact D-62 three-source registry")
    if set(exact_inputs) != required_sources:
        raise ValueError("training provenance lacks the exact three-source input registry")
    manifest_report_path = Path(str(resolved.get("manifest_report", ""))).resolve()
    if not manifest_report_path.is_file():
        raise ValueError("training resolved config lacks its signed manifest report")
    signed_manifest_report = _json(manifest_report_path)
    signed_registry = signed_manifest_report.get("exact_canonical_inputs")
    signed_registry_sha = signed_manifest_report.get("exact_canonical_inputs_registry_sha256")
    contract_registry_sha = contract["resolved_config"].get(
        "exact_canonical_inputs_registry_sha256", signed_registry_sha
    )
    if (
        exact_inputs != signed_registry
        or signed_registry_sha != stable_sha256(signed_registry)
        or signed_registry_sha != contract_registry_sha
        or artifacts.get("manifest_report", {}).get("sha256") != sha256_file(manifest_report_path)
    ):
        raise ValueError("training provenance exact-input registry differs from signed manifest report")
    manifest_rows = _jsonl(Path(contract["manifest_path"]))
    manifest_qa_ids = {str(row["qa_id"]) for row in manifest_rows}
    bindings = set()
    for source in FORMAL_SOURCE_REGISTRY:
        qa_record = signed_registry[source]["files"]["qa_manifest_exact_verified.jsonl"]
        for row in _jsonl(Path(qa_record["path"])):
            if str(row.get("qa_id")) in manifest_qa_ids:
                bindings.add(row.get("frame_binding_sha256"))
    observed_bindings = {
        binding for row in steps for binding in row.get("frame_binding_sha256", ())
    }
    if not observed_bindings or not observed_bindings.issubset(bindings):
        raise ValueError("training log exact-frame bindings are not in signed manifest")
    return {
        "root": str(root.resolve()), "status": status, "completion": completion,
        "resolved": resolved, "steps": steps,
        "run_status_sha256": sha256_file(root / "run_status.json"),
        "completion_sha256": sha256_file(root / "completion.json"),
        "training_log_sha256": sha256_file(root / "train_steps.jsonl"),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "exact_frame_bindings": sorted(observed_bindings),
    }


def _gpu_evidence() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("formal producer requires CUDA")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {"device_index": index, "device_name": properties.name,
            "total_memory_bytes": properties.total_memory, "cuda_available": True}


def _engineering_coverage(args, contract):
    receipt = _json(args.engineering_coverage_receipt)
    if receipt.get("schema_version") != "parta_engineering_coverage_v1":
        raise ValueError("unexpected engineering coverage schema")
    if receipt.get("status") != "complete_passed":
        raise ValueError("engineering coverage receipt is not passed")
    if tuple(receipt.get("source_registry", ())) != FORMAL_SOURCE_REGISTRY:
        raise ValueError("engineering coverage does not bind the D-62 source registry")
    if receipt.get("manifest_sha256") != contract.get("manifest_sha256"):
        raise ValueError("engineering coverage manifest differs from signed contract")
    matrix = receipt.get("coverage_matrix", {})
    if set(matrix) != set(ENGINEERING_COVERAGE_REQUIREMENTS):
        raise ValueError("engineering coverage matrix is incomplete or has unknown checks")
    expected_schemas = {
        "t0_a_final_commit": "parta_t0_report_v1",
        "three_source_t0_b": "parta_t0_b_report_v1",
        "a1o_fixed_train_subset_learnability": "parta_engineering_runner_receipt_v1",
        "matched_a0_a1o_real_runner_steps": "parta_matched_engineering_receipt_v1",
        "checkpoint_save_resume": "parta_checkpoint_resume_audit_v1",
        "a1o_drop_head_free_val_load": "parta_head_free_load_audit_v1",
        "validator": "parta_validation_report_v1",
        "resource_preflight": "parta_resource_preflight_v1",
        "provenance": "parta_engineering_provenance_audit_v1",
        "engineering_state_non_promotable": "parta_non_promotable_audit_v1",
        "formal_startup_step0": "parta_formal_startup_v1",
        "engineering_subset_not_extra_weighted": "parta_engineering_subset_v1",
    }
    current_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    for name, item in matrix.items():
        artifact = Path(str(item.get("artifact_path", ""))).resolve()
        if item.get("passed") is not True or not artifact.is_file():
            raise ValueError(f"engineering coverage item is not passed: {name}")
        if item.get("artifact_sha256") != sha256_file(artifact):
            raise ValueError(f"engineering coverage artifact hash mismatch: {name}")
        payload = _json(artifact)
        if payload.get("schema_version") != expected_schemas[name]:
            raise ValueError(f"engineering coverage artifact schema mismatch: {name}")
        if name == "t0_a_final_commit":
            checks = payload.get("checks", {})
            if (payload.get("status") != "complete_passed"
                    or set(checks) != set(T0_A_REQUIRED_CHECKS)
                    or any(checks[check].get("passed") is not True for check in T0_A_REQUIRED_CHECKS)
                    or item.get("code_revision") != current_revision):
                raise ValueError("T0-A is not a current-code complete hard-gate PASS")
        if name == "three_source_t0_b" and (
            payload.get("status") != "complete_passed"
            or tuple(payload.get("expected_sources", ())) != FORMAL_SOURCE_REGISTRY
            or payload.get("formal_gpu_evidence") is not True
            or payload.get("runtime_status") != "gpu_complete"
        ):
            raise ValueError("T0-B coverage is not an exact three-source PASS")
        if name == "a1o_fixed_train_subset_learnability":
            if (payload.get("status") != "complete"
                    or payload.get("arm") != "a1o"
                    or payload.get("promotable") is not False
                    or payload.get("engineering_mode") != "overfit"
                    or payload.get("all_losses_finite") is not True
                    or payload.get("optimizer_step_indices")
                       != list(range(1, int(payload.get("optimizer_steps", 0)) + 1))
                    or float(payload.get("final_state_loss", float("inf")))
                    >= float(payload.get("initial_state_loss", float("-inf")))):
                raise ValueError("A1-O engineering subset did not demonstrate learnability")
        if name == "matched_a0_a1o_real_runner_steps":
            arms = payload.get("arms", {})
            matched = payload.get("matched_fields", {})
            if (payload.get("status") != "complete_passed" or set(arms) != {"a0", "a1o"}
                    or payload.get("promotable") is not False
                    or any(arms[arm].get("engineering_mode") != "matched_runner" for arm in arms)
                    or any(arms[arm].get("promotable") is not False for arm in arms)
                    or any(arms[arm].get("all_losses_finite") is not True for arm in arms)
                    or any(int(arms[arm].get("optimizer_steps", 0)) < 1 for arm in arms)
                    or any(arms[arm].get("optimizer_step_indices")
                           != list(range(1, int(arms[arm].get("optimizer_steps", 0)) + 1))
                           for arm in arms)
                    or any(not arms[arm].get("frame_binding_sha256") for arm in arms)
                    or any(arms[arm].get("manifest_sha256") != matched.get("manifest_sha256") for arm in arms)
                    or any(arms[arm].get("engineering_subset_sha256")
                           != matched.get("engineering_subset_sha256") for arm in arms)):
                raise ValueError("matched engineering arms lack real matched contiguous identities")
        if name == "checkpoint_save_resume" and not (
            payload.get("status") == "complete_passed"
            and payload.get("model_state_equivalent") is True
            and payload.get("optimizer_state_equivalent") is True
            and payload.get("scheduler_state_equivalent") is True
            and payload.get("sampler_state_equivalent") is True
            and payload.get("rng_state_equivalent") is True
        ):
            raise ValueError("checkpoint save/resume equivalence is incomplete")
        if name == "a1o_drop_head_free_val_load" and (
            payload.get("status") != "complete_passed" or payload.get("fixture_split") != "val"
            or payload.get("independent_model_construction") is not True
            or payload.get("forward_passed") is not True
            or payload.get("missing_keys") or payload.get("unexpected_keys")
        ):
            raise ValueError("head-free audit is not a passed val-fixture load")
        if name == "validator":
            counts = payload.get("source_counts", {})
            observed = {
                source: {"qa": int(counts.get(source, {}).get("qa", -1)),
                         "scenes": int(counts.get(source, {}).get("scenes", -1))}
                for source in FORMAL_SOURCE_REGISTRY
            }
            if (set(counts) != set(FORMAL_SOURCE_REGISTRY)
                    or observed != {key: dict(value) for key, value in FROZEN_SOURCE_INVENTORY.items()}
                    or int(payload.get("qa", -1)) != FROZEN_TOTAL_INVENTORY["qa"]
                    or int(payload.get("scenes", -1)) != FROZEN_TOTAL_INVENTORY["scenes"]):
                raise ValueError("validator does not bind the exact frozen three-source inventory")
        if name == "resource_preflight" and not (
            payload.get("status") == "complete_passed"
            and payload.get("passed") is True and not payload.get("failures")
        ):
            raise ValueError("resource preflight is not a clean PASS")
        if name == "provenance" and not (
            payload.get("status") == "complete_passed"
            and tuple(payload.get("source_registry", ())) == FORMAL_SOURCE_REGISTRY
            and payload.get("manifest_sha256") == contract.get("manifest_sha256")
            and payload.get("code_revision") == current_revision
        ):
            raise ValueError("engineering provenance is not current and fully bound")
        if name == "engineering_state_non_promotable":
            states = payload.get("states", {})
            if (payload.get("status") != "complete_passed"
                    or not states or any(value.get("promotable") is not False for value in states.values())
                    or set(states) != {"model", "optimizer", "scheduler", "rng", "sampler"}):
                raise ValueError("not every engineering state is explicitly non-promotable")
        if name == "formal_startup_step0":
            arms = payload.get("arms", {})
            if (payload.get("status") != "complete_passed" or set(arms) != {"a0", "a1o"}
                    or any(arms[arm].get("start_step") != 0 for arm in arms)
                    or len({arms[arm].get("initialization_sha256") for arm in arms}) != 1):
                raise ValueError("formal A0/A1-O startup is not identical initialization at step 0")
        if name == "engineering_subset_not_extra_weighted" and not (
            payload.get("split") == "train"
            and payload.get("extra_weight") is False
            and payload.get("transaction_promotable") is False
            and payload.get("artifact_sha256") == sha256_file(Path(payload["artifact_path"]))
        ):
            raise ValueError("engineering subset lifecycle/hash is invalid")
    subset = receipt.get("engineering_subset", {})
    subset_artifact = Path(str(subset.get("artifact_path", ""))).resolve()
    if (
        subset.get("split") != "train"
        or subset.get("extra_weight") is not False
        or subset.get("transaction_promotable") is not False
        or not subset_artifact.is_file()
        or subset.get("artifact_sha256") != sha256_file(subset_artifact)
    ):
        raise ValueError("engineering subset lifecycle is not D-62 compliant")
    return {
        "coverage_matrix": {
            name: {"passed": True, "artifact_sha256": item["artifact_sha256"]}
            for name, item in sorted(matrix.items())
        },
        "engineering_subset": dict(subset),
        "receipt_sha256": sha256_file(args.engineering_coverage_receipt),
    }


def _profile(args, contract):
    receipt = _json(args.profile_receipt)
    if receipt.get("schema_version") != "parta_resource_profile_v2":
        raise ValueError("resource profile must come from the independent v2 transaction")
    if receipt.get("transaction_kind") != "engineering_resource_profile":
        raise ValueError("resource profile is not an independent engineering transaction")
    if receipt.get("formal_training") is not False or receipt.get("promotable") is not False:
        raise ValueError("resource profile transaction must be non-formal and non-promotable")
    if receipt.get("required_world_size") != 4:
        raise ValueError("resource profile must use the frozen four-rank contract")
    if tuple(receipt.get("source_registry", ())) != FORMAL_SOURCE_REGISTRY:
        raise ValueError("resource profile does not bind the D-62 source registry")
    if receipt.get("manifest_sha256") != contract.get("manifest_sha256"):
        raise ValueError("resource profile manifest differs from signed contract")
    runner = receipt.get("train_runner", {})
    canonical_runner = (PROJECT / "scripts/parta/train_parta.py").resolve()
    current_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    profile_producer = (PROJECT / "scripts/parta/run_resource_profile.py").resolve()
    if (Path(str(runner.get("path", ""))).resolve() != canonical_runner
            or runner.get("sha256") != sha256_file(canonical_runner)
            or receipt.get("producer_git_revision") != current_revision
            or receipt.get("producer") != {"path": str(profile_producer),
                "sha256": sha256_file(profile_producer), "git_revision": current_revision}):
        raise ValueError("resource profile is not bound to the current canonical train runner")
    measurements = receipt.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError("resource profile lacks measurements")
    if {(item.get("distributed_strategy"), item.get("frame_count")) for item in measurements} \
            != {("ddp", 32), ("fsdp", 32)}:
        raise ValueError("resource profile must contain DDP and FSDP 32-frame points")
    preflight_normalized = {}
    runtime_matched_normalized = {}
    for item in measurements:
        required = {
            "frame_count", "peak_memory_bytes", "total_memory_bytes",
            "step_time_seconds", "throughput_samples_per_second", "oom",
            "batch_size", "gradient_accumulation_steps", "forward_backward_steps",
            "world_size", "per_rank_peak_memory_bytes",
            "peak_reserved_memory_bytes", "distributed_strategy", "finite",
        }
        if not required.issubset(item):
            raise ValueError("resource profile measurement is incomplete")
        artifacts = item.get("artifacts", {})
        required_artifacts = ({"command_record", "oom_rank_evidence", "preflight_matched_contract"} if item["oom"]
                              else {"command_record", "run_status", "resolved_config",
                                    "engineering_receipt", "train_steps", "matched_contract",
                                    "preflight_matched_contract"})
        if not required_artifacts.issubset(artifacts) or (
            not item["oom"] and set(artifacts) != required_artifacts
        ):
            raise ValueError("resource profile evidence paths are incomplete")
        for evidence in artifacts.values():
            path = Path(str(evidence.get("path", ""))).resolve()
            if not path.is_file() or evidence.get("sha256") != sha256_file(path):
                raise ValueError("resource profile evidence hash mismatch")
        command = _json(Path(artifacts["command_record"]["path"]))
        normalized = normalize_profile_worker_argv(command.get("argv", ()))
        if (item.get("normalized_execution_contract") != normalized
                or item.get("normalized_execution_contract_sha256")
                   != normalized_contract_sha256(normalized)
                or float(normalized["lambda_state"]) != LAMBDA_STATE):
            raise ValueError("resource profile command/measurement contract mismatch")
        preflight = _json(Path(artifacts["preflight_matched_contract"]["path"]))
        validate_preexecution_profile(preflight, command.get("argv", ()),
            manifest=normalized["manifest"], manifest_report=normalized["manifest_report"],
            engineering_subset=normalized["engineering_subset"])
        if preflight.get("distributed_strategy") != item["distributed_strategy"]:
            raise ValueError("resource profile preflight strategy mismatch")
        preflight_normalized[item["distributed_strategy"]] = {
            key: value for key, value in preflight.items() if key != "distributed_strategy"
        }
        matched = (_json(Path(artifacts["matched_contract"]["path"]))
                   if "matched_contract" in artifacts else None)
        if matched is not None:
            execution = normalize_profile_matched_execution(
                matched.get("execution_contract", {}), item["distributed_strategy"]
            )
            runtime_matched_normalized[item["distributed_strategy"]] = {
                **matched, "execution_contract": execution
            }
        status = (_json(Path(artifacts["run_status"]["path"]))
                  if "run_status" in artifacts else {})
        if not item["oom"]:
            validate_resolved_profile(_json(Path(artifacts["resolved_config"]["path"])),
                                      normalized, item["distributed_strategy"])
        if (Path(str(command.get("script_path", ""))).resolve() != canonical_runner
                or command.get("script_sha256") != sha256_file(canonical_runner)
                or command.get("git_revision") != current_revision):
            raise ValueError("resource profile command record is not canonical")
        if item["oom"]:
            rank_evidence = _json(Path(artifacts["oom_rank_evidence"]["path"]))
            try:
                validate_rank_failure_rows(rank_evidence.get("ranks", ()))
            except ValueError:
                raise ValueError("resource profile OOM evidence cannot be reopened")
        if item["oom"]:
            if (item.get("peak_memory_bytes") is not None
                    or item.get("step_time_seconds") is not None
                    or item.get("throughput_samples_per_second") is not None
                    or not isinstance(item.get("oom_evidence"), dict)):
                raise ValueError("OOM profile point must use nullable metrics and explicit evidence")
        elif (int(item["forward_backward_steps"]) < 1
              or item.get("peak_memory_bytes") is None
              or item.get("step_time_seconds") is None
              or item.get("throughput_samples_per_second") is None):
            raise ValueError("non-OOM profile point lacks real measured forward/backward metrics")
        if not item["oom"]:
            per_rank = item.get("per_rank_peak_memory_bytes")
            if (item.get("world_size") != 4
                    or not isinstance(per_rank, list) or len(per_rank) != 4
                    or [rank.get("rank") for rank in per_rank] != [0, 1, 2, 3]
                    or any(not isinstance(rank.get("peak_allocated_bytes"), int)
                           or rank["peak_allocated_bytes"] <= 0
                           or not isinstance(rank.get("peak_reserved_bytes"), int)
                           or rank["peak_reserved_bytes"] <= 0
                           or not isinstance(rank.get("total_memory_bytes"), int)
                           or rank["total_memory_bytes"] <= 0 for rank in per_rank)
                    or any("NVIDIA H20" not in str(rank.get("device_name")) for rank in per_rank)):
                raise ValueError("resource profile lacks four-rank H20 CUDA peak evidence")
            if item.get("finite") is not True:
                raise ValueError("resource profile successful point is non-finite")
    if (set(preflight_normalized) != {"ddp", "fsdp"}
            or preflight_normalized["ddp"] != preflight_normalized["fsdp"]):
        raise ValueError("profile preflight contracts differ beyond strategy")
    if set(runtime_matched_normalized) == {"ddp", "fsdp"} and \
            runtime_matched_normalized["ddp"] != runtime_matched_normalized["fsdp"]:
        raise ValueError("profile runtime matched contracts differ beyond strategy")
    safe = [item for item in measurements if not item["oom"] and item.get("finite") is True
            and all(rank["peak_allocated_bytes"] < rank["total_memory_bytes"] * 0.90
                    for rank in item["per_rank_peak_memory_bytes"])]
    selected = min(
        safe,
        key=lambda item: (-item["throughput_samples_per_second"],
                          item["peak_memory_bytes"], item["distributed_strategy"]),
    ) if safe else None
    selected_strategy = selected["distributed_strategy"] if selected else None
    if receipt.get("selected_strategy") != selected_strategy:
        raise ValueError("resource profile receipt selection is non-deterministic")
    return {
        "measurements": measurements,
        "normalized_execution_contract": receipt.get("normalized_execution_contract"),
        "normalized_execution_contract_sha256": receipt.get(
            "normalized_execution_contract_sha256"
        ),
        "recommendation": {
            "status": "provisional_not_frozen", "frame_count": 32,
            "selected_strategy": selected["distributed_strategy"] if selected else None,
            "selection_rule": (
                "max_throughput_then_min_max_rank_allocated_then_strategy_lexical_v1"
            ),
        },
        "independent_profile_receipt_sha256": sha256_file(args.profile_receipt),
    }


def _engineering_coverage_v2(args, contract):
    receipt = _json(args.engineering_coverage_receipt)
    builder = (PROJECT / "scripts/parta/build_engineering_coverage.py").resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if (receipt.get("schema_version") != "parta_engineering_coverage_v2"
            or receipt.get("status") != "complete_passed"
            or tuple(receipt.get("source_registry", ())) != FORMAL_SOURCE_REGISTRY
            or Path(str(receipt.get("producer", {}).get("path", ""))).resolve() != builder
            or receipt.get("producer", {}).get("sha256") != sha256_file(builder)
            or receipt.get("producer", {}).get("git_revision") != revision):
        raise ValueError("coverage must be the current repository-authoritative v2 producer artifact")
    matrix = receipt.get("coverage_matrix", {})
    if set(matrix) != set(ENGINEERING_COVERAGE_REQUIREMENTS):
        raise ValueError("coverage v2 matrix is incomplete")
    for name, item in matrix.items():
        path = Path(str(item.get("artifact_path", ""))).resolve()
        if not path.is_file() or item.get("artifact_sha256") != sha256_file(path):
            raise ValueError(f"coverage v2 evidence changed: {name}")
    subset = receipt.get("engineering_subset", {})
    subset_path = Path(str(subset.get("path", ""))).resolve()
    subset_payload = _json(subset_path)
    resolved_contract = contract.get("resolved_config", {})
    manifest_report_path = Path(str(resolved_contract.get(
        "manifest_report_path", resolved_contract.get("manifest_report", "")
    ))).resolve()
    manifest_report = _json(manifest_report_path)
    frozen = manifest_report.get("engineering_subset", {})
    if (subset.get("sha256") != sha256_file(subset_path)
            or Path(str(frozen.get("path", ""))).resolve() != subset_path
            or frozen.get("sha256") != subset["sha256"]
            or subset.get("formal_train_reuse") != subset_payload.get("formal_train_reuse")
            or subset.get("transaction_promotion") != subset_payload.get("transaction_promotion")
            or subset_payload.get("formal_train_reuse", {}).get("extra_sampling_weight") is not False
            or subset_payload.get("transaction_promotion", {}).get(
                "promotable_to_formal_training"
            ) is not False):
        raise ValueError("coverage v2 subset is not the manifest-anchored authoritative artifact")
    if (receipt.get("manifest_report_sha256") != sha256_file(manifest_report_path)
            or receipt.get("exact_registry_sha256")
               != stable_sha256(manifest_report["exact_canonical_inputs"])):
        raise ValueError("coverage v2 exact registry identity mismatch")
    # Re-open the two most consequential hard gates instead of trusting summaries.
    t0a = _json(Path(matrix["t0_a_final_commit"]["artifact_path"]))
    t0b = _json(Path(matrix["three_source_t0_b"]["artifact_path"]))
    if (t0a.get("status") != "complete_passed"
            or set(t0a.get("checks", {})) != set(T0_A_REQUIRED_CHECKS)
            or any(t0a["checks"][name].get("passed") is not True for name in T0_A_REQUIRED_CHECKS)
            or t0b.get("status") != "complete_passed"
            or t0b.get("source_registry") != list(FORMAL_SOURCE_REGISTRY)
            or any(item.get("passed") is not True for item in t0b.get("checks", {}).values())):
        raise ValueError("coverage v2 embedded T0 hard gates are not passed")
    overfit = _json(Path(matrix["a1o_fixed_train_subset_learnability"]["artifact_path"]))
    overfit_receipt = overfit.get("receipt", {})
    matched = _json(Path(matrix["matched_a0_a1o_real_runner_steps"]["artifact_path"]))
    head = _json(Path(matrix["a1o_drop_head_free_val_load"]["artifact_path"]))
    validator = _json(Path(matrix["validator"]["artifact_path"]))
    nonpromotable = _json(Path(matrix["engineering_state_non_promotable"]["artifact_path"]))
    nonpromotable_producer = (PROJECT / "scripts/parta/audit_engineering_nonpromotable.py").resolve()
    startup_summary = matrix["formal_startup_step0"].get("semantic_summary", {})
    startup = _json(Path(matrix["formal_startup_step0"]["artifact_path"]))
    startup_producer = (PROJECT / "scripts/parta/audit_formal_startup.py").resolve()
    if (overfit.get("status") != "complete_passed"
            or overfit_receipt.get("actual_unique_examples", 0) < 64
            or overfit_receipt.get("optimizer_steps") != 100
            or float(overfit_receipt.get("state_loss_decrease_fraction", -1)) < 0.20
            or matched.get("status") != "complete_passed"
            or matched.get("promotable") is not False
            or head.get("status") != "complete_passed" or head.get("fixture_split") != "val"
            or validator.get("schema_version") != "parta_three_source_validator_audit_v2"
            or validator.get("recomputed_summary", {}).get("total_qa")
               != FROZEN_TOTAL_INVENTORY["qa"]
            or validator.get("recomputed_summary", {}).get("total_scenes")
               != FROZEN_TOTAL_INVENTORY["scenes"]
            or nonpromotable.get("schema_version") != "parta_non_promotable_audit_v1"
            or nonpromotable.get("status") != "complete_passed"
            or Path(str(nonpromotable.get("producer", {}).get("path", ""))).resolve()
               != nonpromotable_producer
            or nonpromotable.get("producer", {}).get("sha256")
               != sha256_file(nonpromotable_producer)
            or nonpromotable.get("producer", {}).get("git_revision")
               != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
            or set(nonpromotable.get("states", {}))
               != {"model", "optimizer", "scheduler", "rng", "sampler"}
            or any(item.get("promotable") is not False
                   for item in nonpromotable.get("states", {}).values())
            or not isinstance(startup_summary.get("initialization_sha256"), str)):
        raise ValueError("coverage v2 producer semantics failed independent gate revalidation")
    if (startup.get("schema_version") != "parta_formal_startup_audit_v1"
            or startup.get("status") != "complete_passed"
            or Path(str(startup.get("producer", {}).get("path", ""))).resolve() != startup_producer
            or startup.get("producer", {}).get("sha256") != sha256_file(startup_producer)
            or set(startup.get("arms", {})) != {"a0", "a1o"}
            or any(arm.get("start_step") != 0 for arm in startup.get("arms", {}).values())):
        raise ValueError("coverage v2 formal startup audit is invalid")
    for name, item in startup.get("inputs", {}).items():
        sys.path.insert(0, str(PROJECT / "scripts/parta"))
        from audit_formal_startup import validate_startup_input
        validate_startup_input(name, item)
    for item in nonpromotable.get("inputs", ()):
        for path_key, hash_key in (("run_status_path", "run_status_sha256"),
                                   ("receipt_path", "receipt_sha256"),
                                   ("checkpoint_path", "checkpoint_sha256")):
            path = Path(str(item.get(path_key, ""))).resolve()
            if not path.is_file() or item.get(hash_key) != sha256_file(path):
                raise ValueError("coverage v2 non-promotion source hash mismatch")
    return {"schema_version": "parta_engineering_coverage_v2",
            "coverage_matrix": matrix, "engineering_subset": subset,
            "receipt_sha256": sha256_file(args.engineering_coverage_receipt)}


def _t0_b(args, contract):
    source = _json(args.t0_b_report)
    if source.get("schema_version") != "parta_t0_b_report_v1":
        raise ValueError("unexpected T0-B report schema")
    if source.get("status") != "complete_passed" or not source.get("formal_gpu_evidence"):
        raise ValueError("T0-B source report is not formal GPU PASS")
    resolved = contract.get("resolved_config", {})
    expected = {
        "manifest_sha256": contract.get("manifest_sha256"),
        "manifest_report_sha256": resolved.get("manifest_report_sha256"),
        "exact_registry_sha256": resolved.get("exact_canonical_inputs_registry_sha256"),
        "source_registry": list(FORMAL_SOURCE_REGISTRY),
    }
    mismatches = [key for key, value in expected.items() if source.get(key) != value]
    if mismatches:
        raise ValueError(f"T0-B identity differs from signed phase contract: {mismatches}")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--engineering-coverage-receipt", type=Path)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--t0-b-report", type=Path)
    args = parser.parse_args()
    contract = _json(args.contract)
    if contract.get("phase") != args.phase:
        raise ValueError("producer phase differs from signed contract")
    if args.phase == "t0_b":
        result = _t0_b(args, contract)
    elif args.phase == "engineering_coverage":
        result = _engineering_coverage_v2(args, contract)
    elif args.phase == "resource_profile":
        result = _profile(args, contract)
    atomic_json_dump({
        "schema_version": f"parta_{args.phase}_report_v1",
        "phase": args.phase, "status": "complete_passed",
        "formal_gpu_evidence": True, "gpu_evidence": _gpu_evidence(),
        "producer_contract": _json(args.contract), "result": result,
    }, args.report)


if __name__ == "__main__":
    main()
