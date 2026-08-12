#!/usr/bin/env python3
"""Build the unique D-62 engineering coverage v2 from repository producer artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.provenance import sha256_file, stable_sha256  # noqa: E402
from parta.t0 import T0_A_REQUIRED_CHECKS  # noqa: E402
from parta.unified_data import (FROZEN_SOURCE_INVENTORY, FROZEN_SOURCE_REGISTRY,
                                FROZEN_TOTAL_INVENTORY)  # noqa: E402
sys.path.insert(0, str(PROJECT / "scripts/parta"))
from audit_formal_startup import validate_startup_input  # noqa: E402


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(path: Path, summary: dict) -> dict:
    return {"artifact_path": str(path.resolve()), "artifact_sha256": sha256_file(path),
            "semantic_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("t0-a-report", "t0-a-provenance", "t0-a-run-status", "t0-a-resolved-config", "t0-b-report",
                 "t0-b-provenance", "t0-b-run-status", "overfit-audit", "matched-audit", "head-free-audit",
                 "validator-report", "profile-report", "manifest-report", "engineering-subset",
                 "nonpromotable-audit", "formal-startup-audit"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    t0a, t0a_prov = load(args.t0_a_report), load(args.t0_a_provenance)
    t0a_status, t0a_config = load(args.t0_a_run_status), load(args.t0_a_resolved_config)
    checks = t0a.get("checks", {})
    t0a_resolved_sha = stable_sha256(t0a_config)
    if (t0a.get("status") != "complete_passed" or t0a_prov.get("status") != "complete_passed"
            or t0a_status.get("status") != "complete"
            or t0a_prov.get("git_revision") != revision
            or t0a_status.get("code_revision", t0a_status.get("git_revision")) != revision
            or t0a_prov.get("resolved_config_sha256") != t0a_resolved_sha
            or t0a_status.get("resolved_config_sha256") != t0a_resolved_sha
            or Path(str(t0a_prov.get("resolved_config_path", ""))).resolve()
               != args.t0_a_resolved_config.resolve()
            or Path(str(t0a_status.get("resolved_config_path", ""))).resolve()
               != args.t0_a_resolved_config.resolve()
            or t0a_prov.get("a1_checkpoint_state_sha256") != t0a_status.get("checkpoint_sha256")
            or set(checks) != set(T0_A_REQUIRED_CHECKS)
            or any(checks[name].get("passed") is not True for name in T0_A_REQUIRED_CHECKS)):
        raise ValueError("T0-A current-commit evidence is not a complete PASS")
    t0b, t0b_prov = load(args.t0_b_report), load(args.t0_b_provenance)
    t0b_status = load(args.t0_b_run_status)
    t0b_expected = {
        "schema_version": "parta_t0_b_provenance_v1", "run_id": "t0-b-three-source",
        "status": "complete_passed", "git_revision": revision,
        "report_sha256": sha256_file(args.t0_b_report),
        "manifest_sha256": t0b.get("manifest_sha256"),
        "manifest_report_sha256": t0b.get("manifest_report_sha256"),
        "exact_registry_sha256": t0b.get("exact_registry_sha256"),
    }
    if (t0b.get("status") != "complete_passed" or t0b_prov.get("status") != "complete_passed"
            or args.t0_b_provenance.resolve() == args.t0_b_run_status.resolve()
            or any(t0b_prov.get(key) != value for key, value in t0b_expected.items())
            or any(t0b_status.get(key) != value for key, value in t0b_expected.items())
            or t0b_prov != t0b_status
            or t0b.get("source_registry") != list(FROZEN_SOURCE_REGISTRY)
            or any(item.get("passed") is not True for item in t0b.get("checks", {}).values())):
        raise ValueError("T0-B producer evidence is not a current exact-three-source PASS")
    overfit = load(args.overfit_audit)
    overfit_receipt = overfit.get("receipt", {})
    overfit_producer = (PROJECT / "scripts/parta/run_a1o_overfit.py").resolve()
    if (overfit.get("schema_version") != "parta_a1o_overfit_audit_v1"
            or overfit.get("status") != "complete_passed"
            or overfit.get("producer") != {"path": str(overfit_producer),
                "sha256": sha256_file(overfit_producer), "git_revision": revision}
            or overfit_receipt.get("actual_unique_examples", 0) < 64
            or overfit_receipt.get("optimizer_steps") != 100
            or float(overfit_receipt.get("state_loss_decrease_fraction", -1)) < 0.20):
        raise ValueError("A1-O overfit producer evidence failed provisional thresholds")
    overfit_receipt_path = Path(str(overfit.get("run_dir", ""))).resolve() / "engineering_receipt.json"
    if (not overfit_receipt_path.is_file()
            or overfit.get("receipt_sha256") != sha256_file(overfit_receipt_path)
            or overfit_receipt != load(overfit_receipt_path)):
        raise ValueError("A1-O overfit wrapper is not bound to its runner receipt")
    matched = load(args.matched_audit)
    arms = matched.get("arms", {})
    matched_producer = (PROJECT / "scripts/parta/run_matched_engineering.py").resolve()
    if (matched.get("status") != "complete_passed" or set(arms) != {"a0", "a1o"}
            or matched.get("producer") != {"path": str(matched_producer),
                "sha256": sha256_file(matched_producer), "git_revision": revision}
            or any(arms[arm].get("promotable") is not False for arm in arms)
            or any(arms[arm].get("optimizer_step_indices")
                   != list(range(1, int(arms[arm].get("optimizer_steps", 0)) + 1)) for arm in arms)):
        raise ValueError("matched engineering producer evidence is invalid")
    for arm, item in arms.items():
        receipt_path = Path(str(item.get("receipt_path", ""))).resolve()
        raw_receipt = load(receipt_path)
        wrapped = {key: value for key, value in item.items()
                   if key not in {"receipt_path", "receipt_sha256"}}
        if item.get("receipt_sha256") != sha256_file(receipt_path) or wrapped != raw_receipt:
            raise ValueError(f"matched {arm} wrapper is not bound to its runner receipt")
    head = load(args.head_free_audit)
    head_producer = (PROJECT / "scripts/parta/audit_a1o_drop_load.py").resolve()
    if (head.get("status") != "complete_passed" or head.get("fixture_split") != "val"
            or head.get("producer") != {"path": str(head_producer),
                "sha256": sha256_file(head_producer), "git_revision": revision}
            or head.get("forward_passed") is not True or head.get("missing_keys")
            or head.get("unexpected_keys")):
        raise ValueError("head-free val producer evidence failed")
    manifest_report = load(args.manifest_report)
    validator = load(args.validator_report)
    validator_producer = (PROJECT / "scripts/parta/audit_three_source_validator.py").resolve()
    if (validator.get("schema_version") != "parta_three_source_validator_audit_v2"
            or validator.get("status") != "complete_passed"
            or validator.get("producer") != {"path": str(validator_producer),
                "sha256": sha256_file(validator_producer), "git_revision": revision}
            or validator.get("manifest_report", {}).get("sha256") != sha256_file(args.manifest_report)
            or Path(str(validator.get("manifest_report", {}).get("path", ""))).resolve()
               != args.manifest_report.resolve()):
        raise ValueError("validator wrapper is not canonical or manifest-bound")
    manifest_path = Path(str(validator.get("manifest", {}).get("path", ""))).resolve()
    if (not manifest_path.is_file()
            or validator.get("manifest", {}).get("sha256") != sha256_file(manifest_path)):
        raise ValueError("validator wrapper manifest hash mismatch")
    for source, item in validator.get("source_validations", {}).items():
        path = Path(str(item.get("path", ""))).resolve()
        if source not in FROZEN_SOURCE_REGISTRY or not path.is_file() \
                or item.get("sha256") != sha256_file(path):
            raise ValueError("validator wrapper source-validation hash mismatch")
    if set(validator.get("source_validations", {})) != set(FROZEN_SOURCE_REGISTRY):
        raise ValueError("validator wrapper source-validation registry mismatch")
    expected_counts = {key: dict(value) for key, value in FROZEN_SOURCE_INVENTORY.items()}
    summary = validator.get("recomputed_summary", {})
    observed_counts = summary.get("frozen_source_inventory")
    if (observed_counts != expected_counts or summary.get("total_qa") != FROZEN_TOTAL_INVENTORY["qa"]
            or summary.get("total_scenes") != FROZEN_TOTAL_INVENTORY["scenes"]
            or validator.get("exact_registry") != manifest_report.get("exact_canonical_inputs")
            or validator.get("exact_registry_sha256")
               != manifest_report.get("exact_canonical_inputs_registry_sha256")):
        raise ValueError("validator evidence differs from frozen inventory")
    profile = load(args.profile_report)
    profile_producer = (PROJECT / "scripts/parta/run_resource_profile.py").resolve()
    if (profile.get("schema_version") != "parta_resource_profile_v2"
            or profile.get("promotable") is not False
            or profile.get("producer") != {"path": str(profile_producer),
                "sha256": sha256_file(profile_producer), "git_revision": revision}
            or sorted(item.get("frame_count") for item in profile.get("measurements", ())) != [16, 24, 32]):
        raise ValueError("resource profile producer evidence is invalid")
    for measurement in profile["measurements"]:
        artifacts = measurement.get("artifacts", {})
        required = {"command_record", "run_status", "resolved_config"}
        required |= set() if measurement.get("oom") else {"engineering_receipt", "train_steps"}
        if set(artifacts) != required:
            raise ValueError("resource profile artifact registry is incomplete")
        reopened = {}
        for name, item in artifacts.items():
            path = Path(str(item.get("path", ""))).resolve()
            if not path.is_file() or item.get("sha256") != sha256_file(path):
                raise ValueError("resource profile artifact hash mismatch")
            reopened[name] = load(path) if path.suffix == ".json" else None
        command = reopened["command_record"]
        status = reopened["run_status"]
        resolved_point = reopened["resolved_config"]
        argv = command.get("argv", {})
        if (command.get("script_path") != str((PROJECT / "scripts/parta/train_parta.py").resolve())
                or command.get("script_sha256") != sha256_file(PROJECT / "scripts/parta/train_parta.py")
                or command.get("git_revision") != revision
                or measurement.get("frame_count") != int(resolved_point.get("required_frame_count", -1))):
            raise ValueError("resource profile worker identity mismatch")
        if measurement.get("oom"):
            if status.get("status") not in {"failed", "complete_failed"} \
                    or "out of memory" not in str(status.get("error", "")).lower():
                raise ValueError("resource profile OOM evidence is not real")
        else:
            receipt = reopened["engineering_receipt"]
            steps_path = Path(artifacts["train_steps"]["path"])
            steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line]
            if (status.get("status") != "complete"
                    or receipt.get("engineering_mode") != "resource_profile"
                    or receipt.get("promotable") is not False
                    or len(steps) != measurement.get("forward_backward_steps")):
                raise ValueError("resource profile successful point cannot be reopened")
    preflight = t0a_config.get("resource_preflight", {})
    if preflight.get("passed") is not True or preflight.get("failures"):
        raise ValueError("resource preflight did not pass")
    subset = load(args.engineering_subset)
    frozen_subset = manifest_report.get("engineering_subset", {})
    if (Path(str(frozen_subset.get("path", ""))).resolve() != args.engineering_subset.resolve()
            or frozen_subset.get("sha256") != sha256_file(args.engineering_subset)
            or subset.get("formal_train_reuse", {}).get("extra_sampling_weight") is not False
            or subset.get("formal_train_reuse", {}).get("subset_rows_remain_in_train_manifest") is not True
            or subset.get("transaction_promotion", {}).get("promotable_to_formal_training") is not False
            or subset.get("transaction_promotion", {}).get("formal_restart_optimizer_step") != 0):
        raise ValueError("engineering subset is not the manifest-anchored train-only artifact")
    startup = load(args.formal_startup_audit)
    startup_producer = (PROJECT / "scripts/parta/audit_formal_startup.py").resolve()
    arms = startup.get("arms", {})
    if (startup.get("schema_version") != "parta_formal_startup_audit_v1"
            or startup.get("status") != "complete_passed"
            or startup.get("producer") != {"path": str(startup_producer),
                "sha256": sha256_file(startup_producer), "git_revision": revision}
            or set(arms) != {"a0", "a1o"}
            or any(item.get("start_step") != 0 for item in arms.values())
            or len({item.get("initialization_sha256") for item in arms.values()}) != 1):
        raise ValueError("formal startup is not A0/A1-O same initialization at step 0")
    for name, item in startup.get("inputs", {}).items():
        validate_startup_input(name, item)
    if t0b.get("checks", {}).get("checkpoint_resume_equivalence", {}).get("passed") is not True:
        raise ValueError("repository T0-B checkpoint round-trip comparator did not pass")
    nonpromotable = load(args.nonpromotable_audit)
    states = nonpromotable.get("states", {})
    nonpromotable_producer = (PROJECT / "scripts/parta/audit_engineering_nonpromotable.py").resolve()
    if (nonpromotable.get("schema_version") != "parta_non_promotable_audit_v1"
            or nonpromotable.get("status") != "complete_passed"
            or Path(str(nonpromotable.get("producer", {}).get("path", ""))).resolve()
               != nonpromotable_producer
            or nonpromotable.get("producer", {}).get("sha256")
               != sha256_file(nonpromotable_producer)
            or nonpromotable.get("producer", {}).get("git_revision") != revision
            or set(states) != {"model", "optimizer", "scheduler", "rng", "sampler"}
            or any(item.get("promotable") is not False for item in states.values())
            or not nonpromotable.get("inputs")):
        raise ValueError("an engineering transaction is promotable")
    for item in nonpromotable["inputs"]:
        for path_key, sha_key in (("run_status_path", "run_status_sha256"),
                                  ("receipt_path", "receipt_sha256"),
                                  ("checkpoint_path", "checkpoint_sha256")):
            path = Path(str(item.get(path_key, ""))).resolve()
            if not path.is_file() or item.get(sha_key) != sha256_file(path):
                raise ValueError("non-promotion audit input hash mismatch")
    matrix = {
        "t0_a_final_commit": evidence(args.t0_a_report, {"git_revision": revision}),
        "three_source_t0_b": evidence(args.t0_b_report, {"identity": "exact-three-source"}),
        "a1o_fixed_train_subset_learnability": evidence(args.overfit_audit, overfit["thresholds"]),
        "matched_a0_a1o_real_runner_steps": evidence(args.matched_audit, matched["matched_fields"]),
        "checkpoint_save_resume": evidence(args.t0_b_report, {"actual_roundtrip": True}),
        "a1o_drop_head_free_val_load": evidence(args.head_free_audit, {"split": "val"}),
        "validator": evidence(args.validator_report, {"inventory": expected_counts}),
        "resource_preflight": evidence(args.t0_a_resolved_config, preflight),
        "provenance": evidence(args.t0_b_provenance, {"git_revision": revision}),
        "engineering_state_non_promotable": evidence(args.nonpromotable_audit, {
            "states": {name: item["promotable"] for name, item in states.items()}}),
        "formal_startup_step0": evidence(args.formal_startup_audit, {
            "initialization_sha256": arms["a0"]["initialization_sha256"]}),
        "engineering_subset_not_extra_weighted": evidence(args.engineering_subset, {
            "formal_train_reuse": subset["formal_train_reuse"],
            "transaction_promotion": subset["transaction_promotion"]}),
    }
    payload = {
        "schema_version": "parta_engineering_coverage_v2", "status": "complete_passed",
        "producer": {"path": str(Path(__file__).resolve()),
                     "sha256": sha256_file(Path(__file__).resolve()), "git_revision": revision},
        "source_registry": list(FROZEN_SOURCE_REGISTRY),
        "manifest_report_sha256": sha256_file(args.manifest_report),
        "exact_registry_sha256": stable_sha256(manifest_report["exact_canonical_inputs"]),
        "engineering_subset": {"path": str(args.engineering_subset.resolve()),
                               "sha256": sha256_file(args.engineering_subset),
                               "formal_train_reuse": subset["formal_train_reuse"],
                               "transaction_promotion": subset["transaction_promotion"]},
        "coverage_matrix": matrix,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
