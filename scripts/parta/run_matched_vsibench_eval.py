#!/usr/bin/env python3
"""Prepare or execute a matched A0/A1-O-drop lmms-eval transaction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from parta.checkpoint_selection import (assert_matched_selection_rule,
                                        validate_selection_report)  # noqa: E402
from parta.vsibench_eval import (artifact_digest, assert_zero_scene_overlap,
                                 environment_snapshot, extract_lmms_paired_records,
                                 plugin_environment, sample_identity,
                                 validate_head_free_audit, validate_matched_training_runs)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-checkpoint", type=Path, required=True)
    parser.add_argument("--a1o-drop-checkpoint", type=Path, required=True)
    parser.add_argument("--a1o-drop-audit", type=Path, required=True)
    parser.add_argument("--a0-run-dir", type=Path, required=True)
    parser.add_argument("--a1o-run-dir", type=Path, required=True)
    parser.add_argument("--a0-selection-report", type=Path, required=True)
    parser.add_argument("--a1o-selection-report", type=Path, required=True)
    parser.add_argument("--unified-gate", type=Path, required=True)
    parser.add_argument("--frozen-config-artifact", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--vsibench-manifest", type=Path, required=True)
    parser.add_argument("--vsibench-video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lmms-eval-root", type=Path, default=PROJECT / "src/vendor/lmms-eval")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--main-port", type=int, default=29517)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse eval output: {args.output_dir}")
    if args.num_processes < 1:
        raise ValueError("num-processes must be positive")
    for path in (args.a0_checkpoint, args.a1o_drop_checkpoint, args.base_model,
                 args.training_manifest, args.vsibench_manifest, args.vsibench_video_root,
                 args.lmms_eval_root):
        if not path.exists():
            raise FileNotFoundError(path)
    overlap = assert_zero_scene_overlap(args.training_manifest, args.vsibench_manifest)
    gate = json.loads(args.unified_gate.read_text(encoding="utf-8"))
    freeze = json.loads(args.frozen_config_artifact.read_text(encoding="utf-8"))
    if (gate.get("status") != "complete_passed"
            or gate.get("training_authorized_by_this_artifact") is not True
            or freeze.get("schema_version") != "parta_formal_config_freeze_v1"
            or freeze.get("status") != "frozen"
            or gate.get("frozen_config_artifact_sha256") != stable_sha256(freeze)):
        raise ValueError("VSI-Bench requires the passed Gate@CONFIG freeze transaction")
    selection_payloads = {
        "a0": json.loads(args.a0_selection_report.read_text(encoding="utf-8")),
        "a1o": json.loads(args.a1o_selection_report.read_text(encoding="utf-8")),
    }
    assert_matched_selection_rule(selection_payloads["a0"], selection_payloads["a1o"])
    freeze_sha = sha256_file(args.frozen_config_artifact)
    validate_selection_report(selection_payloads["a0"], arm="a0",
                              checkpoint=args.a0_checkpoint, frozen_config_sha256=freeze_sha)
    a1o_selected = Path(str(selection_payloads["a1o"].get("selected", {}).get(
        "checkpoint_path", ""
    )))
    validate_selection_report(selection_payloads["a1o"], arm="a1o",
                              checkpoint=a1o_selected, frozen_config_sha256=freeze_sha)
    head_audit = validate_head_free_audit(args.a1o_drop_audit, args.a1o_drop_checkpoint)
    training_match = validate_matched_training_runs(
        args.a0_run_dir, args.a1o_run_dir, args.a0_checkpoint, args.a1o_drop_checkpoint
    )
    task_source = args.lmms_eval_root / "lmms_eval/tasks/vsibench"
    source_yaml = task_source / "vsibench.yaml"
    source_utils = task_source / "utils.py"
    task_dir = args.output_dir / "task"
    task_dir.mkdir(parents=True)
    text = source_yaml.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.startswith("task:"):
            line = "task: parta_vsibench"
        elif line.startswith("    test:"):
            line = f"    test: {args.vsibench_manifest.resolve()}"
        elif line.startswith("  media_dir:"):
            line = f"  media_dir: {args.vsibench_video_root.resolve()}"
        lines.append(line)
    frozen_yaml = task_dir / "parta_vsibench.yaml"
    frozen_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(source_utils, task_dir / "utils.py")
    shared = {
        "task": "parta_vsibench", "batch_size": 1, "max_pixels": 268324,
        "min_pixels": 8192, "attn_implementation": "flash_attention_2",
        "generation": {"max_new_tokens": 16, "temperature": 0, "top_p": 1.0,
                       "num_beams": 1, "do_sample": False},
        "task_yaml_sha256": sha256_file(frozen_yaml),
        "task_utils_sha256": sha256_file(task_dir / "utils.py"),
        "vsibench_manifest_sha256": sha256_file(args.vsibench_manifest),
    }
    arms = {"a0": args.a0_checkpoint, "a1o_drop": args.a1o_drop_checkpoint}
    commands = {}
    for index, (arm, artifact) in enumerate(arms.items()):
        arm_output = args.output_dir / arm / "raw_results"
        model_args = (
            f"pretrained={args.base_model.resolve()},parta_artifact={artifact.resolve()},"
            f"parta_arm={arm},head_free_audit={args.a1o_drop_audit.resolve() if arm == 'a1o_drop' else ''},"
            "max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"
        )
        commands[arm] = ["accelerate", "launch", f"--num_processes={args.num_processes}",
            "--main_process_port", str(args.main_port + index), "-m", "lmms_eval",
            "--include_path", str(task_dir), "--model", "qwen3_vl_parta",
            "--model_args", model_args, "--tasks", "parta_vsibench", "--batch_size", "1",
            "--log_samples", "--log_samples_suffix", arm, "--output_path", str(arm_output),
            "--force_simple"]
    plan = {
        "schema_version": "parta_matched_vsibench_plan_v1", "status": "planned",
        "run_id": str(uuid.uuid4()), "created_at_unix": time.time(),
        "evaluation_arms": ["a0", "a1o_drop"], "shared_eval_contract": shared,
        "shared_eval_contract_sha256": stable_sha256(shared), "overlap_audit": overlap,
        "a1o_drop_head_free_audit": head_audit,
        "matched_training_audit": training_match,
        "gate_config": {"unified_gate_sha256": sha256_file(args.unified_gate),
                        "frozen_config_sha256": freeze_sha},
        "checkpoint_selection_reports": {
            "a0": sha256_file(args.a0_selection_report),
            "a1o": sha256_file(args.a1o_selection_report),
        },
        "sample_identity": sample_identity(args.vsibench_manifest),
        "artifacts": {arm: {"path": str(path.resolve()), "sha256": artifact_digest(path)}
                      for arm, path in arms.items()},
        "base_model": {"path": str(args.base_model.resolve()), "sha256": artifact_digest(args.base_model)},
        "environment": environment_snapshot(), "commands": commands,
        "command_environment": plugin_environment(
            PROJECT, args.lmms_eval_root, args.vsibench_video_root
        ),
    }
    plan["plan_sha256"] = stable_sha256(plan)
    atomic_json_dump(plan, args.output_dir / "eval_plan.json")
    if not args.execute:
        atomic_json_dump({"status": "awaiting_gpu_eval", "formal_gpu_evidence": False,
                          "plan_sha256": plan["plan_sha256"]}, args.output_dir / "run_status.json")
        return
    child_env = os.environ.copy()
    child_env.update(plan["command_environment"])
    child_env["PYTHONPATH"] += os.pathsep + child_env.get("PYTHONPATH", "")
    completed = []
    raw_by_arm = {}
    try:
        for arm in ("a0", "a1o_drop"):
            arm_root = args.output_dir / arm
            before = set(arm_root.rglob("results*.json")) if arm_root.exists() else set()
            started = time.time()
            subprocess.run(commands[arm], cwd=args.lmms_eval_root, env=child_env, check=True)
            fresh = [path for path in arm_root.rglob("results*.json")
                     if path not in before and path.stat().st_mtime >= started]
            if len(fresh) != 1:
                raise RuntimeError(f"{arm} produced {len(fresh)} fresh result files, expected one")
            raw_path = fresh[0]
            receipt = {
                "schema_version": "parta_vsibench_arm_receipt_v1", "status": "complete",
                "plan_sha256": plan["plan_sha256"], "run_id": plan["run_id"], "arm": arm,
                "artifact_sha256": plan["artifacts"][arm]["sha256"],
                "shared_eval_contract_sha256": plan["shared_eval_contract_sha256"],
                "sample_identity": plan["sample_identity"], "raw_result": str(raw_path.resolve()),
                "raw_result_sha256": sha256_file(raw_path), "started_at_unix": started,
                "finished_at_unix": time.time(),
                "evaluation_mode": "one_shot_after_checkpoint_and_config_freeze",
                "used_for_checkpoint_selection": False,
            }
            atomic_json_dump(receipt, arm_root / "result_receipt.json")
            raw_by_arm[arm] = raw_path
            completed.append(arm)
        paired_records = {
            arm: extract_lmms_paired_records(json.loads(path.read_text(encoding="utf-8")))
            for arm, path in raw_by_arm.items()
        }
        identities = {
            arm: [(row["source_dataset"], row["scene_id"], row["sample_id"])
                  for row in rows]
            for arm, rows in paired_records.items()
        }
        if identities["a0"] != identities["a1o_drop"]:
            raise ValueError("raw lmms A0/A1-O per-sample identities differ")
        paired_receipt = {
            "schema_version": "parta_vsibench_paired_records_receipt_v1",
            "status": "complete", "plan_sha256": plan["plan_sha256"],
            "run_id": plan["run_id"], "producer_script": str(Path(__file__).resolve()),
            "producer_script_sha256": sha256_file(Path(__file__).resolve()),
            "producer_git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            "raw_result_sha256": {arm: sha256_file(path) for arm, path in raw_by_arm.items()},
            "records": paired_records, "paired_sample_count": len(identities["a0"]),
            "identity_sha256": stable_sha256(identities["a0"]),
        }
        paired_receipt["receipt_payload_sha256"] = stable_sha256(paired_receipt)
        atomic_json_dump(paired_receipt, args.output_dir / "paired_records_receipt.json")
        atomic_json_dump({"status": "raw_eval_complete_pending_comparison", "formal_gpu_evidence": True,
                          "plan_sha256": plan["plan_sha256"], "completed_arms": completed,
                          "paired_records_receipt_sha256": sha256_file(
                              args.output_dir / "paired_records_receipt.json"
                          )},
                         args.output_dir / "run_status.json")
    except BaseException as error:
        atomic_json_dump({"status": "failed", "formal_gpu_evidence": False,
                          "plan_sha256": plan["plan_sha256"], "completed_arms": completed,
                          "error_type": type(error).__name__, "error": str(error),
                          "traceback": traceback.format_exc()}, args.output_dir / "run_status.json")
        raise


if __name__ == "__main__":
    main()
