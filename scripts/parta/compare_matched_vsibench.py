#!/usr/bin/env python3
"""Parse and compare matched raw lmms-eval VSI-Bench result files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from parta.provenance import atomic_json_dump, sha256_file  # noqa: E402
from parta.vsibench_eval import (extract_scores, paired_comparison,
                                 paired_scene_video_bootstrap,
                                 validate_paired_records_receipt,
                                 validate_result_receipt)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--a0-results", type=Path, required=True)
    parser.add_argument("--a1o-drop-results", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    plan = json.loads((args.eval_root / "eval_plan.json").read_text(encoding="utf-8"))
    if plan.get("schema_version") != "parta_matched_vsibench_plan_v1":
        raise ValueError("eval plan schema mismatch")
    parsed = {}
    raw = {}
    try:
        for arm, path in (("a0", args.a0_results), ("a1o_drop", args.a1o_drop_results)):
            if not path.is_file() or args.eval_root.resolve() not in path.resolve().parents:
                raise ValueError(f"{arm} raw result must be a file inside eval-root")
            receipt_path = args.eval_root / arm / "result_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            validate_result_receipt(receipt, plan=plan, arm=arm, raw_path=path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed[arm] = extract_scores(payload)
            raw[arm] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                        "receipt_sha256": sha256_file(receipt_path)}
        if raw["a0"]["sha256"] == raw["a1o_drop"]["sha256"]:
            raise ValueError("A0 and A1-O-drop raw result files are identical")
        comparison = paired_comparison(parsed["a0"], parsed["a1o_drop"])
        paired_receipt_path = args.eval_root / "paired_records_receipt.json"
        paired_receipt = json.loads(paired_receipt_path.read_text(encoding="utf-8"))
        payload_hash = paired_receipt.get("receipt_payload_sha256")
        canonical_producer = (PROJECT / "scripts/parta/run_matched_vsibench_eval.py").resolve()
        paired_records = validate_paired_records_receipt(
            paired_receipt, plan_sha256=plan["plan_sha256"],
            raw_result_paths={"a0": args.a0_results, "a1o_drop": args.a1o_drop_results},
            producer_path=canonical_producer, producer_sha256=sha256_file(canonical_producer),
            git_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
        )
        bootstrap = paired_scene_video_bootstrap(
            paired_records["a0"], paired_records["a1o_drop"],
            seed=args.bootstrap_seed, replicates=args.bootstrap_replicates,
        )
        comparison.update({"plan_sha256": plan["plan_sha256"], "run_id": plan["run_id"],
                           "raw_results": raw,
                           "shared_eval_contract_sha256": plan["shared_eval_contract_sha256"],
                           "aggregate_scores_role": "diagnostic_only",
                           "paired_scene_video_bootstrap": bootstrap,
                           "decision_status": bootstrap["decision"],
                           "paired_records": {
                               "receipt_sha256": sha256_file(paired_receipt_path),
                               "receipt_payload_sha256": payload_hash,
                           }})
        atomic_json_dump(comparison, args.eval_root / "paired_comparison.json")
        atomic_json_dump({"status": "complete", "decision_status": bootstrap["decision"],
                          "paired_comparison_sha256": sha256_file(args.eval_root / "paired_comparison.json")},
                         args.eval_root / "run_status.json")
    except BaseException as error:
        atomic_json_dump({"status": "failed", "error_type": type(error).__name__,
                          "error": str(error)}, args.eval_root / "run_status.json")
        raise


if __name__ == "__main__":
    main()
