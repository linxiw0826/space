#!/usr/bin/env python3
"""Run matched A0/A1-O real engineering transactions on one frozen subset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hashlib import sha256

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_RUNNER = (PROJECT / "scripts/parta/train_parta.py").resolve()
sys.path.insert(0, str(PROJECT / "src"))
from parta.worker_trust import (TRAIN_WORKER_SWITCH_FLAGS, train_worker_flag_contract,
                                validate_python_worker)  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-command", type=Path, required=True)
    parser.add_argument("--a1o-command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = {}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    runner_sha = _hash(TRAIN_RUNNER)
    for arm, command_path in (("a0", args.a0_command), ("a1o", args.a1o_command)):
        command = _json(command_path)
        argv = command.get("argv")
        validate_python_worker(command, argv, script=TRAIN_RUNNER, script_sha256=runner_sha,
                               git_revision=revision, engineering_mode="matched_runner",
                               allowed_value_flags=train_worker_flag_contract("matched_runner"),
                               allowed_switch_flags=TRAIN_WORKER_SWITCH_FLAGS)
        run_dir = Path(command["run_dir"]).resolve()
        if run_dir.exists():
            raise FileExistsError(run_dir)
        subprocess.run(argv, check=True)
        receipt_path = run_dir / "engineering_receipt.json"
        receipt = _json(receipt_path)
        if (
            receipt.get("schema_version") != "parta_engineering_runner_receipt_v1"
            or receipt.get("engineering_mode") != "matched_runner"
            or receipt.get("arm") != arm
            or receipt.get("promotable") is not False
            or receipt.get("all_losses_finite") is not True
            or int(receipt.get("optimizer_steps", 0)) < 1
        ):
            raise ValueError(f"invalid matched engineering receipt: {arm}")
        receipts[arm] = receipt | {"receipt_path": str(receipt_path.resolve()),
                                   "receipt_sha256": _hash(receipt_path)}
    matched_fields = (
        "manifest_sha256", "engineering_subset_sha256", "optimizer_steps",
        "optimizer_step_indices", "actual_frame_counts", "frame_binding_sha256",
        "exact_canonical_inputs_registry_sha256",
    )
    mismatches = [name for name in matched_fields if receipts["a0"].get(name) != receipts["a1o"].get(name)]
    if mismatches:
        raise ValueError(f"engineering A0/A1-O are not matched: {mismatches}")
    payload = {
        "schema_version": "parta_matched_engineering_receipt_v1",
        "status": "complete_passed",
        "producer": {"path": str(Path(__file__).resolve()), "sha256": _hash(Path(__file__).resolve()),
                     "git_revision": revision},
        "promotable": False,
        "matched_fields": {name: receipts["a0"][name] for name in matched_fields},
        "arms": receipts,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
