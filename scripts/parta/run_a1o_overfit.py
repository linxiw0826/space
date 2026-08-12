#!/usr/bin/env python3
"""Run the canonical A1-O provisional D-62 overfit transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_RUNNER = (PROJECT / "scripts/parta/train_parta.py").resolve()
sys.path.insert(0, str(PROJECT / "src"))
from parta.worker_trust import (TRAIN_WORKER_SWITCH_FLAGS, train_worker_flag_contract,
                                validate_python_worker)  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = json.loads(args.command.read_text(encoding="utf-8"))
    argv = command.get("argv")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    validate_python_worker(command, argv, script=TRAIN_RUNNER,
                           script_sha256=digest(TRAIN_RUNNER), git_revision=revision,
                           engineering_mode="overfit",
                           allowed_value_flags=train_worker_flag_contract("overfit"),
                           allowed_switch_flags=TRAIN_WORKER_SWITCH_FLAGS)
    if ("--arm" not in argv or argv[argv.index("--arm") + 1] != "a1o"
            or "--max-steps" not in argv or argv[argv.index("--max-steps") + 1] != "100"):
        raise ValueError("untrusted or non-provisional A1-O overfit worker")
    run_dir = Path(str(command.get("run_dir", ""))).resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    subprocess.run(argv, check=True)
    receipt_path = run_dir / "engineering_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("engineering_mode") != "overfit" or receipt.get("arm") != "a1o"
            or receipt.get("promotable") is not False
            or receipt.get("actual_unique_examples", 0) < 64
            or receipt.get("optimizer_steps") != 100
            or receipt.get("state_loss_decrease_fraction") is None
            or float(receipt["state_loss_decrease_fraction"]) < 0.20):
        raise ValueError("A1-O provisional overfit thresholds failed")
    payload = {
        "schema_version": "parta_a1o_overfit_audit_v1", "status": "complete_passed",
        "producer": {"path": str(Path(__file__).resolve()),
                     "sha256": digest(Path(__file__).resolve()), "git_revision": revision},
        "defaults_status": "provisional_D62_execution_default_v1",
        "thresholds": {"minimum_unique_examples": 64, "optimizer_steps": 100,
                       "minimum_state_loss_decrease_fraction": 0.20},
        "worker": {"python_executable": sys.executable, "script_path": str(TRAIN_RUNNER),
                   "script_sha256": digest(TRAIN_RUNNER), "git_revision": revision},
        "run_dir": str(run_dir), "receipt": receipt,
        "receipt_sha256": digest(receipt_path),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
