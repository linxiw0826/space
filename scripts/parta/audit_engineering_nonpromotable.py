#!/usr/bin/env python3
"""Derive the five-state non-promotion proof from real engineering transactions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from parta.provenance import sha256_file, atomic_json_dump  # noqa: E402
from parta.checkpoint import TRAINING_CHECKPOINT_SCHEMA  # noqa: E402


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) < 3:
        raise ValueError("non-promotion audit requires overfit, matched, and profile transactions")
    inputs = []
    modes = set()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    train_runner = (PROJECT / "scripts/parta/train_parta.py").resolve()
    lifecycle_keys = {f"{name}_promotable" for name in
                      ("model", "optimizer", "scheduler", "rng", "sampler")}
    for raw_dir in args.run_dir:
        run_dir = raw_dir.resolve()
        receipt_path, status_path = run_dir / "engineering_receipt.json", run_dir / "run_status.json"
        receipt, status = load(receipt_path), load(status_path)
        checkpoint = Path(str(status.get("checkpoint_path", ""))).resolve()
        lifecycle = status.get("transaction_lifecycle", {})
        if (receipt.get("schema_version") != "parta_engineering_runner_receipt_v1"
                or receipt.get("status") != "complete"
                or receipt.get("transaction_kind") != "engineering"
                or receipt.get("promotable") is not False
                or status.get("status") != "complete"
                or status.get("schema_version") != "parta_training_provenance_v1"
                or status.get("git_revision") != revision
                or status.get("runner") != {"path": str(train_runner),
                                             "sha256": sha256_file(train_runner)}
                or lifecycle.get("transaction_kind") != "engineering"
                or lifecycle.get("promotable") is not False
                or any(lifecycle.get(key) is not False for key in lifecycle_keys)
                or status.get("checkpoint_role") != "engineering_final"
                or not checkpoint.is_file()
                or status.get("checkpoint_sha256") != sha256_file(checkpoint)
                or receipt.get("checkpoint_sha256") != status.get("checkpoint_sha256")):
            raise ValueError(f"invalid engineering transaction: {run_dir}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        contract = checkpoint_payload.get("contract", {})
        required_state = {"model", "optimizer", "scheduler", "rng_state",
                          "sampler_position"}
        if (checkpoint_payload.get("schema_version") != TRAINING_CHECKPOINT_SCHEMA
                or required_state - set(checkpoint_payload)
                or contract.get("transaction_kind") != "engineering"
                or contract.get("promotable") is not False):
            raise ValueError(f"engineering checkpoint contract is invalid: {checkpoint}")
        modes.add(receipt.get("engineering_mode"))
        inputs.append({"run_dir": str(run_dir), "run_status_path": str(status_path),
                       "run_status_sha256": sha256_file(status_path),
                       "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
                       "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                       "engineering_mode": receipt["engineering_mode"]})
    if not {"overfit", "matched_runner", "resource_profile"}.issubset(modes):
        raise ValueError("required engineering transaction modes are incomplete")
    producer = Path(__file__).resolve()
    payload = {"schema_version": "parta_non_promotable_audit_v1", "status": "complete_passed",
               "producer": {"path": str(producer), "sha256": sha256_file(producer),
                            "git_revision": revision},
               "inputs": inputs,
               "states": {name: {"promotable": False, "disposition": "discard_before_formal_step0"}
                          for name in ("model", "optimizer", "scheduler", "rng", "sampler")}}
    if args.output.exists():
        raise FileExistsError(args.output)
    atomic_json_dump(payload, args.output)


if __name__ == "__main__":
    main()
