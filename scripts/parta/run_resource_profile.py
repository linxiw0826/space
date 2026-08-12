#!/usr/bin/env python3
"""Run independent, non-promotable 16/24/32 real forward/backward probes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


FRAME_COUNTS = (16, 24, 32)
SOURCE_REGISTRY = ("adt", "hypersim", "scannetppv2")
PROJECT = Path(__file__).resolve().parents[2]
TRAIN_RUNNER = (PROJECT / "scripts/parta/train_parta.py").resolve()
sys.path.insert(0, str(PROJECT / "src"))
from parta.worker_trust import (TRAIN_WORKER_SWITCH_FLAGS, train_worker_flag_contract,
                                validate_python_worker)  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engineering-subset", type=Path, required=True)
    parser.add_argument("--point-command", action="append", required=True, metavar="FRAMES=JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commands: dict[int, Path] = {}
    for value in args.point_command:
        frame_text, path_text = value.split("=", 1)
        frame_count = int(frame_text)
        if frame_count in commands:
            raise ValueError(f"duplicate profile point: {frame_count}")
        commands[frame_count] = Path(path_text)
    if tuple(sorted(commands)) != FRAME_COUNTS:
        raise ValueError("resource profile requires exactly 16/24/32 frame commands")
    measurements = []
    producer_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    train_runner_sha256 = _sha256(TRAIN_RUNNER)
    for frame_count in FRAME_COUNTS:
        command_record = _json(commands[frame_count])
        argv = command_record.get("argv")
        run_dir = Path(str(command_record.get("run_dir", ""))).resolve()
        receipt = run_dir / "engineering_receipt.json"
        command_path = commands[frame_count].resolve()
        if not isinstance(argv, list) or not argv or run_dir.exists():
            raise ValueError(f"invalid or stale profile command for {frame_count} frames")
        validate_python_worker(command_record, argv, script=TRAIN_RUNNER,
                               script_sha256=train_runner_sha256,
                               git_revision=producer_revision,
                               engineering_mode="resource_profile",
                               allowed_value_flags=train_worker_flag_contract("resource_profile"),
                               allowed_switch_flags=TRAIN_WORKER_SWITCH_FLAGS)
        required_pairs = {
            "--engineering-mode": "resource_profile",
            "--engineering-subset": str(args.engineering_subset.resolve()),
            "--required-frame-count": str(frame_count),
            "--output-dir": str(run_dir),
        }
        if "--dry-run" not in argv:
            raise ValueError(f"profile command is not the canonical non-promotable worker: {frame_count}")
        for flag, expected_value in required_pairs.items():
            if flag not in argv or argv.index(flag) + 1 >= len(argv):
                raise ValueError(f"profile command lacks {flag}: {frame_count}")
            actual = argv[argv.index(flag) + 1]
            if flag in {"--engineering-subset", "--output-dir"}:
                actual = str(Path(actual).resolve())
            if actual != expected_value:
                raise ValueError(f"profile command has wrong {flag}: {frame_count}")
        started = time.time_ns()
        completed = subprocess.run(argv, check=False, text=True, capture_output=True)
        if completed.returncode:
            status_path = run_dir / "run_status.json"
            status = _json(status_path) if status_path.is_file() else {}
            worker_error = "\n".join((completed.stdout or "", completed.stderr or ""))
            is_oom = ("OutOfMemory" in str(status.get("error_type"))
                      or "out of memory" in str(status.get("error", "")).lower()
                      or "out of memory" in worker_error.lower())
            if not is_oom:
                raise RuntimeError(f"profile worker failed without OOM evidence: {frame_count}")
            resolved_path = run_dir / "resolved_config.json"
            resolved_failure = _json(resolved_path) if resolved_path.is_file() else {}
            total_memory = resolved_failure.get("cuda_total_memory_bytes")
            if not isinstance(total_memory, int) or total_memory <= 0:
                raise RuntimeError("OOM worker lacks its own CUDA total-memory evidence")
            measurements.append({
                "frame_count": frame_count,
                "peak_memory_bytes": None,
                "total_memory_bytes": total_memory,
                "step_time_seconds": None,
                "throughput_samples_per_second": None,
                "oom": True,
                "batch_size": int(resolved_failure.get("world_size", 0)),
                "gradient_accumulation_steps": int(
                    resolved_failure.get("gradient_accumulation_steps", 0)
                ),
                "forward_backward_steps": 0,
                "artifacts": {
                    "command_record": {"path": str(command_path), "sha256": _sha256(command_path)},
                    "run_status": {"path": str(status_path), "sha256": _sha256(status_path)},
                    "resolved_config": {"path": str(resolved_path), "sha256": _sha256(resolved_path)},
                },
                "run_status_sha256": _sha256(status_path) if status_path.is_file() else None,
                "resolved_config_sha256": _sha256(resolved_path),
                "oom_evidence": {"error_type": status.get("error_type"),
                                 "error": status.get("error"),
                                 "worker_output_sha256": __import__("hashlib").sha256(
                                     worker_error.encode("utf-8")
                                 ).hexdigest()},
            })
            continue
        if not receipt.is_file() or receipt.stat().st_mtime_ns < started:
            raise RuntimeError(f"profile worker did not publish a fresh receipt: {frame_count}")
        point = _json(receipt)
        resolved = _json(run_dir / "resolved_config.json")
        steps = [json.loads(line) for line in (run_dir / "train_steps.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line]
        expected = {
            "schema_version": "parta_engineering_runner_receipt_v1",
            "transaction_kind": "engineering",
            "engineering_mode": "resource_profile",
            "promotable": False,
            "manifest_sha256": _sha256(args.manifest),
            "engineering_subset_sha256": _sha256(args.engineering_subset),
        }
        if any(point.get(key) != value for key, value in expected.items()):
            raise ValueError(f"profile receipt identity mismatch: {frame_count}")
        if point.get("actual_frame_counts") != [frame_count] or not steps:
            raise ValueError(f"profile worker did not use only {frame_count} exact frames")
        if int(point.get("optimizer_steps", 0)) != len(steps):
            raise ValueError(f"profile point lacks a real forward/backward transaction: {frame_count}")
        peak = max(row["peak_cuda_memory_bytes"] for row in steps)
        if peak is None:
            raise ValueError("profile worker lacks CUDA peak-memory evidence")
        measurements.append({
            "frame_count": frame_count,
            "peak_memory_bytes": peak,
            "total_memory_bytes": int(point["cuda_total_memory_bytes"]),
            "step_time_seconds": sum(float(row["step_seconds"]) for row in steps) / len(steps),
            "throughput_samples_per_second": sum(
                float(row["samples_per_second"]) for row in steps
            ) / len(steps),
            "oom": False,
            "batch_size": int(resolved["world_size"]),
            "gradient_accumulation_steps": int(resolved["gradient_accumulation_steps"]),
            "forward_backward_steps": len(steps),
            "artifacts": {
                "command_record": {"path": str(command_path), "sha256": _sha256(command_path)},
                "run_status": {"path": str((run_dir / "run_status.json").resolve()),
                               "sha256": _sha256(run_dir / "run_status.json")},
                "resolved_config": {"path": str((run_dir / "resolved_config.json").resolve()),
                                    "sha256": _sha256(run_dir / "resolved_config.json")},
                "engineering_receipt": {"path": str(receipt.resolve()), "sha256": _sha256(receipt)},
                "train_steps": {"path": str((run_dir / "train_steps.jsonl").resolve()),
                                "sha256": _sha256(run_dir / "train_steps.jsonl")},
            },
            "receipt_sha256": _sha256(receipt),
            "training_log_sha256": _sha256(run_dir / "train_steps.jsonl"),
            "derived_frame_binding_sha256": sorted({
                binding for row in steps for binding in row.get("frame_binding_sha256", ())
            }),
        })
    payload = {
        "schema_version": "parta_resource_profile_v2",
        "status": "complete",
        "transaction_kind": "engineering_resource_profile",
        "formal_training": False,
        "promotable": False,
        "source_registry": list(SOURCE_REGISTRY),
        "manifest_sha256": _sha256(args.manifest),
        "engineering_subset_sha256": _sha256(args.engineering_subset),
        "producer": {"path": str(Path(__file__).resolve()),
                     "sha256": _sha256(Path(__file__).resolve()),
                     "git_revision": producer_revision},
        "producer_git_revision": producer_revision,
        "train_runner": {"path": str(TRAIN_RUNNER), "sha256": train_runner_sha256},
        "measurements": measurements,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
