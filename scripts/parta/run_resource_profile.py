#!/usr/bin/env python3
"""Run the independent, non-promotable four-GPU 32-frame worst-case probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import hashlib
import os
import signal
from pathlib import Path


SOURCE_REGISTRY = ("adt", "hypersim", "scannetppv2")
PROJECT = Path(__file__).resolve().parents[2]
TRAIN_RUNNER = (PROJECT / "scripts/parta/train_parta.py").resolve()
sys.path.insert(0, str(PROJECT / "src"))
from parta.resource_profile_contract import (FRAME_COUNT, LAMBDA_STATE,
    STRATEGIES, WORLD_SIZE as REQUIRED_WORLD_SIZE, normalize_profile_worker_argv,
    normalize_profile_matched_execution, normalized_contract_sha256,
    validate_profile_pair)  # noqa: E402
from parta.resource_profile_contract import validate_preexecution_profile  # noqa: E402
from parta.resource_profile_contract import validate_resolved_profile  # noqa: E402
from parta.resource_profile_contract import validate_rank_failure_rows  # noqa: E402
from parta.resource_profile_contract import is_safe_profile_measurement  # noqa: E402
from parta.resource_profile_contract import validate_physical_execution_provenance  # noqa: E402
from parta.worker_trust import (TRAIN_WORKER_SWITCH_FLAGS, train_worker_flag_contract,
                                validate_torchrun_worker)  # noqa: E402


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_with_timeout(argv: list[str], timeout_seconds: int, evidence_path: Path,
                     strategy: str) -> subprocess.CompletedProcess:
    process = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({"schema_version": "parta_profile_timeout_v1",
            "strategy": strategy, "timeout_seconds": timeout_seconds,
            "terminated": True, "stdout_sha256": hashlib.sha256((stdout or "").encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256((stderr or "").encode()).hexdigest()},
            indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"profile worker timed out: {strategy}; evidence={evidence_path}")
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def write_worker_failure_diagnostics(
    output_path: Path,
    strategy: str,
    frame_count: int,
    run_dir: Path,
    completed: subprocess.CompletedProcess,
) -> Path:
    """Persist captured worker output before classifying a failed profile run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{output_path.stem}.{strategy}.worker_failure"
    stdout_path = output_path.parent / f"{stem}.stdout.log"
    stderr_path = output_path.parent / f"{stem}.stderr.log"
    diagnostic_path = output_path.parent / f"{stem}.json"
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    diagnostic_path.write_text(json.dumps({
        "schema_version": "parta_profile_worker_failure_v1",
        "strategy": strategy,
        "frame_count": frame_count,
        "returncode": completed.returncode,
        "run_dir": str(run_dir.resolve()),
        "stdout": {
            "path": str(stdout_path.resolve()),
            "sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "size_bytes": len(stdout.encode()),
        },
        "stderr": {
            "path": str(stderr_path.resolve()),
            "sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            "size_bytes": len(stderr.encode()),
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return diagnostic_path


def collect_rank_failure_evidence(run_dir: Path) -> list[dict]:
    """Load four rank records and require at least one authentic on-disk OOM."""
    rows = []
    real_oom = False
    failure_dir = run_dir / "rank_failures"
    for rank in range(REQUIRED_WORLD_SIZE):
        rank_path = failure_dir / f"rank-{rank}.json"
        if rank_path.is_file():
            row = _json(rank_path)
            real_oom = real_oom or row.get("oom") is True
        else:
            row = {
                "schema_version": "parta_rank_failure_v1", "rank": rank,
                "local_rank": rank, "stage": "torchrun_peer_termination",
                "reason": "rank artifact unavailable after torchrun termination",
                "oom": None, "device_name": None, "total_memory_bytes": None,
                "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                "finite": None,
            }
        rows.append(row)
    validate_rank_failure_rows(rows)
    if not real_oom:
        raise ValueError("OOM worker lacks a real on-disk rank artifact with oom=true")
    return rows


def command_physical_execution_provenance(command_records: dict[str, dict]) -> dict:
    records = {
        strategy: {
            "assigned_physical_gpus": record.get("assigned_physical_gpus"),
            "execution_environment": record.get("execution_environment"),
            "throughput_evidence_final": record.get("throughput_evidence_final"),
        }
        for strategy, record in command_records.items()
    }
    if set(records) != set(STRATEGIES) or records["ddp"] != records["fsdp"]:
        raise ValueError("DDP/FSDP physical execution provenance differs")
    return validate_physical_execution_provenance(
        records["ddp"], visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")
    )


def capture_gpu_preflight_snapshot(assigned: list[int]) -> dict:
    query = subprocess.check_output([
        "nvidia-smi", f"--id={','.join(map(str, assigned))}",
        "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in query.splitlines():
        if not line.strip():
            continue
        index, used, free = (int(item.strip()) for item in line.split(","))
        rows.append({"index": index, "memory_used_mib": used, "memory_free_mib": free})
    process_query = subprocess.run([
        "nvidia-smi", f"--id={','.join(map(str, assigned))}",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True, capture_output=True, check=False)
    if process_query.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed: {process_query.stderr.strip()}")
    return {
        "schema_version": "parta_profile_gpu_snapshot_v1",
        "gpus": sorted(rows, key=lambda row: row["index"]),
        "compute_processes": [line.strip() for line in process_query.stdout.splitlines()
                              if line.strip()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engineering-subset", type=Path, required=True)
    parser.add_argument("--point-command", action="append", required=True,
                        metavar="STRATEGY=JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    commands: dict[str, Path] = {}
    for value in args.point_command:
        strategy, path_text = value.split("=", 1)
        if strategy in commands:
            raise ValueError(f"duplicate profile strategy: {strategy}")
        commands[strategy] = Path(path_text)
    if tuple(sorted(commands)) != STRATEGIES:
        raise ValueError("resource profile requires exactly one DDP and one FSDP command")
    measurements = []
    reopened_preflight = {}
    reopened_runtime_matched = {}
    normalized_contract = None
    producer_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    train_runner_sha256 = _sha256(TRAIN_RUNNER)
    command_payloads = {strategy: _json(path).get("argv") for strategy, path in commands.items()}
    normalized_contract, normalized_contract_hash = validate_profile_pair(command_payloads)
    command_records = {strategy: _json(path) for strategy, path in commands.items()}
    execution_provenance = command_physical_execution_provenance(command_records)
    execution_provenance = {
        **execution_provenance,
        "preflight_gpu_snapshot": capture_gpu_preflight_snapshot(
            execution_provenance["assigned_physical_gpus"]
        ),
    }
    validate_physical_execution_provenance(execution_provenance, require_snapshot=True)
    final_throughput = execution_provenance["throughput_evidence_final"]
    for strategy in STRATEGIES:
        frame_count = FRAME_COUNT
        command_record = _json(commands[strategy])
        argv = command_record.get("argv")
        run_dir = Path(str(command_record.get("run_dir", ""))).resolve()
        receipt = run_dir / "engineering_receipt.json"
        command_path = commands[strategy].resolve()
        if not isinstance(argv, list) or not argv or run_dir.exists():
            raise ValueError(f"invalid or stale profile command for {frame_count} frames")
        validate_torchrun_worker(command_record, argv, script=TRAIN_RUNNER,
                               script_sha256=train_runner_sha256,
                               git_revision=producer_revision,
                               engineering_mode="resource_profile",
                               allowed_value_flags=train_worker_flag_contract("resource_profile"),
                               allowed_switch_flags=TRAIN_WORKER_SWITCH_FLAGS,
                               required_world_size=REQUIRED_WORLD_SIZE)
        required_pairs = {
            "--engineering-mode": "resource_profile",
            "--engineering-subset": str(args.engineering_subset.resolve()),
            "--required-frame-count": str(frame_count),
            "--output-dir": str(run_dir),
            "--distributed-strategy": strategy,
            "--arm": "a1o",
            "--lambda-state": str(LAMBDA_STATE),
            "--matched-contract": str((run_dir / "matched_fairness_contract.json").resolve()),
            "--gradient-accumulation-steps": "1",
        }
        if "--dry-run" not in argv:
            raise ValueError(f"profile command is not the canonical non-promotable worker: {frame_count}")
        for flag, expected_value in required_pairs.items():
            if flag not in argv or argv.index(flag) + 1 >= len(argv):
                raise ValueError(f"profile command lacks {flag}: {frame_count}")
            actual = argv[argv.index(flag) + 1]
            if flag in {"--engineering-subset", "--output-dir", "--matched-contract"}:
                actual = str(Path(actual).resolve())
            if actual != expected_value:
                raise ValueError(f"profile command has wrong {flag}: {frame_count}")
        if "--gradient-checkpointing" not in argv:
            raise ValueError("resource profile requires auditable gradient checkpointing")
        point_contract = normalize_profile_worker_argv(argv)
        if normalized_contract is None:
            normalized_contract = point_contract
        elif point_contract != normalized_contract:
            raise ValueError("DDP and FSDP profile execution contracts differ")
        point_contract_sha256 = normalized_contract_sha256(point_contract)
        started = time.time_ns()
        completed = run_with_timeout(
            argv, args.timeout_seconds,
            args.output.parent / f"{args.output.stem}.{strategy}.timeout.json", strategy,
        )
        if completed.returncode:
            diagnostic_path = write_worker_failure_diagnostics(
                args.output, strategy, frame_count, run_dir, completed
            )
            status_path = run_dir / "run_status.json"
            status = _json(status_path) if status_path.is_file() else {}
            worker_error = "\n".join((completed.stdout or "", completed.stderr or ""))
            is_oom = ("OutOfMemory" in str(status.get("error_type"))
                      or "out of memory" in str(status.get("error", "")).lower()
                      or "out of memory" in worker_error.lower())
            if not is_oom:
                raise RuntimeError(
                    "profile worker failed without OOM evidence: "
                    f"{frame_count}; diagnostics={diagnostic_path}"
                )
            resolved_path = run_dir / "resolved_config.json"
            resolved_failure = _json(resolved_path) if resolved_path.is_file() else {}
            rank_rows = collect_rank_failure_evidence(run_dir)
            aggregate_path = args.output.parent / f"{args.output.stem}.{strategy}.oom.json"
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_path.write_text(json.dumps({"strategy": strategy, "ranks": rank_rows,
                "worker_output_sha256": hashlib.sha256(worker_error.encode()).hexdigest()},
                indent=2, sort_keys=True) + "\n")
            existing_artifacts = {
                "command_record": {"path": str(command_path), "sha256": _sha256(command_path)},
                "oom_rank_evidence": {"path": str(aggregate_path.resolve()),
                                      "sha256": _sha256(aggregate_path)},
            }
            preflight_path = run_dir / "profile_preflight_matched_contract.json"
            if not preflight_path.is_file():
                raise RuntimeError("OOM candidate lacks pre-execution matched artifact")
            existing_artifacts["preflight_matched_contract"] = {
                "path": str(preflight_path.resolve()), "sha256": _sha256(preflight_path)
            }
            preflight_payload = _json(preflight_path)
            validate_preexecution_profile(preflight_payload, argv, manifest=args.manifest,
                manifest_report=Path(point_contract["manifest_report"]),
                engineering_subset=args.engineering_subset)
            if preflight_payload.get("distributed_strategy") != strategy:
                raise ValueError("OOM preflight matched strategy mismatch")
            reopened_preflight[strategy] = {
                key: value for key, value in preflight_payload.items()
                if key != "distributed_strategy"
            }
            if status_path.is_file():
                existing_artifacts["run_status"] = {"path": str(status_path),
                                                    "sha256": _sha256(status_path)}
            if resolved_path.is_file():
                existing_artifacts["resolved_config"] = {"path": str(resolved_path),
                                                         "sha256": _sha256(resolved_path)}
            matched_path = run_dir / "matched_fairness_contract.json"
            if matched_path.is_file():
                existing_artifacts["runtime_matched_contract"] = {
                    "path": str(matched_path.resolve()), "sha256": _sha256(matched_path)
                }
                matched_payload = _json(matched_path)
                execution = normalize_profile_matched_execution(
                    matched_payload.get("execution_contract", {}), strategy
                )
                reopened_runtime_matched[strategy] = {
                    **matched_payload, "execution_contract": execution
                }
            totals = [row["total_memory_bytes"] for row in rank_rows
                      if isinstance(row.get("total_memory_bytes"), int)]
            measurements.append({
                "frame_count": frame_count,
                "distributed_strategy": strategy,
                "peak_memory_bytes": None,
                "peak_reserved_memory_bytes": None,
                "total_memory_bytes": min(totals) if totals else None,
                "step_time_seconds": None,
                "throughput_samples_per_second": None,
                "oom": True,
                "batch_size": 1,
                "per_rank_batch_size": 1,
                "world_size": REQUIRED_WORLD_SIZE,
                "per_rank_peak_memory_bytes": rank_rows,
                "finite": None,
                "gradient_accumulation_steps": int(
                    point_contract["gradient_accumulation_steps"]
                ),
                "forward_backward_steps": 0,
                "normalized_execution_contract": point_contract,
                "normalized_execution_contract_sha256": point_contract_sha256,
                "artifacts": existing_artifacts,
                "run_status_sha256": _sha256(status_path) if status_path.is_file() else None,
                "resolved_config_sha256": _sha256(resolved_path) if resolved_path.is_file() else None,
                "oom_evidence": {"error_type": status.get("error_type"),
                                 "error": status.get("error"),
                                 "worker_output_contains_oom": is_oom,
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
        if int(resolved.get("world_size", 0)) != REQUIRED_WORLD_SIZE:
            raise ValueError("resource profile did not run on exactly four ranks")
        validate_resolved_profile(resolved, point_contract, strategy)
        if point.get("resolved_execution_contract", {}).get("distributed_strategy") != strategy:
            raise ValueError("profile receipt resolved execution contract mismatch")
        per_rank_peak = point.get("per_rank_cuda_peak_memory_bytes")
        if (not isinstance(per_rank_peak, list)
                or len(per_rank_peak) != REQUIRED_WORLD_SIZE
                or [item.get("rank") for item in per_rank_peak] != list(range(REQUIRED_WORLD_SIZE))
                or any(not isinstance(item.get("peak_allocated_bytes"), int)
                       or item["peak_allocated_bytes"] <= 0
                       or not isinstance(item.get("peak_reserved_bytes"), int)
                       or item["peak_reserved_bytes"] <= 0
                       or not isinstance(item.get("total_memory_bytes"), int)
                       or item["total_memory_bytes"] <= 0 for item in per_rank_peak)
                or any("NVIDIA H20" not in str(item.get("device_name"))
                       for item in per_rank_peak)):
            raise ValueError("profile worker lacks four-rank H20 CUDA peak-memory evidence")
        if point.get("all_losses_finite") is not True:
            raise ValueError("profile worker produced non-finite losses")
        peak = max(item["peak_allocated_bytes"] for item in per_rank_peak)
        peak_reserved = max(item["peak_reserved_bytes"] for item in per_rank_peak)
        if peak is None:
            raise ValueError("profile worker lacks CUDA peak-memory evidence")
        measurements.append({
            "frame_count": frame_count,
            "distributed_strategy": strategy,
            "peak_memory_bytes": peak,
            "peak_reserved_memory_bytes": peak_reserved,
            "total_memory_bytes": int(point["cuda_total_memory_bytes"]),
            "step_time_seconds": sum(float(row["step_seconds"]) for row in steps) / len(steps),
            "throughput_samples_per_second": REQUIRED_WORLD_SIZE * sum(
                float(row["samples_per_second"]) for row in steps
            ) / len(steps),
            "throughput_scope": "global_four_rank",
            "oom": False,
            "batch_size": 1,
            "per_rank_batch_size": 1,
            "world_size": int(resolved["world_size"]),
            "per_rank_peak_memory_bytes": per_rank_peak,
            "finite": True,
            "normalized_execution_contract": point_contract,
            "normalized_execution_contract_sha256": point_contract_sha256,
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
                "matched_contract": {
                    "path": str((run_dir / "matched_fairness_contract.json").resolve()),
                    "sha256": _sha256(run_dir / "matched_fairness_contract.json"),
                },
                "preflight_matched_contract": {
                    "path": str((run_dir / "profile_preflight_matched_contract.json").resolve()),
                    "sha256": _sha256(run_dir / "profile_preflight_matched_contract.json"),
                },
            },
            "receipt_sha256": _sha256(receipt),
            "training_log_sha256": _sha256(run_dir / "train_steps.jsonl"),
            "derived_frame_binding_sha256": sorted({
                binding for row in steps for binding in row.get("frame_binding_sha256", ())
            }),
        })
        matched_payload = _json(run_dir / "matched_fairness_contract.json")
        execution = normalize_profile_matched_execution(
            matched_payload.get("execution_contract", {}), strategy
        )
        reopened_runtime_matched[strategy] = {**matched_payload, "execution_contract": execution}
        preflight_payload = _json(run_dir / "profile_preflight_matched_contract.json")
        validate_preexecution_profile(preflight_payload, argv, manifest=args.manifest,
            manifest_report=Path(point_contract["manifest_report"]),
            engineering_subset=args.engineering_subset)
        if preflight_payload.get("distributed_strategy") != strategy:
            raise ValueError("preflight matched strategy mismatch")
        reopened_preflight[strategy] = {
            key: value for key, value in preflight_payload.items()
            if key != "distributed_strategy"
        }
    if set(reopened_preflight) != set(STRATEGIES):
        raise ValueError("both profile strategies require pre-execution matched artifacts")
    if reopened_preflight["ddp"] != reopened_preflight["fsdp"]:
        raise ValueError("DDP/FSDP preflight matched contracts differ beyond strategy")
    if set(reopened_runtime_matched) == set(STRATEGIES) and \
            reopened_runtime_matched["ddp"] != reopened_runtime_matched["fsdp"]:
        raise ValueError("DDP/FSDP matched contracts differ beyond strategy")
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
        "required_world_size": REQUIRED_WORLD_SIZE,
        "required_device_name_substring": "NVIDIA H20",
        "measurements": measurements,
        "normalized_execution_contract": normalized_contract,
        "normalized_execution_contract_sha256": normalized_contract_hash,
        "lambda_state": LAMBDA_STATE,
        "physical_execution": execution_provenance,
        "throughput_evidence_final": final_throughput,
    }
    safe = [item for item in measurements if is_safe_profile_measurement(item)]
    selected = min(
        safe,
        key=lambda item: (-item["throughput_samples_per_second"],
                          item["peak_memory_bytes"], item["distributed_strategy"]),
    ) if safe else None
    payload["selection_rule"] = (
        "max_throughput_then_min_max_rank_allocated_then_strategy_lexical_v1"
    )
    payload["selected_strategy"] = (
        selected["distributed_strategy"] if selected is not None else None
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
