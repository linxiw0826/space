"""Canonical four-rank A1-O resource-profile command identity."""

from __future__ import annotations

from typing import Any, Sequence
from pathlib import Path
import json

from parta.provenance import stable_sha256, sha256_file

LAMBDA_STATE = 0.02150771327925621
WORLD_SIZE = 4
FRAME_COUNT = 32
STRATEGIES = ("ddp", "fsdp")

VALUE_FLAGS = (
    "--arm", "--manifest", "--manifest-report", "--media-root", "--model-path",
    "--vggt-path", "--seed", "--learning-rate", "--weight-decay", "--lambda-state",
    "--max-grad-norm", "--gradient-accumulation-steps", "--dtype", "--num-workers",
    "--engineering-subset", "--engineering-mode", "--required-frame-count",
)


def normalize_profile_worker_argv(argv: Sequence[str]) -> dict[str, Any]:
    values = {flag.removeprefix("--").replace("-", "_"): argv[argv.index(flag) + 1]
              for flag in VALUE_FLAGS if flag in argv}
    values["source_roots"] = sorted(
        argv[index + 1] for index, value in enumerate(argv) if value == "--source-root"
    )
    values.update({
        "gradient_checkpointing": "--gradient-checkpointing" in argv,
        "dry_run": "--dry-run" in argv,
        "per_rank_batch_size": 1,
        "world_size": WORLD_SIZE,
        "effective_global_batch_size": WORLD_SIZE * int(
            values.get("gradient_accumulation_steps", 0)
        ),
    })
    if values.get("arm") != "a1o" or int(values.get("required_frame_count", 0)) != FRAME_COUNT:
        raise ValueError("profile command must be A1-O 32-frame")
    if float(values.get("lambda_state", "nan")) != LAMBDA_STATE:
        raise ValueError("profile command lambda_state differs from T0-B calibration")
    if not values["gradient_checkpointing"] or not values["dry_run"]:
        raise ValueError("profile command requires gradient checkpointing and dry-run")
    return values


def normalized_contract_sha256(contract: dict[str, Any]) -> str:
    return stable_sha256(contract)


def validate_profile_pair(commands: dict[str, Sequence[str]]) -> tuple[dict[str, Any], str]:
    if set(commands) != set(STRATEGIES):
        raise ValueError("profile requires exact DDP/FSDP command pair")
    normalized = {strategy: normalize_profile_worker_argv(argv)
                  for strategy, argv in commands.items()}
    if normalized["ddp"] != normalized["fsdp"]:
        raise ValueError("DDP and FSDP profile execution contracts differ")
    contract = normalized["ddp"]
    return contract, normalized_contract_sha256(contract)


def validate_rank_failure_rows(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != WORLD_SIZE or [row.get("rank") for row in rows] != list(range(WORLD_SIZE)):
        raise ValueError("OOM evidence requires exact rank IDs 0..3")
    for row in rows:
        if (row.get("schema_version") != "parta_rank_failure_v1"
                or not isinstance(row.get("stage"), str) or not row["stage"]
                or not isinstance(row.get("reason"), str) or not row["reason"]):
            raise ValueError("invalid structured rank failure evidence")
    if not any(row.get("oom") is True for row in rows):
        raise ValueError("OOM evidence lacks a real rank artifact with oom=true")


def checkpoint_artifact_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    configs = sorted(item for item in root.glob("*.json")
                     if not item.name.endswith(".index.json")) if root.is_dir() else []
    config_records = [{"name": item.name, "size_bytes": item.stat().st_size,
                       "sha256": sha256_file(item)} for item in configs]
    indexes = sorted(root.glob("*.index.json")) if root.is_dir() else []
    if len(indexes) > 1:
        raise ValueError("ambiguous checkpoint indexes")
    if not root.is_dir():
        weights = [root]
    elif indexes:
        index_payload = json.loads(indexes[0].read_text())
        weight_map = index_payload.get("weight_map", {})
        names = list(dict.fromkeys(weight_map.values()))
        weights = [root / name for name in names]
    else:
        weights = sorted(root.glob("*.safetensors")) or [
            item for item in sorted(root.glob("*.bin")) if "optimizer" not in item.name
        ]
    records = [{"name": item.name, "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item)} for item in weights]
    if indexes:
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        index_record = {"name": indexes[0].name, "size_bytes": indexes[0].stat().st_size,
                        "sha256": sha256_file(indexes[0]),
                        "ordered_weight_map_sha256": stable_sha256(sorted(weight_map.items()))}
        payload = {"mode": "indexed_weight_map", "index": index_record,
                   "config_files": config_records, "ordered_shards": records}
    else:
        payload = {"mode": "no_index_explicit_manifest", "config_files": config_records,
                   "ordered_shards": records}
    return {**payload, "index": payload.get("index"), "artifact_sha256": stable_sha256(payload)}


def validate_preexecution_profile(payload: dict[str, Any], argv: Sequence[str], *,
                                  manifest: str | Path, manifest_report: str | Path,
                                  engineering_subset: str | Path) -> dict[str, Any]:
    exact_keys = {"schema_version", "status", "distributed_strategy",
                  "normalized_execution_contract", "normalized_execution_contract_sha256",
                  "manifest", "manifest_report", "engineering_subset", "guide", "vggt"}
    if (set(payload) != exact_keys
            or payload.get("schema_version") != "parta_profile_preexecution_matched_v1"
            or payload.get("status") != "complete_preexecution"
            or payload.get("distributed_strategy") not in set(STRATEGIES)):
        raise ValueError("invalid pre-execution profile schema")
    contract = normalize_profile_worker_argv(argv)
    if (payload.get("normalized_execution_contract") != contract
            or payload.get("normalized_execution_contract_sha256")
               != normalized_contract_sha256(contract)):
        raise ValueError("pre-execution normalized command mismatch")
    for key, expected in (("manifest", manifest), ("manifest_report", manifest_report),
                          ("engineering_subset", engineering_subset)):
        path = Path(expected).resolve()
        if payload.get(key) != {"path": str(path), "sha256": sha256_file(path)}:
            raise ValueError(f"pre-execution {key} identity mismatch")
    for key, contract_key in (("guide", "model_path"), ("vggt", "vggt_path")):
        expected = checkpoint_artifact_identity(contract[contract_key])
        if payload.get(key) != expected:
            raise ValueError(f"pre-execution {key} artifact mismatch")
    return contract


def validate_resolved_profile(resolved: dict[str, Any], contract: dict[str, Any],
                              strategy: str) -> None:
    expected = {
        "lambda_state": LAMBDA_STATE, "per_rank_batch_size": 1,
        "effective_global_batch_size": int(contract["effective_global_batch_size"]),
        "gradient_accumulation_steps": int(contract["gradient_accumulation_steps"]),
        "gradient_checkpointing": True, "distributed_strategy": strategy,
        "learning_rate": float(contract["learning_rate"]),
        "weight_decay": float(contract["weight_decay"]),
        "max_grad_norm": float(contract["max_grad_norm"]), "dtype": contract["dtype"],
        "num_workers": int(contract["num_workers"]), "seed": int(contract["seed"]),
        "required_frame_count": FRAME_COUNT, "engineering_mode": "resource_profile",
        "dry_run": True, "world_size": WORLD_SIZE,
        "manifest": str(Path(contract["manifest"]).resolve()),
        "manifest_report": str(Path(contract["manifest_report"]).resolve()),
        "media_root": str(Path(contract["media_root"]).resolve()),
        "model_path": str(Path(contract["model_path"]).resolve()),
        "vggt_path": str(Path(contract["vggt_path"]).resolve()),
        "engineering_subset": str(Path(contract["engineering_subset"]).resolve()),
    }
    if any(resolved.get(key) != value for key, value in expected.items()):
        raise ValueError("profile resolved config differs from normalized command")
    roots = dict(value.split("=", 1) for value in contract["source_roots"])
    expected_roots = {key: str(Path(value).resolve()) for key, value in roots.items()}
    if resolved.get("source_roots") != expected_roots:
        raise ValueError("profile resolved source roots differ from normalized command")
