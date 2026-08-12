"""Fail-closed contracts for matched A0/A1-O-drop VSI-Bench evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provenance import sha256_file, stable_sha256

VSIBENCH_COMPONENTS = (
    "object_rel_direction", "object_rel_distance", "route_planning",
    "obj_appearance_order", "object_abs_distance", "object_counting",
    "object_size_estimation", "room_size_estimation",
)
SOURCE_ALIASES = {
    "adt": "adt", "hypersim": "hypersim", "scannet": "scannet",
    "scannetpp": "scannetppv2", "scannetppv2": "scannetppv2",
}


def _norm_source(value: Any) -> str:
    raw = str(value).lower().strip()
    key = "scannetpp" if "++" in raw else "".join(
        character for character in raw if character.isalnum()
    )
    if key not in SOURCE_ALIASES:
        raise ValueError(f"unknown source_dataset alias: {value!r}")
    return SOURCE_ALIASES[key]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{number}") from error


def scene_keys(path: Path, *, training: bool) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    rows = 0
    for row in _read_jsonl(path):
        if training and row.get("split") not in (None, "train"):
            continue
        source = row.get("source_dataset", row.get("dataset"))
        scene = row.get("canonical_scene_id", row.get("scene_id", row.get("scene_name")))
        if source is None or scene is None or not str(scene).strip():
            raise ValueError(f"manifest row lacks source/canonical scene identity: {row}")
        keys.add((_norm_source(source), str(scene).strip()))
        rows += 1
    if not rows:
        raise ValueError(f"manifest contains no {'train ' if training else ''}rows: {path}")
    return keys


def assert_zero_scene_overlap(training_manifest: Path, vsi_manifest: Path) -> dict[str, Any]:
    train = scene_keys(training_manifest, training=True)
    evaluation = scene_keys(vsi_manifest, training=False)
    overlap = sorted(train & evaluation)
    if overlap:
        raise ValueError(f"training/VSI-Bench scene overlap detected: {overlap[:20]}")
    return {
        "schema_version": "parta_vsibench_overlap_audit_v1", "passed": True,
        "identity": "source_dataset+canonical_scene_id", "training_scenes": len(train),
        "evaluation_scenes": len(evaluation), "overlap_count": 0,
        "training_manifest_sha256": sha256_file(training_manifest),
        "vsibench_manifest_sha256": sha256_file(vsi_manifest),
    }


def artifact_digest(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    records = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        records.append({"path": str(item.relative_to(path)), "size": item.stat().st_size,
                        "sha256": sha256_file(item)})
    if not records:
        raise ValueError(f"empty artifact directory: {path}")
    return stable_sha256(records)


def validate_head_free_audit(audit_path: Path, artifact_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "parta_head_free_load_audit_v1",
        "status": "complete_passed", "independent_model_construction": True,
        "forward_passed": True, "qa_forward_contract": "a0_shared_forward_v1",
    }
    failures = [key for key, expected in required.items() if audit.get(key) != expected]
    if audit.get("missing_keys") or audit.get("unexpected_keys"):
        failures.append("incompatible_keys")
    if not audit.get("dropped_state_head_keys"):
        failures.append("dropped_state_head_keys")
    digest = artifact_digest(artifact_path)
    if audit.get("head_free_artifact_sha256") != digest:
        failures.append("head_free_artifact_sha256")
    if failures:
        raise ValueError(f"A1-O-drop independent head-free audit failed: {sorted(set(failures))}")
    return {"path": str(audit_path.resolve()), "sha256": sha256_file(audit_path),
            "artifact_sha256": digest, "passed": True}


def environment_snapshot() -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(),
            "executable": sys.executable, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")}


def plugin_environment(project: Path, lmms_root: Path, video_root: Path) -> dict[str, str]:
    return {"LMMS_EVAL_PLUGINS": "eval",
            "PYTHONPATH": os.pathsep.join((str(project / "src"), str(lmms_root))),
            "VSIBENCH_VIDEO_ROOT": str(video_root.resolve())}


def validate_matched_training_runs(a0_run: Path, a1o_run: Path,
                                   a0_checkpoint: Path, a1o_drop: Path) -> dict[str, Any]:
    """Cross-check the two training transactions and their exported checkpoints."""
    import torch
    a0 = torch.load(a0_checkpoint, map_location="cpu", weights_only=False)
    drop = torch.load(a1o_drop, map_location="cpu", weights_only=False)
    if a0.get("schema_version") != "parta_training_checkpoint_v1":
        raise ValueError("A0 checkpoint schema mismatch")
    if a0.get("contract", {}).get("arm") != "a0":
        raise ValueError("A0 checkpoint arm mismatch")
    if (a0.get("contract", {}).get("transaction_kind") != "formal"
            or a0.get("contract", {}).get("promotable") is not True):
        raise ValueError("A0 evaluation requires a formal/promotable checkpoint")
    if drop.get("schema_version") != "parta_a1o_drop_checkpoint_v1":
        raise ValueError("A1-O-drop checkpoint schema mismatch")
    if drop.get("source_contract", {}).get("arm") != "a1o":
        raise ValueError("A1-O-drop source arm mismatch")
    if (drop.get("source_contract", {}).get("transaction_kind") != "formal"
            or drop.get("source_contract", {}).get("promotable") is not True):
        raise ValueError("A1-O-drop evaluation requires a formal/promotable checkpoint")
    comparable_contract = ("manifest_sha256", "matched_contract_sha256")
    mismatches = [key for key in comparable_contract
                  if a0["contract"].get(key) != drop["source_contract"].get(key)]
    run_payloads = {}
    for arm, root in (("a0", a0_run), ("a1o", a1o_run)):
        values = {}
        for name in ("matched_fairness_contract.json", "resolved_config.json", "run_status.json"):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            values[name] = json.loads(path.read_text(encoding="utf-8"))
        if values["run_status.json"].get("status") != "complete":
            raise ValueError(f"{arm} training run is not complete")
        run_payloads[arm] = values
    for arm, contract in (("a0", a0["contract"]), ("a1o", drop["source_contract"])):
        status = run_payloads[arm]["run_status.json"]
        if contract.get("resolved_config_sha256") != status.get("resolved_config_sha256"):
            mismatches.append(f"{arm}_checkpoint_resolved_config")
        if contract.get("manifest_sha256") != status.get("artifacts", {}).get("manifest", {}).get("sha256"):
            mismatches.append(f"{arm}_checkpoint_manifest")
    a0_status = run_payloads["a0"]["run_status.json"]
    if (a0_status.get("checkpoint_role") != "selected_validation"
            or Path(str(a0_status.get("checkpoint_path", ""))).resolve() != a0_checkpoint.resolve()
            or a0_status.get("checkpoint_sha256") != sha256_file(a0_checkpoint)
            or int(a0_status.get("selected_step", -1)) != int(a0.get("global_step", -2))):
        mismatches.append("a0_selected_checkpoint_identity")
    source = drop["source_contract"].get("source_checkpoint", {})
    source_path = Path(str(source.get("path", "")))
    a1_status = run_payloads["a1o"]["run_status.json"]
    source_payload = (torch.load(source_path, map_location="cpu", weights_only=False)
                      if source_path.is_file() else {})
    if (source.get("role") != "selected_validation" or not source_path.is_file()
            or source.get("sha256") != sha256_file(source_path)
            or source_payload.get("schema_version") != "parta_training_checkpoint_v1"
            or source_payload.get("contract", {}).get("arm") != "a1o"
            or source_payload.get("contract", {}).get("matched_contract_sha256")
               != drop["source_contract"].get("matched_contract_sha256")
            or int(source_payload.get("global_step", -1)) != int(source.get("global_step", -2))
            or a1_status.get("checkpoint_role") != "selected_validation"
            or Path(str(a1_status.get("checkpoint_path", ""))).resolve() != source_path.resolve()
            or a1_status.get("checkpoint_sha256") != source.get("sha256")
            or int(a1_status.get("selected_step", -1)) != int(source.get("global_step", -2))):
        mismatches.append("a1o_source_selected_checkpoint_identity")
    matched_a0 = run_payloads["a0"]["matched_fairness_contract.json"]
    matched_a1 = run_payloads["a1o"]["matched_fairness_contract.json"]
    if matched_a0 != matched_a1:
        mismatches.append("matched_fairness_contract")
    allowed = {"arm", "lambda_state"}
    config_a0 = run_payloads["a0"]["resolved_config.json"]
    config_a1 = run_payloads["a1o"]["resolved_config.json"]
    common_a0 = {key: value for key, value in config_a0.items() if key not in allowed}
    common_a1 = {key: value for key, value in config_a1.items() if key not in allowed}
    if common_a0 != common_a1:
        mismatches.append("resolved_config_except_arm_lambda_state")
    required_matched = {"manifest_sha256", "initialization_sha256",
                        "exact_frame_binding_sha256", "seed", "max_steps"}
    if not required_matched.issubset(matched_a0):
        mismatches.append("matched_contract_required_fields")
    if mismatches:
        raise ValueError(f"A0/A1-O training runs are not matched: {sorted(set(mismatches))}")
    return {
        "passed": True, "a0_run": str(a0_run.resolve()), "a1o_run": str(a1o_run.resolve()),
        "matched_contract_sha256": stable_sha256(matched_a0),
        "common_resolved_config_sha256": stable_sha256(common_a0),
        "matched_fields": {key: matched_a0[key] for key in sorted(required_matched)},
    }


def extract_scores(payload: Any) -> dict[str, float]:
    candidates: list[Mapping[str, Any]] = []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "overall" in value:
                candidates.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)
    complete = []
    for candidate in candidates:
        parsed: dict[str, float] = {}
        for component in VSIBENCH_COMPONENTS:
            matches = [value for key, value in candidate.items() if str(key).startswith(component + "_")]
            if len(matches) == 1 and isinstance(matches[0], (int, float)):
                parsed[component] = float(matches[0])
        if len(parsed) == len(VSIBENCH_COMPONENTS) and isinstance(candidate["overall"], (int, float)):
            parsed["Overall"] = float(candidate["overall"])
            if not all(value == value and abs(value) != float("inf") for value in parsed.values()):
                raise ValueError("VSI-Bench scores contain non-finite values")
            complete.append(parsed)
    if len(complete) != 1:
        raise ValueError(f"expected exactly one complete VSI-Bench score mapping, found {len(complete)}")
    return complete[0]


def sample_identity(path: Path) -> dict[str, Any]:
    identities = sorted(scene_keys(path, training=False))
    row_count = sum(1 for _ in _read_jsonl(path))
    return {"row_count": row_count, "scene_count": len(identities),
            "scene_identity_sha256": stable_sha256(identities)}


def extract_lmms_paired_records(payload: Any) -> list[dict[str, Any]]:
    """Extract only explicit lmms per-sample identities/scores; unknown variants fail closed."""
    candidates = []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if "doc_id" in value and isinstance(value.get("doc"), Mapping):
                candidates.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)
    records = []
    seen = set()
    for row in candidates:
        doc = row["doc"]
        source = doc.get("source_dataset", doc.get("dataset"))
        unit = doc.get("scene_id", doc.get("video_id"))
        score = row.get("score", row.get("exact_match"))
        if source is None or unit is None or not isinstance(score, (int, float)):
            raise ValueError("lmms per-sample row lacks explicit source, scene/video, or numeric score")
        record = {"source_dataset": _norm_source(source), "scene_id": str(unit),
                  "sample_id": str(row["doc_id"]), "score": float(score)}
        identity = (record["source_dataset"], record["scene_id"], record["sample_id"])
        if identity in seen or not math.isfinite(record["score"]):
            raise ValueError("lmms paired records contain duplicate identity or non-finite score")
        seen.add(identity)
        records.append(record)
    if not records:
        raise ValueError("raw lmms result contains no explicit supported per-sample records")
    return sorted(records, key=lambda item: (
        item["source_dataset"], item["scene_id"], item["sample_id"]
    ))


def validate_paired_records_receipt(
    receipt: Mapping[str, Any], *, plan_sha256: str,
    raw_result_paths: Mapping[str, Path], producer_path: Path,
    producer_sha256: str, git_revision: str,
) -> dict[str, Any]:
    payload = dict(receipt)
    declared = payload.pop("receipt_payload_sha256", None)
    records = payload.get("records", {})
    canonical_producer = producer_path.resolve()
    actual_raw_sha256 = {arm: sha256_file(path) for arm, path in raw_result_paths.items()}
    if (payload.get("schema_version") != "parta_vsibench_paired_records_receipt_v1"
            or payload.get("status") != "complete" or payload.get("plan_sha256") != plan_sha256
            or payload.get("raw_result_sha256") != actual_raw_sha256
            or Path(str(payload.get("producer_script", ""))).resolve() != canonical_producer
            or payload.get("producer_script_sha256") != producer_sha256
            or sha256_file(canonical_producer) != producer_sha256
            or payload.get("producer_git_revision") != git_revision
            or declared != stable_sha256(payload) or set(records) != {"a0", "a1o_drop"}):
        raise ValueError("signed paired-record receipt is invalid or tampered")
    extracted = {
        arm: extract_lmms_paired_records(json.loads(path.read_text(encoding="utf-8")))
        for arm, path in raw_result_paths.items()
    }
    if records != extracted:
        raise ValueError("signed paired records differ from hash-bound raw results")
    left = [(row["source_dataset"], row["scene_id"], row["sample_id"])
            for row in records["a0"]]
    right = [(row["source_dataset"], row["scene_id"], row["sample_id"])
             for row in records["a1o_drop"]]
    if left != right or payload.get("identity_sha256") != stable_sha256(left):
        raise ValueError("signed paired-record identities differ or were tampered")
    return records


def validate_result_receipt(receipt: Mapping[str, Any], *, plan: Mapping[str, Any],
                            arm: str, raw_path: Path) -> None:
    expected = {
        "schema_version": "parta_vsibench_arm_receipt_v1", "status": "complete",
        "plan_sha256": plan["plan_sha256"], "run_id": plan["run_id"], "arm": arm,
        "artifact_sha256": plan["artifacts"][arm]["sha256"],
        "shared_eval_contract_sha256": plan["shared_eval_contract_sha256"],
        "sample_identity": plan["sample_identity"], "raw_result_sha256": sha256_file(raw_path),
        "evaluation_mode": "one_shot_after_checkpoint_and_config_freeze",
        "used_for_checkpoint_selection": False,
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    if failures or float(receipt.get("finished_at_unix", 0)) < float(plan["created_at_unix"]):
        raise ValueError(f"raw result receipt mismatch: {failures or ['freshness']}")


def paired_comparison(a0: Mapping[str, float], a1o_drop: Mapping[str, float]) -> dict[str, Any]:
    names = ("Overall",) + VSIBENCH_COMPONENTS
    if set(a0) != set(names) or set(a1o_drop) != set(names):
        raise ValueError("score schemas differ or are incomplete")
    return {
        "schema_version": "parta_matched_vsibench_comparison_v1",
        "decision_status": "not_frozen_report_only",
        "scores": {name: {"a0": a0[name], "a1o_drop": a1o_drop[name],
                          "delta_a1o_drop_minus_a0": a1o_drop[name] - a0[name]}
                   for name in names},
    }


def paired_scene_video_bootstrap(a0_rows: Iterable[Mapping[str, Any]],
                                 a1o_rows: Iterable[Mapping[str, Any]], *,
                                 seed: int = 42, replicates: int = 10_000) -> dict[str, Any]:
    """Deterministic paired bootstrap over scene/video units, never aggregate scores."""
    if replicates < 1000:
        raise ValueError("paired bootstrap requires at least 1000 replicates")

    def normalize(rows):
        result = {}
        for row in rows:
            source = _norm_source(row.get("source_dataset", row.get("dataset")))
            unit = row.get("scene_id", row.get("video_id"))
            sample = row.get("sample_id", row.get("doc_id", row.get("qa_id")))
            score = row.get("score")
            if unit is None or sample is None or not isinstance(score, (int, float)):
                raise ValueError("paired record requires source, scene/video, sample identity and score")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("paired record score is non-finite")
            identity = (source, str(unit), str(sample))
            if identity in result:
                raise ValueError(f"duplicate paired sample identity: {identity}")
            result[identity] = score
        if not result:
            raise ValueError("paired bootstrap input is empty")
        return result

    a0 = normalize(a0_rows)
    a1 = normalize(a1o_rows)
    if set(a0) != set(a1):
        raise ValueError("A0/A1-O paired sample identities differ")
    by_unit = {}
    for identity in sorted(a0):
        unit = identity[:2]
        by_unit.setdefault(unit, []).append(a1[identity] - a0[identity])
    unit_deltas = [sum(values) / len(values) for _, values in sorted(by_unit.items())]
    point = sum(unit_deltas) / len(unit_deltas)
    rng = random.Random(seed)
    draws = sorted(
        sum(unit_deltas[rng.randrange(len(unit_deltas))] for _ in unit_deltas) / len(unit_deltas)
        for _ in range(replicates)
    )

    def percentile(probability):
        position = probability * (len(draws) - 1)
        low = int(position)
        high = min(low + 1, len(draws) - 1)
        weight = position - low
        return draws[low] * (1.0 - weight) + draws[high] * weight

    lower, upper = percentile(0.025), percentile(0.975)
    decision = "GO" if lower > 0 else ("NO-GO" if point <= 0 else "INCONCLUSIVE")
    return {
        "schema_version": "parta_paired_scene_video_bootstrap_v1",
        "unit": "source_dataset+scene_or_video_id",
        "sample_identity": "source_dataset+scene_or_video_id+sample_id",
        "seed": seed, "replicates": replicates,
        "paired_sample_count": len(a0), "paired_unit_count": len(unit_deltas),
        "delta_a1o_drop_minus_a0": point,
        "ci95": {"lower": lower, "upper": upper},
        "decision": decision,
        "decision_rule": "lower>0:GO;point<=0:NO-GO;otherwise:INCONCLUSIVE",
    }
