import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from parta.vsibench_eval import (VSIBENCH_COMPONENTS, assert_zero_scene_overlap,
                                 extract_scores, paired_comparison, plugin_environment,
                                 extract_lmms_paired_records,
                                 paired_scene_video_bootstrap,
                                 validate_matched_training_runs,
                                 validate_paired_records_receipt, validate_result_receipt)
from parta.provenance import sha256_file, stable_sha256
from parta.checkpoint_selection import select_validation_checkpoint, validate_selection_report


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_overlap_is_source_plus_canonical_scene(tmp_path):
    train, vsi = tmp_path / "train.jsonl", tmp_path / "vsi.jsonl"
    _jsonl(train, [{"source_dataset": "scannetppv2", "scene_id": "a", "split": "train"},
                   {"source_dataset": "adt", "scene_id": "same", "split": "val"}])
    _jsonl(vsi, [{"dataset": "ScanNet++", "scene_name": "b"},
                 {"dataset": "scannet", "scene_name": "a"}])
    assert assert_zero_scene_overlap(train, vsi)["overlap_count"] == 0
    _jsonl(vsi, [{"dataset": "ScanNet++", "scene_name": "a"}])
    with pytest.raises(ValueError, match="overlap"):
        assert_zero_scene_overlap(train, vsi)


def test_extract_and_compare_all_nine_scores():
    metrics = {f"{name}_accuracy": index / 10 for index, name in enumerate(VSIBENCH_COMPONENTS)}
    metrics["overall"] = 0.4
    scores = extract_scores({"results": {"vsibench": {"vsibench_score,none": metrics}}})
    assert scores["Overall"] == 0.4 and len(scores) == 9
    comparison = paired_comparison(scores, {key: value + 0.1 for key, value in scores.items()})
    assert comparison["decision_status"] == "not_frozen_report_only"
    assert comparison["scores"]["Overall"]["delta_a1o_drop_minus_a0"] == pytest.approx(0.1)


def test_missing_component_fails_closed():
    with pytest.raises(ValueError, match="complete VSI-Bench"):
        extract_scores({"overall": 0.5, "object_counting_accuracy": 0.2})


def test_score_parser_rejects_ambiguous_complete_mappings():
    metrics = {f"{name}_accuracy": 0.5 for name in VSIBENCH_COMPONENTS}
    metrics["overall"] = 0.5
    with pytest.raises(ValueError, match="found 2"):
        extract_scores({"first": metrics, "second": dict(metrics)})


def test_plugin_environment_is_explicit(tmp_path):
    env = plugin_environment(tmp_path / "project", tmp_path / "lmms", tmp_path / "videos")
    assert env["LMMS_EVAL_PLUGINS"] == "eval"
    assert str(tmp_path / "project/src") in env["PYTHONPATH"]
    assert str(tmp_path / "lmms") in env["PYTHONPATH"]


def test_result_receipt_rejects_wrong_arm_and_hash(tmp_path):
    raw = tmp_path / "result.json"
    raw.write_text("{}", encoding="utf-8")
    plan = {"plan_sha256": "a" * 64, "run_id": "run", "created_at_unix": 1,
            "artifacts": {"a0": {"sha256": "b" * 64}},
            "shared_eval_contract_sha256": "c" * 64,
            "sample_identity": {"row_count": 1}}
    receipt = {"schema_version": "parta_vsibench_arm_receipt_v1", "status": "complete",
               "plan_sha256": "a" * 64, "run_id": "run", "arm": "a1o_drop",
               "artifact_sha256": "b" * 64, "shared_eval_contract_sha256": "c" * 64,
               "sample_identity": {"row_count": 1}, "raw_result_sha256": "bad",
               "finished_at_unix": 2}
    with pytest.raises(ValueError, match="receipt mismatch"):
        validate_result_receipt(receipt, plan=plan, arm="a0", raw_path=raw)


def test_matched_training_rejects_intermediate_a0_checkpoint(tmp_path):
    manifest_sha = "1" * 64
    matched = {"schema_version": "parta_matched_fairness_v1",
               "manifest_sha256": manifest_sha, "initialization_sha256": "2" * 64,
               "exact_frame_binding_sha256": "3" * 64, "seed": 42, "max_steps": 10}
    matched_sha = stable_sha256(matched)
    a0_dir, a1_dir = tmp_path / "a0", tmp_path / "a1"
    a0_dir.mkdir(); a1_dir.mkdir()
    a0_ckpt = a0_dir / "checkpoint-5.pt"
    torch.save({"schema_version": "parta_training_checkpoint_v1", "global_step": 5,
                "contract": {"arm": "a0", "manifest_sha256": manifest_sha,
                             "resolved_config_sha256": "4" * 64,
                             "matched_contract_sha256": matched_sha}}, a0_ckpt)
    source = a1_dir / "checkpoint-final.pt"
    torch.save({"schema_version": "parta_training_checkpoint_v1", "global_step": 10}, source)
    drop = a1_dir / "checkpoint-a1o-drop.pt"
    torch.save({"schema_version": "parta_a1o_drop_checkpoint_v1",
                "source_contract": {"arm": "a1o", "manifest_sha256": manifest_sha,
                    "resolved_config_sha256": "5" * 64, "matched_contract_sha256": matched_sha,
                    "source_checkpoint": {"path": str(source), "sha256": sha256_file(source),
                                          "role": "final", "global_step": 10}}}, drop)
    common = {"seed": 42, "max_steps": 10}
    for arm, root, config_sha, checkpoint in (
        ("a0", a0_dir, "4" * 64, a0_ckpt), ("a1o", a1_dir, "5" * 64, source)
    ):
        (root / "matched_fairness_contract.json").write_text(json.dumps(matched))
        (root / "resolved_config.json").write_text(json.dumps({**common, "arm": arm,
                                                                  "lambda_state": 0.05}))
        status = {"status": "complete", "global_step": 5 if arm == "a0" else 10,
                  "resolved_config_sha256": config_sha,
                  "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "checkpoint_role": "intermediate" if arm == "a0" else "final",
                  "artifacts": {"manifest": {"sha256": manifest_sha}}}
        (root / "run_status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="formal/promotable|a0_selected_checkpoint_identity"):
        validate_matched_training_runs(a0_dir, a1_dir, a0_ckpt, drop)


def test_checkpoint_selection_uses_source_balanced_val_and_earliest_tie():
    rows = [{"step": step, "checkpoint_sha256": str(step) * 64,
             "source_val_qa_loss": {"adt": loss, "hypersim": loss, "scannetppv2": loss}}
            for step, loss in ((20, 0.2), (10, 0.2), (5, 0.3))]
    report = select_validation_checkpoint(rows)
    assert report["selected"]["step"] == 10
    assert report["metric_source"] == "validation_only"
    assert report["vsibench_used_for_selection"] is False


@pytest.mark.parametrize("delta,expected", [(1.0, "GO"), (0.0, "NO-GO"), (-0.1, "NO-GO")])
def test_scene_video_paired_bootstrap_decision_extremes(delta, expected):
    a0 = [{"source_dataset": "adt", "scene_id": f"s{i}", "sample_id": f"q{i}",
           "score": 0.2} for i in range(20)]
    a1 = [{**row, "score": row["score"] + delta} for row in a0]
    report = paired_scene_video_bootstrap(a0, a1, seed=42, replicates=1000)
    assert report["decision"] == expected
    assert report["paired_unit_count"] == 20


def test_scene_video_paired_bootstrap_is_deterministic_and_can_be_inconclusive():
    deltas = [1.0, -0.8] * 10
    a0 = [{"source_dataset": "hypersim", "video_id": f"v{i}", "doc_id": f"d{i}",
           "score": 0.0} for i in range(20)]
    a1 = [{**row, "score": delta} for row, delta in zip(a0, deltas)]
    first = paired_scene_video_bootstrap(a0, a1, seed=7, replicates=1000)
    second = paired_scene_video_bootstrap(a0, a1, seed=7, replicates=1000)
    assert first == second
    assert first["decision"] == "INCONCLUSIVE"


def test_selection_report_binds_freeze_and_selected_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint-selected.pt"
    checkpoint.write_bytes(b"selected")
    report = select_validation_checkpoint([
        {"step": 10, "checkpoint_sha256": sha256_file(checkpoint),
         "source_val_qa_loss": {"adt": 1.0, "hypersim": 2.0, "scannetppv2": 3.0}}
    ])
    report.update({"status": "complete_selected", "arm": "a0",
                   "source_registry": ["adt", "hypersim", "scannetppv2"],
                   "frozen_config_artifact_sha256": "f" * 64})
    report["selected"]["checkpoint_path"] = str(checkpoint.resolve())
    validate_selection_report(report, arm="a0", checkpoint=checkpoint,
                              frozen_config_sha256="f" * 64)
    with pytest.raises(ValueError, match="selection report"):
        validate_selection_report(report, arm="a0", checkpoint=checkpoint,
                                  frozen_config_sha256="e" * 64)


def test_lmms_paired_record_extraction_fails_closed_on_missing_identity_or_score():
    valid = {"samples": [{"doc_id": 7, "doc": {"dataset": "adt", "scene_id": "s"},
                           "exact_match": 1.0}]}
    assert extract_lmms_paired_records(valid)[0]["sample_id"] == "7"
    with pytest.raises(ValueError, match="explicit source"):
        extract_lmms_paired_records({"samples": [{"doc_id": 7, "doc": {"dataset": "adt"},
                                                   "exact_match": 1.0}]})


def test_signed_paired_records_reject_rehashed_record_forgery(tmp_path):
    producer = tmp_path / "run_matched_vsibench_eval.py"
    producer.write_text("# canonical producer\n", encoding="utf-8")
    raw_paths = {}
    records = {}
    for arm, score in (("a0", 1.0), ("a1o_drop", 0.0)):
        raw = tmp_path / f"{arm}.json"
        raw.write_text(json.dumps({"samples": [{
            "doc_id": "q", "doc": {"dataset": "adt", "scene_id": "s"},
            "score": score,
        }]}), encoding="utf-8")
        raw_paths[arm] = raw
        records[arm] = extract_lmms_paired_records(json.loads(raw.read_text()))
    payload = {"schema_version": "parta_vsibench_paired_records_receipt_v1",
               "status": "complete", "plan_sha256": "p" * 64,
               "raw_result_sha256": {arm: sha256_file(path) for arm, path in raw_paths.items()},
               "producer_script": str(producer.resolve()),
               "producer_script_sha256": sha256_file(producer),
               "producer_git_revision": "g" * 40,
               "records": records,
               "identity_sha256": stable_sha256([("adt", "s", "q")])}
    payload["receipt_payload_sha256"] = stable_sha256(payload)
    kwargs = {"plan_sha256": "p" * 64, "raw_result_paths": raw_paths,
              "producer_path": producer, "producer_sha256": sha256_file(producer),
              "git_revision": "g" * 40}
    validate_paired_records_receipt(payload, **kwargs)
    payload["records"]["a1o_drop"][0]["score"] = 0.5
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256")
    payload["receipt_payload_sha256"] = stable_sha256(unsigned)
    with pytest.raises(ValueError, match="hash-bound raw results"):
        validate_paired_records_receipt(payload, **kwargs)
