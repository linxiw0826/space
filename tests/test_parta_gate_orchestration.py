import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.gate_orchestration import (
    ENGINEERING_COVERAGE_REQUIREMENTS,
    FORMAL_SOURCE_REGISTRY,
    PHASES,
    ProvisionalGateDefaults,
    unified_gate_report,
    validate_formal_training_authorization,
    validate_phase_report,
)
from parta.provenance import sha256_file, stable_sha256


def test_d62_registry_and_mandatory_matrix_have_no_smoke():
    assert FORMAL_SOURCE_REGISTRY == ("adt", "hypersim", "scannetppv2")
    assert PHASES == ("t0_b", "engineering_coverage", "resource_profile")
    assert "guide_smoke" not in PHASES
    assert "a1o_drop_head_free_val_load" in ENGINEERING_COVERAGE_REQUIREMENTS
    assert "matched_a0_a1o_real_runner_steps" in ENGINEERING_COVERAGE_REQUIREMENTS


def test_engineering_coverage_requires_exact_machine_matrix():
    matrix = {name: {"artifact_path": "/evidence/report.json",
                     "artifact_sha256": "a" * 64, "semantic_summary": {}}
              for name in ENGINEERING_COVERAGE_REQUIREMENTS}
    report = {
        "status": "complete_passed", "formal_gpu_evidence": True,
        "schema_version": "parta_engineering_coverage_v2",
        "coverage_matrix": matrix,
        "engineering_subset": {
            "path": "/data/engineering.jsonl", "sha256": "b" * 64,
            "formal_train_reuse": {
                "subset_rows_remain_in_train_manifest": True,
                "source_balanced_weights_unchanged": True,
                "extra_sampling_weight": False,
            },
            "transaction_promotion": {
                "promotable_to_formal_training": False,
                "discard": ["model", "optimizer", "scheduler", "rng", "sampler"],
                "formal_restart_optimizer_step": 0,
            },
        },
    }
    assert validate_phase_report("engineering_coverage", report, ProvisionalGateDefaults()) == []
    del matrix["checkpoint_save_resume"]
    assert "coverage_matrix_exact" in validate_phase_report(
        "engineering_coverage", report, ProvisionalGateDefaults()
    )


def test_independent_profile_requires_real_forward_backward_and_all_points():
    report = {
        "status": "complete_passed", "formal_gpu_evidence": True,
        "measurements": [{
            "frame_count": count, "peak_memory_bytes": 80, "total_memory_bytes": 100,
            "step_time_seconds": 1.0, "throughput_samples_per_second": 1.0,
            "batch_size": 1, "gradient_accumulation_steps": 1,
            "forward_backward_steps": 1, "oom": False,
        } for count in (16, 24, 32)],
        "recommendation": {"status": "provisional_not_frozen", "frame_count": 32},
    }
    assert validate_phase_report("resource_profile", report, ProvisionalGateDefaults()) == []
    report["measurements"][0]["forward_backward_steps"] = 0
    assert "measurement_no_forward_backward" in validate_phase_report(
        "resource_profile", report, ProvisionalGateDefaults()
    )


def test_unified_gate_fails_when_any_new_phase_is_missing():
    statuses = [{"phase": phase, "status": "complete_passed", "manifest_sha256": "d" * 64}
                for phase in PHASES]
    assert unified_gate_report(statuses, ProvisionalGateDefaults())["status"] == "complete_passed"
    failed = unified_gate_report(statuses[:-1], ProvisionalGateDefaults())
    assert failed["status"] == "complete_failed"
    assert failed["missing_phases"] == ["resource_profile"]


def test_formal_authorization_rejects_missing_lifecycle_and_extra_scannet(tmp_path):
    resolved = tmp_path / "resolved.json"
    expected = {
        "manifest_sha256": "1" * 64, "manifest_report_sha256": "2" * 64,
        "matched_contract_sha256": "3" * 64, "profile_report_sha256": "4" * 64,
        "guide_artifact_sha256": "5" * 64, "vggt_artifact_sha256": "6" * 64,
        "code_revision": "7" * 40, "training_config": {"max_steps": 10},
    }
    resolved.write_text(json.dumps(expected))
    freeze = tmp_path / "freeze.json"
    lifecycle = {
        "subset_is_train_internal": True, "subset_extra_weight": False,
        "model_promotable": False, "optimizer_promotable": False,
        "scheduler_promotable": False, "rng_promotable": False,
        "sampler_promotable": False, "formal_arms_start_at_step": 0,
        "formal_arms_share_initialization": True,
    }
    freeze_payload = {
        "resolved_training_config_path": str(resolved),
        "resolved_training_config_sha256": sha256_file(resolved),
        "manifest_sha256": ["1" * 64], "profile_report_sha256": "4" * 64,
        "formal_source_registry": list(FORMAL_SOURCE_REGISTRY),
        "engineering_lifecycle": lifecycle,
        "formal_startup_contract": {
            "schema_version": "parta_formal_startup_v1",
            "arms": {
                "a0": {"start_step": 0, "initialization_sha256": "9" * 64},
                "a1o": {"start_step": 0, "initialization_sha256": "9" * 64},
            },
        },
    }
    freeze.write_text(json.dumps(freeze_payload))
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({
        "schema_version": "parta_unified_pretrain_gate_v1", "status": "complete_passed",
        "formal_gpu_evidence": True, "formal_config_frozen": True,
        "training_authorized_by_this_artifact": True,
        "freeze_artifact_sha256": sha256_file(freeze),
        "frozen_config_artifact_sha256": stable_sha256(freeze_payload),
    }))
    kwargs = {**expected, "resolved_training_config": {
        "artifact_path": str(resolved), "training_config": expected["training_config"]}}
    kwargs.pop("training_config")
    validate_formal_training_authorization(gate, freeze, **kwargs)
    freeze_payload["formal_source_registry"].append("scannet")
    freeze.write_text(json.dumps(freeze_payload))
    with pytest.raises(ValueError, match="three-source"):
        validate_formal_training_authorization(gate, freeze, **kwargs)
