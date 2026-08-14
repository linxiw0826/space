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


def test_independent_profile_requires_four_h20_ranks_and_32_frame_worst_case():
    contract = {"lambda_state": "0.02150771327925621", "per_rank_batch_size": 1}
    report = {
        "status": "complete_passed", "formal_gpu_evidence": True,
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": stable_sha256(contract),
        "measurements": [{
            "frame_count": 32, "distributed_strategy": strategy,
            "peak_memory_bytes": 80, "peak_reserved_memory_bytes": 85,
            "total_memory_bytes": 100,
            "step_time_seconds": 1.0, "throughput_samples_per_second": 1.0,
            "batch_size": 1, "per_rank_batch_size": 1, "gradient_accumulation_steps": 1,
            "normalized_execution_contract": contract,
            "normalized_execution_contract_sha256": stable_sha256(contract),
            "forward_backward_steps": 1, "oom": False, "finite": True, "world_size": 4,
            "per_rank_peak_memory_bytes": [
                {"rank": rank, "device_name": "NVIDIA H20",
                 "peak_allocated_bytes": 80, "peak_reserved_bytes": 85,
                 "total_memory_bytes": 100}
                for rank in range(4)
            ],
        } for strategy in ("ddp", "fsdp")],
        "recommendation": {
            "status": "provisional_not_frozen", "frame_count": 32,
            "selected_strategy": "ddp",
            "selection_rule": (
                "max_throughput_then_min_max_rank_allocated_then_strategy_lexical_v1"
            ),
        },
    }
    assert validate_phase_report("resource_profile", report, ProvisionalGateDefaults()) == []
    report["measurements"][0]["forward_backward_steps"] = 0
    assert "measurement_no_forward_backward" in validate_phase_report(
        "resource_profile", report, ProvisionalGateDefaults()
    )


def test_profile_defaults_reject_legacy_multi_point_contract():
    with pytest.raises(ValueError, match="only the frozen 32-frame"):
        ProvisionalGateDefaults(profile_frame_counts=(16, 24, 32)).validate()


def test_profile_allows_one_closed_oom_candidate():
    contract = {"lambda_state": "0.02150771327925621", "per_rank_batch_size": 1}
    successful = {
        "frame_count": 32, "distributed_strategy": "fsdp", "peak_memory_bytes": 80,
        "peak_reserved_memory_bytes": 85, "total_memory_bytes": 100,
        "step_time_seconds": 1.0, "throughput_samples_per_second": 1.0,
        "batch_size": 1, "per_rank_batch_size": 1, "gradient_accumulation_steps": 1,
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": stable_sha256(contract),
        "forward_backward_steps": 1, "oom": False, "finite": True, "world_size": 4,
        "per_rank_peak_memory_bytes": [
            {"rank": rank, "device_name": "NVIDIA H20", "peak_allocated_bytes": 80,
             "peak_reserved_bytes": 85, "total_memory_bytes": 100}
            for rank in range(4)
        ],
    }
    oom = {
        "frame_count": 32, "distributed_strategy": "ddp", "peak_memory_bytes": None,
        "peak_reserved_memory_bytes": None, "total_memory_bytes": 100,
        "step_time_seconds": None, "throughput_samples_per_second": None,
        "batch_size": 1, "per_rank_batch_size": 1, "gradient_accumulation_steps": 1,
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": stable_sha256(contract),
        "forward_backward_steps": 0, "oom": True, "finite": None, "world_size": 4,
        "per_rank_peak_memory_bytes": [
            {"schema_version": "parta_rank_failure_v1", "rank": rank,
             "stage": "torchrun_peer_termination", "reason": "peer terminated",
             "oom": rank == 2, "peak_allocated_bytes": None,
             "peak_reserved_bytes": None, "total_memory_bytes": None}
            for rank in range(4)
        ], "oom_evidence": {"error": "out of memory"},
    }
    report = {
        "status": "complete_passed", "formal_gpu_evidence": True,
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": stable_sha256(contract),
        "measurements": [oom, successful],
        "recommendation": {
            "status": "provisional_not_frozen", "frame_count": 32,
            "selected_strategy": "fsdp",
            "selection_rule": (
                "max_throughput_then_min_max_rank_allocated_then_strategy_lexical_v1"
            ),
        },
    }
    assert validate_phase_report("resource_profile", report, ProvisionalGateDefaults()) == []


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
