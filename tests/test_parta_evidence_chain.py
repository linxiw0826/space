import sys
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from parta.worker_trust import validate_python_worker, validate_torchrun_worker
from scripts.parta.audit_a1o_drop_load import producer_record
from parta.resource_profile_contract import (LAMBDA_STATE, normalize_profile_worker_argv,
    checkpoint_artifact_identity, normalized_contract_sha256,
    validate_preexecution_profile, validate_resolved_profile)
from scripts.parta.freeze_pretrain_config import validate_profile_selected_config
from scripts.parta.run_resource_profile import (collect_rank_failure_evidence,
                                                 run_with_timeout)
from scripts.parta import train_parta as train_parta_script
from scripts.parta.audit_three_source_validator import validate_recomputed_summary
from scripts.parta.audit_formal_startup import (validate_single_checkpoint_artifact,
                                                validate_startup_configs,
                                                checkpoint_artifact,
                                                validate_startup_input)
from parta.provenance import sha256_file, stable_sha256


def _record(script, sha="a" * 64, revision="b" * 40):
    return {"python_executable": sys.executable, "script_path": str(script.resolve()),
            "script_sha256": sha, "git_revision": revision}


def test_head_free_audit_producer_record_has_real_canonical_schema():
    record = producer_record("c" * 40)
    assert set(record) == {"path", "sha256", "git_revision"}
    assert Path(record["path"]).resolve().name == "audit_a1o_drop_load.py"
    assert record["sha256"] == sha256_file(Path(record["path"]))
    assert record["git_revision"] == "c" * 40


def test_validator_recomputation_rejects_tampered_counts_and_other_manifest():
    base = {
        "source_registry": ["adt", "hypersim", "scannetppv2"],
        "manifest_rows_sha256": "a" * 64, "total_scenes": 3, "total_qa": 9,
        "frozen_source_inventory": {}, "frozen_total_inventory": {"qa": 9, "scenes": 3},
        "scene_intersections": {"train_val": []}, "sources": {},
    }
    validate_recomputed_summary(base, dict(base))
    tampered = dict(base, total_qa=10)
    with pytest.raises(ValueError, match="recomputation"):
        validate_recomputed_summary(base, tampered)
    other_manifest = dict(base, manifest_rows_sha256="b" * 64)
    with pytest.raises(ValueError, match="recomputation"):
        validate_recomputed_summary(base, other_manifest)


def test_formal_startup_rejects_handwritten_initialization_and_config_drift():
    digest = "d" * 64
    configs = {arm: {"arm": arm, "state_head_enabled": arm == "a1o",
                     "start_step": 0, "initialization_sha256": digest, "lr": 1e-5}
               for arm in ("a0", "a1o")}
    validate_startup_configs(configs, digest)
    forged = {arm: dict(value) for arm, value in configs.items()}
    forged["a1o"]["initialization_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="matched at step zero"):
        validate_startup_configs(forged, digest)
    forged = {arm: dict(value) for arm, value in configs.items()}
    forged["a1o"]["lr"] = 2e-5
    with pytest.raises(ValueError, match="matched at step zero"):
        validate_startup_configs(forged, digest)


def test_t0a_single_checkpoint_artifact_uses_manifest_digest(tmp_path):
    checkpoint = tmp_path / "t0-a.pt"
    checkpoint.write_bytes(b"checkpoint")
    record = {"name": checkpoint.name, "size_bytes": checkpoint.stat().st_size,
              "sha256": sha256_file(checkpoint)}
    payload = {"mode": "no_index_explicit_manifest", "config_files": [],
               "ordered_shards": [record]}
    artifact = {**payload, "index": None, "artifact_sha256": stable_sha256(payload)}
    validate_single_checkpoint_artifact(artifact, checkpoint)
    forged = dict(artifact, artifact_sha256=sha256_file(checkpoint))
    with pytest.raises(ValueError, match="artifact manifest"):
        validate_single_checkpoint_artifact(forged, checkpoint)


def test_model_directory_artifact_uses_explicit_manifest_digest(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    artifact = checkpoint_artifact(model)
    assert artifact["mode"] == "no_index_explicit_manifest"
    assert [row["name"] for row in artifact["ordered_shards"]] == ["model.safetensors"]
    payload = {key: artifact[key] for key in ("mode", "config_files", "ordered_shards")}
    assert artifact["artifact_sha256"] == stable_sha256(payload)
    item = {"path": str(model.resolve()), "artifact": artifact,
            "artifact_sha256": artifact["artifact_sha256"]}
    validate_startup_input("guide", item)
    forged = dict(item, artifact_sha256="f" * 64)
    with pytest.raises(ValueError, match="artifact manifest"):
        validate_startup_input("guide", forged)


def test_worker_trust_rejects_shell_wrapper_duplicate_script_and_wrong_commit(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    suffix = ["--engineering-mode", "resource_profile"]
    valid = [sys.executable, str(script.resolve()), *suffix]
    validate_python_worker(_record(script), valid, script=script, script_sha256="a" * 64,
                           git_revision="b" * 40, engineering_mode="resource_profile",
                           allowed_value_flags={"--engineering-mode": "resource_profile"},
                           source_registry=())
    for argv, record in (
        (["bash", "-c", *valid], _record(script)),
        ([*valid, str(script.resolve())], _record(script)),
        (valid, _record(script, revision="c" * 40)),
    ):
        with pytest.raises(ValueError, match="untrusted"):
            validate_python_worker(record, argv, script=script, script_sha256="a" * 64,
                                   git_revision="b" * 40,
                                   engineering_mode="resource_profile",
                                   allowed_value_flags={"--engineering-mode": "resource_profile"},
                                   source_registry=())


@pytest.mark.parametrize("suffix", [
    ["--engineering-mode", "resource_profile", "--unknown", "x"],
    ["--engineering-mode", "resource_profile", "--engineering-mode", "resource_profile"],
    ["--engineering-mode"],
])
def test_worker_trust_rejects_unknown_duplicate_and_missing_value(tmp_path, suffix):
    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    with pytest.raises(ValueError, match="untrusted"):
        validate_python_worker(
            _record(script), [sys.executable, str(script.resolve()), *suffix],
            script=script, script_sha256="a" * 64, git_revision="b" * 40,
            engineering_mode="resource_profile",
            allowed_value_flags={"--engineering-mode": "resource_profile"},
            source_registry=(),
        )


def test_worker_trust_requires_three_unique_source_roots(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    roots = ["adt=/d/a", "hypersim=/d/h", "scannetppv2=/d/s"]
    manifest_report = tmp_path / "manifest_report.json"
    manifest_report.write_text(__import__("json").dumps({"exact_canonical_inputs": {
        source: {"root": path} for source, path in
        (("adt", "/d/a"), ("hypersim", "/d/h"), ("scannetppv2", "/d/s"))
    }}))
    argv = [sys.executable, str(script.resolve()), "--engineering-mode", "matched_runner",
            "--manifest-report", str(manifest_report)]
    for value in roots:
        argv.extend(["--source-root", value])
    validate_python_worker(
        _record(script), argv, script=script, script_sha256="a" * 64,
        git_revision="b" * 40, engineering_mode="matched_runner",
        allowed_value_flags={"--engineering-mode": "matched_runner", "--source-root": None,
                             "--manifest-report": str(manifest_report)},
    )
    forged = argv[:-1] + ["adt=/d/other"]
    with pytest.raises(ValueError, match="untrusted"):
        validate_python_worker(
            _record(script), forged, script=script, script_sha256="a" * 64,
            git_revision="b" * 40, engineering_mode="matched_runner",
            allowed_value_flags={"--engineering-mode": "matched_runner", "--source-root": None,
                                 "--manifest-report": str(manifest_report)},
        )


def test_torchrun_worker_requires_exact_four_rank_prefix(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("pass\n")
    record = _record(script)
    argv = [
        sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4",
        str(script.resolve()), "--engineering-mode", "resource_profile",
    ]
    validate_torchrun_worker(
        record, argv, script=script, script_sha256="a" * 64,
        git_revision="b" * 40, engineering_mode="resource_profile",
        allowed_value_flags={"--engineering-mode": "resource_profile"},
        source_registry=(),
    )
    for bad in (argv[:4] + ["3", *argv[5:]], [sys.executable, str(script), *argv[6:]]):
        with pytest.raises(ValueError, match="untrusted"):
            validate_torchrun_worker(
                record, bad, script=script, script_sha256="a" * 64,
                git_revision="b" * 40, engineering_mode="resource_profile",
                allowed_value_flags={"--engineering-mode": "resource_profile"},
                source_registry=(),
            )


def test_profile_normalized_contract_detects_drift_and_lambda():
    base = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4",
            "/repo/train.py", "--arm", "a1o", "--manifest", "/m",
            "--manifest-report", "/r", "--media-root", "/media",
            "--model-path", "/model", "--vggt-path", "/vggt", "--seed", "42",
            "--learning-rate", "2e-5", "--weight-decay", "0", "--lambda-state",
            str(LAMBDA_STATE), "--max-grad-norm", "1", "--gradient-accumulation-steps",
            "1", "--dtype", "bfloat16", "--num-workers", "4", "--engineering-subset",
            "/subset", "--engineering-mode", "resource_profile", "--required-frame-count",
            "32", "--source-root", "adt=/a", "--source-root", "hypersim=/h",
            "--source-root", "scannetppv2=/s", "--gradient-checkpointing", "--dry-run"]
    ddp = normalize_profile_worker_argv([*base, "--distributed-strategy", "ddp"])
    fsdp = normalize_profile_worker_argv([*base, "--distributed-strategy", "fsdp"])
    assert ddp == fsdp
    drift = list(base)
    drift[drift.index("--learning-rate") + 1] = "3e-5"
    assert normalize_profile_worker_argv(drift) != ddp
    bad_lambda = list(base)
    bad_lambda[bad_lambda.index("--lambda-state") + 1] = "0.05"
    with pytest.raises(ValueError, match="lambda_state"):
        normalize_profile_worker_argv(bad_lambda)


def test_freeze_rejects_selected_profile_field_tampering():
    contract = {
        "learning_rate": "2e-5", "weight_decay": "0", "lambda_state": str(LAMBDA_STATE),
        "max_grad_norm": "1", "gradient_accumulation_steps": "2", "dtype": "bfloat16",
        "num_workers": "4", "gradient_checkpointing": True,
        "effective_global_batch_size": 8,
    }
    report = {"result": {"recommendation": {"selected_strategy": "fsdp"},
                         "measurements": [{"distributed_strategy": "fsdp",
                                           "normalized_execution_contract": contract}]}}
    resolved = {
        "distributed_strategy": "fsdp", "world_size": 4, "learning_rate": 2e-5,
        "weight_decay": 0.0, "lambda_state": LAMBDA_STATE, "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 2, "dtype": "bfloat16", "num_workers": 4,
        "gradient_checkpointing": True, "per_rank_batch_size": 1,
        "effective_global_batch_size": 8,
    }
    validate_profile_selected_config(report, resolved)
    for key, bad in (("lambda_state", 0.05), ("gradient_checkpointing", False),
                     ("effective_global_batch_size", 4)):
        with pytest.raises(ValueError, match="differs"):
            validate_profile_selected_config(report, {**resolved, key: bad})


def test_profile_worker_timeout_is_terminated_and_structured(tmp_path):
    evidence = tmp_path / "timeout.json"
    child_pid = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,time,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    )
    with pytest.raises(RuntimeError, match="timed out"):
        run_with_timeout(
            [sys.executable, "-c", code, str(child_pid)], 1, evidence, "ddp"
        )
    payload = __import__("json").loads(evidence.read_text())
    assert payload["schema_version"] == "parta_profile_timeout_v1"
    assert payload["strategy"] == "ddp"
    assert payload["terminated"] is True
    pid = int(child_pid.read_text())
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("timeout left a torchrun child process alive")


def test_training_failure_coordination_branches_by_engineering_mode(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_parta_script, "synchronize_failure",
                        lambda failed, context: calls.append(failed) or failed)
    context = SimpleNamespace(world_size=4, rank=0, local_rank=0)
    profile = SimpleNamespace(engineering_mode="resource_profile", output_dir=tmp_path)
    error = RuntimeError("profile failure")
    with pytest.raises(RuntimeError, match="profile failure"):
        train_parta_script._coordinate_step_failure(error, "train_step", profile, context)
    assert calls == []
    assert (tmp_path / "rank_failures/rank-0.json").is_file()

    formal = SimpleNamespace(engineering_mode=None, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="formal failure"):
        train_parta_script._coordinate_step_failure(
            RuntimeError("formal failure"), "train_step", formal, context
        )
    assert calls == [True]
    monkeypatch.setattr(train_parta_script, "synchronize_failure",
                        lambda failed, context: True)
    with pytest.raises(RuntimeError, match="peer rank failed"):
        train_parta_script._coordinate_step_failure(None, "load_collate", formal, context)


def test_profile_rejects_stderr_only_oom_without_real_rank_artifact(tmp_path):
    run_dir = tmp_path / "stderr-only-oom"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="OOM evidence|real on-disk"):
        collect_rank_failure_evidence(run_dir)


def _write_rank_oom(run_dir, *, rank=0, schema="parta_rank_failure_v1"):
    path = run_dir / "rank_failures" / "rank-0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps({
        "schema_version": schema, "rank": rank, "local_rank": 0,
        "stage": "train_step", "reason": "CUDA out of memory", "oom": True,
        "device_name": "NVIDIA H20", "total_memory_bytes": 100,
        "peak_allocated_bytes": 99, "peak_reserved_bytes": 100, "finite": None,
    }))


def test_profile_accepts_one_real_oom_and_three_peer_placeholders(tmp_path):
    _write_rank_oom(tmp_path)
    rows = collect_rank_failure_evidence(tmp_path)
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["oom"] is True
    assert [row["oom"] for row in rows[1:]] == [None, None, None]


@pytest.mark.parametrize("field,value", [
    ("rank", 3),
    ("schema", "forged_rank_failure_v0"),
])
def test_profile_rejects_tampered_real_oom_rank_identity(tmp_path, field, value):
    kwargs = {field: value}
    _write_rank_oom(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="rank IDs|structured rank failure"):
        collect_rank_failure_evidence(tmp_path)


def test_top_level_failure_preserves_specific_rank_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    specific = RuntimeError("specific")
    train_parta_script._write_rank_failure(specific, "train_step", tmp_path)
    rank_path = tmp_path / "rank_failures/rank-0.json"
    before = rank_path.read_bytes()
    train_parta_script._write_top_level_rank_failure_if_absent(
        RuntimeError("outer"), tmp_path
    )
    assert rank_path.read_bytes() == before
    assert __import__("json").loads(before)["stage"] == "train_step"


def test_preexecution_profile_rejects_missing_hash_path_and_model_tamper(tmp_path):
    files = {name: tmp_path / name for name in ("manifest", "report", "subset")}
    for path in files.values():
        path.write_text(path.name)
    model, vggt = tmp_path / "model", tmp_path / "vggt"
    for root in (model, vggt):
        root.mkdir(); (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"weights")
    argv = ["python", "train.py", "--arm", "a1o", "--manifest", str(files["manifest"]),
            "--manifest-report", str(files["report"]), "--media-root", str(tmp_path),
            "--model-path", str(model), "--vggt-path", str(vggt), "--seed", "42",
            "--learning-rate", "2e-5", "--weight-decay", "0", "--lambda-state",
            str(LAMBDA_STATE), "--max-grad-norm", "1", "--gradient-accumulation-steps", "1",
            "--dtype", "bfloat16", "--num-workers", "4", "--engineering-subset",
            str(files["subset"]), "--engineering-mode", "resource_profile",
            "--required-frame-count", "32", "--gradient-checkpointing", "--dry-run"]
    contract = normalize_profile_worker_argv(argv)
    payload = {
        "schema_version": "parta_profile_preexecution_matched_v1",
        "status": "complete_preexecution", "distributed_strategy": "ddp",
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": normalized_contract_sha256(contract),
        "manifest": {"path": str(files["manifest"].resolve()),
                     "sha256": sha256_file(files["manifest"])},
        "manifest_report": {"path": str(files["report"].resolve()),
                            "sha256": sha256_file(files["report"])},
        "engineering_subset": {"path": str(files["subset"].resolve()),
                               "sha256": sha256_file(files["subset"])},
        "guide": checkpoint_artifact_identity(model),
        "vggt": checkpoint_artifact_identity(vggt),
    }
    validate_preexecution_profile(payload, argv, manifest=files["manifest"],
        manifest_report=files["report"], engineering_subset=files["subset"])
    for mutate in (
        lambda value: value.pop("guide"),
        lambda value: value["manifest"].update(sha256="0" * 64),
        lambda value: value["manifest_report"].update(path="/wrong"),
        lambda value: value["guide"].update(artifact_sha256="f" * 64),
    ):
        forged = __import__("copy").deepcopy(payload); mutate(forged)
        with pytest.raises(ValueError):
            validate_preexecution_profile(forged, argv, manifest=files["manifest"],
                manifest_report=files["report"], engineering_subset=files["subset"])


def test_profile_to_coverage_reopen_contract_is_consistent(tmp_path):
    files = {name: tmp_path / name for name in ("manifest", "report", "subset")}
    for path in files.values():
        path.write_text(path.name)
    model, vggt = tmp_path / "model", tmp_path / "vggt"
    for root in (model, vggt):
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"weights")
    media = tmp_path / "media"
    media.mkdir()
    source_root = tmp_path / "adt"
    source_root.mkdir()
    argv = [
        "python", "train.py", "--arm", "a1o",
        "--manifest", str(files["manifest"]),
        "--manifest-report", str(files["report"]),
        "--source-root", f"adt={source_root}", "--media-root", str(media),
        "--model-path", str(model), "--vggt-path", str(vggt), "--seed", "42",
        "--learning-rate", "2e-5", "--weight-decay", "0", "--lambda-state",
        str(LAMBDA_STATE), "--max-grad-norm", "1",
        "--gradient-accumulation-steps", "1", "--dtype", "bfloat16",
        "--num-workers", "4", "--engineering-subset", str(files["subset"]),
        "--engineering-mode", "resource_profile", "--required-frame-count", "32",
        "--gradient-checkpointing", "--dry-run",
    ]
    contract = normalize_profile_worker_argv(argv)
    preflight = {
        "schema_version": "parta_profile_preexecution_matched_v1",
        "status": "complete_preexecution", "distributed_strategy": "fsdp",
        "normalized_execution_contract": contract,
        "normalized_execution_contract_sha256": normalized_contract_sha256(contract),
        "manifest": {"path": str(files["manifest"].resolve()),
                     "sha256": sha256_file(files["manifest"])},
        "manifest_report": {"path": str(files["report"].resolve()),
                            "sha256": sha256_file(files["report"])},
        "engineering_subset": {"path": str(files["subset"].resolve()),
                               "sha256": sha256_file(files["subset"])},
        "guide": checkpoint_artifact_identity(model),
        "vggt": checkpoint_artifact_identity(vggt),
    }
    reopened = validate_preexecution_profile(
        preflight, argv, manifest=files["manifest"], manifest_report=files["report"],
        engineering_subset=files["subset"],
    )
    resolved = {
        "lambda_state": LAMBDA_STATE, "per_rank_batch_size": 1,
        "effective_global_batch_size": 4, "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True, "distributed_strategy": "fsdp",
        "learning_rate": 2e-5, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "dtype": "bfloat16", "num_workers": 4, "seed": 42,
        "required_frame_count": 32, "engineering_mode": "resource_profile",
        "dry_run": True, "world_size": 4,
        "manifest": str(files["manifest"].resolve()),
        "manifest_report": str(files["report"].resolve()),
        "media_root": str(media.resolve()), "model_path": str(model.resolve()),
        "vggt_path": str(vggt.resolve()),
        "engineering_subset": str(files["subset"].resolve()),
        "source_roots": {"adt": str(source_root.resolve())},
    }
    validate_resolved_profile(resolved, reopened, "fsdp")
    with pytest.raises(ValueError, match="resolved config"):
        validate_resolved_profile({**resolved, "required_frame_count": 24}, reopened, "fsdp")
