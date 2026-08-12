import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from parta.worker_trust import validate_python_worker
from scripts.parta.audit_a1o_drop_load import producer_record
from scripts.parta.audit_three_source_validator import validate_recomputed_summary
from scripts.parta.audit_formal_startup import (validate_single_checkpoint_artifact,
                                                validate_startup_configs,
                                                checkpoint_artifact,
                                                validate_startup_input)
from parta.provenance import sha256_file


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
