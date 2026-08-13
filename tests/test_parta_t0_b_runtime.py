import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.t0_b_runtime import (
    T0BBatchObservation,
    T0BThresholds,
    build_t0_b_report,
    finalize_t0_b_report,
    validate_t0_a_initialization_transaction,
    parameter_gradient_norm,
    nested_state_digest,
    t0_a_model_state_digest,
    validate_t0_a_model_state_restore,
    validate_t0_a_code_compatibility,
    validate_t0_b_runtime_identity,
    T0_A_COMPATIBILITY_BASE_REVISION,
    T0_A_APPROVED_SEMANTIC_TREE_SHA256,
    T0_B_RUNTIME_APPROVAL_PATH,
    _assert_surface_clean,
)
from parta.checkpoint import (
    ResumeContract, capture_rng_state, load_training_checkpoint, save_training_checkpoint,
)
from parta.provenance import stable_sha256


def _write_runtime_approval(root, revision, tree, **overrides):
    payload = {
        "schema_version": "parta_t0_b_runtime_approval_v1",
        "approved_revision": revision,
        "runtime_tree_sha256": tree,
    }
    payload.update({key: value for key, value in overrides.items() if key in payload})
    record = dict(payload)
    record["payload_sha256"] = stable_sha256(payload)
    if "payload_sha256" in overrides:
        record["payload_sha256"] = overrides["payload_sha256"]
    path = root / T0_B_RUNTIME_APPROVAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _observations(count=30, *, bad_index=None):
    sources = ("adt", "hypersim", "scannetppv2")
    rows = []
    for index in range(count):
        value = 0.0 if index == bad_index else 1.0
        rows.append(T0BBatchObservation(
            batch_index=index, qa_id=f"q{index}", source_dataset=sources[index % 3],
            qa_loss=1.0, state_loss=2.0, qa_gradient_norm=value,
            state_gradient_norm=1.0, shared_gradient_parameter_count=1,
            head_gradient_parameter_count=1, enabled_components=("existence", "category"),
            masked_components=("center", "extent", "visibility"),
            component_losses={"existence": 1.0, "category": 1.0}, matching_valid=True,
            component_valid_counts={"existence": 1, "category": 1, "center": 0, "extent": 0, "visibility": 0},
            matched_pairs=1, gt_objects=1, exact_frame_consistent=True,
            actual_frame_count=16,
        ))
    return rows


def test_gpu_report_is_machine_decidable_and_source_balanced(tmp_path):
    report = build_t0_b_report(
        _observations(), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert report["status"] == "complete_passed"
    assert report["source_batch_counts"] == {"adt": 10, "hypersim": 10, "scannetppv2": 10}
    assert report["gradient_calibration"]["lambda_state_candidate"] == pytest.approx(0.1)
    path = tmp_path / "report.json"
    finalize_t0_b_report(report, str(path))
    assert json.loads(path.read_text())["formal_gpu_evidence"] is True


def test_cpu_mock_never_becomes_formal_pass():
    report = build_t0_b_report(
        _observations(), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="awaiting_gpu",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert report["status"] == "awaiting_gpu"
    assert report["formal_gpu_evidence"] is False


def test_failed_gpu_gate_is_nonzero_at_finalize(tmp_path):
    report = build_t0_b_report(
        _observations(bad_index=0), requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    with pytest.raises(AssertionError, match="shared_qa_gradients"):
        finalize_t0_b_report(report, str(tmp_path / "failed.json"))


def test_t0_b_rejects_unbound_t0_a_initialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "t0.pt"
    torch.save({"model": {}}, checkpoint)
    checkpoint_sha = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema_version": "parta_t0_report_v1",
                                  "status": "complete_passed"}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "complete", "experiment": "parta-t0-a",
                                  "code_revision": "abc", "checkpoint_sha256": "s" * 64}))
    provenance = tmp_path / "provenance.json"
    payload = {
        "status": "complete_passed", "a1_checkpoint_role": "initialization_no_optimizer_updates",
        "a1_checkpoint_optimizer_steps": 0,
        "a1_checkpoint_artifact": {"ordered_shards": [{"sha256": checkpoint_sha}]},
        "a1_checkpoint_state_sha256": "s" * 64,
        "parameter_sha256_before_backward": "s" * 64,
        "parameter_sha256_after_backward": "s" * 64,
        "git_revision": "abc", "checkpoint_sha256": "g" * 64,
        "vggt_checkpoint_sha256": "v" * 64,
        "manifest_sha256": {"adt": "a" * 64, "hypersim": "h" * 64},
        "exact_frame_binding_sha256": "e" * 64,
    }
    provenance.write_text(json.dumps(payload))
    kwargs = dict(
        report_path=report, provenance_path=provenance, run_status_path=status,
        checkpoint_path=checkpoint, current_code_revision="abc",
        guide_artifact_sha256="g" * 64, vggt_artifact_sha256="v" * 64,
        current_manifest_inputs={
            "adt": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "a" * 64}}},
            "hypersim": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "h" * 64}}},
            "scannetppv2": {"files": {"qa_manifest_exact_verified.jsonl": {"sha256": "p" * 64}}},
        },
        project_root=tmp_path,
    )
    monkeypatch.setattr(
        "parta.t0_b_runtime.validate_t0_a_code_compatibility",
        lambda **_kwargs: {"mode": "exact_revision"},
    )
    monkeypatch.setattr(
        "parta.t0_b_runtime.validate_t0_b_runtime_identity",
        lambda **_kwargs: {"tree_sha256": "r" * 64},
    )
    assert validate_t0_a_initialization_transaction(**kwargs)[
        "t0_a_checkpoint_optimizer_steps"
    ] == 0
    payload["a1_checkpoint_optimizer_steps"] = 1
    provenance.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="optimizer_steps"):
        validate_t0_a_initialization_transaction(**kwargs)
    payload["a1_checkpoint_optimizer_steps"] = 0
    provenance.write_text(json.dumps(payload))
    kwargs["current_manifest_inputs"]["adt"]["files"][
        "qa_manifest_exact_verified.jsonl"
    ]["sha256"] = "x" * 64
    with pytest.raises(ValueError, match="manifest.adt"):
        validate_t0_a_initialization_transaction(**kwargs)


def test_t0_a_code_compatibility_exact_revision_passes_clean_head(tmp_path, monkeypatch):
    def clean_git(_root, *arguments):
        if arguments[:2] == ("rev-parse", "HEAD"):
            return b"same\n"
        if arguments[0] in {"diff", "ls-files"}:
            return b""
        raise AssertionError(arguments)
    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", clean_git)
    result = validate_t0_a_code_compatibility(
        t0_a_revision="same", current_code_revision="same", project_root=tmp_path
    )
    assert result["mode"] == "exact_revision"


def test_t0_a_code_compatibility_accepts_reviewed_semantic_tree(tmp_path, monkeypatch):
    current = "c" * 40

    def fake_git(_root, *arguments):
        if arguments[:2] == ("rev-parse", "HEAD"):
            return f"{current}\n".encode()
        if arguments[0] in {"cat-file", "merge-base"}:
            return b""
        if arguments[0] in {"diff", "ls-files"}:
            return b""
        if arguments[0] == "ls-tree":
            return b"reviewed semantic tree"
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    monkeypatch.setattr(
        "parta.t0_b_runtime.T0_A_APPROVED_SEMANTIC_TREE_SHA256",
        __import__("hashlib").sha256(b"reviewed semantic tree").hexdigest(),
    )
    result = validate_t0_a_code_compatibility(
        t0_a_revision=T0_A_COMPATIBILITY_BASE_REVISION,
        current_code_revision=current,
        project_root=tmp_path,
    )
    assert result["mode"] == "reviewed_semantic_tree_v1"


def test_transaction_accepts_reviewed_revision_but_keeps_artifact_gates(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "t0.pt"
    torch.save({"model": {}}, checkpoint)
    checkpoint_sha = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"schema_version": "parta_t0_report_v1",
                                  "status": "complete_passed"}))
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "complete", "experiment": "parta-t0-a",
                                  "code_revision": T0_A_COMPATIBILITY_BASE_REVISION,
                                  "checkpoint_sha256": "s" * 64}))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "status": "complete_passed", "a1_checkpoint_role": "initialization_no_optimizer_updates",
        "a1_checkpoint_optimizer_steps": 0,
        "a1_checkpoint_artifact": {"ordered_shards": [{"sha256": checkpoint_sha}]},
        "a1_checkpoint_state_sha256": "s" * 64,
        "parameter_sha256_before_backward": "s" * 64,
        "parameter_sha256_after_backward": "s" * 64,
        "git_revision": T0_A_COMPATIBILITY_BASE_REVISION,
        "checkpoint_sha256": "g" * 64, "vggt_checkpoint_sha256": "v" * 64,
        "manifest_sha256": {"adt": "a" * 64, "hypersim": "h" * 64},
        "exact_frame_binding_sha256": "e" * 64,
    }))
    monkeypatch.setattr(
        "parta.t0_b_runtime.validate_t0_a_code_compatibility",
        lambda **_kwargs: {"mode": "reviewed_semantic_tree_v1"},
    )
    monkeypatch.setattr(
        "parta.t0_b_runtime.validate_t0_b_runtime_identity",
        lambda **_kwargs: {"tree_sha256": "r" * 64},
    )
    kwargs = {
        "report_path": report, "provenance_path": provenance,
        "run_status_path": status, "checkpoint_path": checkpoint,
        "current_code_revision": "c" * 40, "guide_artifact_sha256": "g" * 64,
        "vggt_artifact_sha256": "v" * 64, "project_root": tmp_path,
        "current_manifest_inputs": {
            "adt": {"files": {"qa": {"sha256": "a" * 64}}},
            "hypersim": {"files": {"qa": {"sha256": "h" * 64}}},
        },
    }
    assert validate_t0_a_initialization_transaction(**kwargs)[
        "t0_a_code_compatibility"
    ]["mode"] == "reviewed_semantic_tree_v1"
    kwargs["guide_artifact_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="guide_hash"):
        validate_t0_a_initialization_transaction(**kwargs)


def test_t0_a_code_compatibility_rejects_unknown_old_revision(tmp_path, monkeypatch):
    def forbidden_git(*_args, **_kwargs):
        raise AssertionError("unknown revision rejection must not access Git")

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", forbidden_git)
    with pytest.raises(ValueError, match="no reviewed compatibility"):
        validate_t0_a_code_compatibility(
            t0_a_revision="d" * 40,
            current_code_revision="c" * 40,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "failure", ["altered", "git_failure", "unstaged", "staged", "untracked"]
)
def test_t0_a_code_compatibility_rejects_altered_or_unverifiable_history(
    tmp_path, monkeypatch, failure
):
    current = "c" * 40

    def fake_git(_root, *arguments):
        if failure == "git_failure" and arguments[0] == "merge-base":
            raise ValueError("reviewed compatibility requires complete usable git history")
        if arguments[:2] == ("rev-parse", "HEAD"):
            return f"{current}\n".encode()
        if arguments[0] in {"cat-file", "merge-base"}:
            return b""
        if arguments[0] == "ls-files":
            return b"src/parta/t0.py\n" if failure == "untracked" else b""
        if arguments[0] == "diff":
            cached = "--cached" in arguments
            if failure == "unstaged" and not cached:
                return b"src/parta/t0.py\n"
            if failure == "staged" and cached:
                return b"src/parta/t0.py\n"
            return b""
        if arguments[0] == "ls-tree":
            return (
                b"altered semantic tree"
                if failure == "altered" and arguments[2] == current
                else b"reviewed semantic tree"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    message = {
        "git_failure": "usable git history",
        "altered": "differs from reviewed tree",
        "unstaged": "safety surface is dirty",
        "staged": "safety surface is dirty",
        "untracked": "safety surface is dirty",
    }[failure]
    monkeypatch.setattr(
        "parta.t0_b_runtime.T0_A_APPROVED_SEMANTIC_TREE_SHA256",
        __import__("hashlib").sha256(b"reviewed semantic tree").hexdigest(),
    )
    with pytest.raises(ValueError, match=message):
        validate_t0_a_code_compatibility(
            t0_a_revision=T0_A_COMPATIBILITY_BASE_REVISION,
            current_code_revision=current,
            project_root=tmp_path,
        )


def test_t0_b_runtime_identity_is_fail_closed_until_phase_two_freeze(
    tmp_path, monkeypatch
):
    _write_runtime_approval(tmp_path, None, None)
    monkeypatch.setattr(
        "parta.t0_b_runtime._git_bytes", lambda _root, *_arguments: b""
    )
    with pytest.raises(ValueError, match="has not been frozen"):
        validate_t0_b_runtime_identity(
            current_code_revision="c" * 40, project_root=tmp_path
        )


@pytest.mark.parametrize(
    "mutation", [None, "tree", "unstaged", "staged", "untracked"]
)
def test_t0_b_runtime_tree_identity_mechanism(tmp_path, monkeypatch, mutation):
    approved = "a" * 40
    current = "c" * 40
    tree_sha256 = __import__("hashlib").sha256(b"reviewed runtime tree").hexdigest()
    _write_runtime_approval(tmp_path, approved, tree_sha256)

    def fake_git(_root, *arguments):
        if arguments[:2] == ("rev-parse", "HEAD"):
            return f"{current}\n".encode()
        if arguments[0] in {"cat-file", "merge-base"}:
            return b""
        if arguments[0] == "diff":
            cached = "--cached" in arguments
            if mutation == "unstaged" and not cached:
                return b"src/parta/t0_b_runtime.py\n"
            if mutation == "staged" and cached:
                return b"scripts/parta/run_t0_b.py\n"
            return b""
        if arguments[0] == "ls-files":
            return b"scripts/parta/run_t0_b.py\n" if mutation == "untracked" else b""
        if arguments[0] == "ls-tree":
            revision = arguments[2]
            if mutation != "tree" or revision == approved:
                return b"reviewed runtime tree"
            return b"mutated runtime tree"
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    if mutation is None:
        result = validate_t0_b_runtime_identity(
            current_code_revision=current, project_root=tmp_path
        )
        assert result["tree_sha256"] == __import__("hashlib").sha256(
            b"reviewed runtime tree"
        ).hexdigest()
    elif mutation == "tree":
        with pytest.raises(ValueError, match="differs from reviewed runtime"):
            validate_t0_b_runtime_identity(
                current_code_revision=current, project_root=tmp_path
            )
    else:
        with pytest.raises(ValueError, match="dirty"):
            validate_t0_b_runtime_identity(
                current_code_revision=current, project_root=tmp_path
            )


def test_t0_b_runtime_identity_rejects_revision_head_disagreement(tmp_path, monkeypatch):
    _write_runtime_approval(tmp_path, "a" * 40, "b" * 64)
    monkeypatch.setattr(
        "parta.t0_b_runtime._git_bytes",
        lambda _root, *arguments: b"different\n"
        if arguments[:2] == ("rev-parse", "HEAD")
        else b"",
    )
    with pytest.raises(ValueError, match="differs from repository HEAD"):
        validate_t0_b_runtime_identity(
            current_code_revision="c" * 40, project_root=tmp_path
        )


@pytest.mark.parametrize(
    "case", ["invalid_json", "extra_key", "bad_digest", "bad_revision", "bad_tree"]
)
def test_t0_b_runtime_identity_rejects_invalid_approval_metadata(
    tmp_path, monkeypatch, case
):
    path = _write_runtime_approval(tmp_path, "a" * 40, "b" * 64)
    if case == "invalid_json":
        path.write_text("{", encoding="utf-8")
    elif case == "extra_key":
        record = json.loads(path.read_text())
        record["unexpected"] = True
        path.write_text(json.dumps(record), encoding="utf-8")
    elif case == "bad_digest":
        record = json.loads(path.read_text())
        record["payload_sha256"] = "0" * 64
        path.write_text(json.dumps(record), encoding="utf-8")
    elif case == "bad_revision":
        _write_runtime_approval(tmp_path, "not-a-revision", "b" * 64)
    else:
        _write_runtime_approval(tmp_path, "a" * 40, "not-a-tree")

    def fake_git(_root, *arguments):
        if arguments[0] in {"diff", "ls-files"}:
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    with pytest.raises(ValueError, match="approval metadata"):
        validate_t0_b_runtime_identity(
            current_code_revision="c" * 40, project_root=tmp_path
        )


@pytest.mark.parametrize("mutation", ["unstaged", "staged", "untracked"])
def test_t0_b_runtime_identity_rejects_dirty_approval_metadata(
    tmp_path, monkeypatch, mutation
):
    _write_runtime_approval(tmp_path, "a" * 40, "b" * 64)

    def fake_git(_root, *arguments):
        if arguments[0] == "diff":
            cached = "--cached" in arguments
            if mutation == "unstaged" and not cached:
                return f"{T0_B_RUNTIME_APPROVAL_PATH}\n".encode()
            if mutation == "staged" and cached:
                return f"{T0_B_RUNTIME_APPROVAL_PATH}\n".encode()
            return b""
        if arguments[0] == "ls-files":
            return (
                f"{T0_B_RUNTIME_APPROVAL_PATH}\n".encode()
                if mutation == "untracked"
                else b""
            )
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    with pytest.raises(ValueError, match="safety surface is dirty"):
        validate_t0_b_runtime_identity(
            current_code_revision="c" * 40, project_root=tmp_path
        )


def test_t0_b_runtime_identity_real_checkout_integration():
    project_root = Path(__file__).resolve().parents[1]
    approval = json.loads(
        (project_root / T0_B_RUNTIME_APPROVAL_PATH).read_text(encoding="utf-8")
    )
    if approval["approved_revision"] is None:
        pytest.skip("bootstrap 2a approval is deliberately not frozen yet")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    evidence = validate_t0_b_runtime_identity(
        current_code_revision=current,
        project_root=project_root,
    )
    assert evidence["approved_revision"] == approval["approved_revision"]
    assert evidence["tree_sha256"] == approval["runtime_tree_sha256"]
    assert evidence["approval_metadata"]["payload_sha256"] == approval[
        "payload_sha256"
    ]


@pytest.mark.parametrize(
    ("untracked", "accepted"),
    [
        ("src/parta/__pycache__/t0_b_runtime.cpython-310.pyc\n", True),
        ("src/parta/__pycache__/t0_b_runtime.pyo\n", True),
        ("src/parta/__pycache__/helper.py\n", False),
        (
            "src/parta/__pycache__/t0_b_runtime.cpython-310.pyc\n"
            "src/parta/unreviewed_helper.py\n",
            False,
        ),
        ("src/parta/t0_b_runtime.pyc\n", False),
        (f"{T0_B_RUNTIME_APPROVAL_PATH}\n", False),
        ("scripts/parta/run_t0_b.py\n", False),
    ],
)
def test_surface_cleanliness_excludes_only_generated_python_cache(
    tmp_path, monkeypatch, untracked, accepted
):
    def fake_git(_root, *arguments):
        if arguments[0] == "diff":
            return b""
        if arguments[0] == "ls-files":
            return untracked.encode()
        raise AssertionError(arguments)

    monkeypatch.setattr("parta.t0_b_runtime._git_bytes", fake_git)
    if accepted:
        evidence = _assert_surface_clean(tmp_path, ("src/parta",))
        assert evidence["untracked"] == []
        assert evidence["ignored_generated_python_cache"] == untracked.splitlines()
    else:
        with pytest.raises(ValueError, match="safety surface is dirty"):
            _assert_surface_clean(tmp_path, ("src/parta",))


def test_parameter_gradient_norm_counts_only_connected_parameters():
    connected = torch.nn.Parameter(torch.tensor(2.0))
    unused = torch.nn.Parameter(torch.tensor(3.0))
    norm, count = parameter_gradient_norm(connected.square(), (connected, unused), retain_graph=False)
    assert norm == pytest.approx(4.0)
    assert count == 1


def test_cli_cpu_mock_writes_awaiting_gpu_atomic_artifacts(tmp_path):
    output = tmp_path / "t0b"
    script = Path(__file__).resolve().parents[1] / "scripts" / "parta" / "run_t0_b.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output), "--cpu-mock", "--batches", "30"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((output / "t0_b_report.json").read_text())["status"] == "awaiting_gpu"
    assert json.loads((output / "run_status.json").read_text())["status"] == "awaiting_gpu"
    resolved = json.loads((output / "resolved_config.json").read_text())
    digest = stable_sha256(resolved)
    assert json.loads((output / "t0_b_report.json").read_text())["resolved_config_sha256"] == digest
    assert json.loads((output / "provenance.json").read_text())["resolved_config_sha256"] == digest
    assert json.loads((output / "run_status.json").read_text())["resolved_config_sha256"] == digest


def test_thresholds_are_cli_overridable_but_bounded():
    T0BThresholds(minimum_batches=20, maximum_batches=50).validate(20)
    with pytest.raises(ValueError, match="requested T0-B batches"):
        T0BThresholds().validate(19)


def test_missing_expected_source_is_a_hard_failure():
    rows = [row for row in _observations() if row.source_dataset != "scannetppv2"]
    report = build_t0_b_report(
        rows, requested_batches=len(rows),
        thresholds=T0BThresholds(minimum_batches=20, maximum_batches=50),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert not report["checks"]["source_registry_exact"]["passed"]
    assert report["status"] == "complete_failed"


def test_component_mask_loss_inconsistency_is_a_hard_failure():
    rows = _observations()
    bad = rows[0]
    rows[0] = T0BBatchObservation(
        **(vars(bad) | {
            "enabled_components": ("existence",),
            "masked_components": ("category", "center", "extent", "visibility"),
            "component_losses": {"existence": 1.0, "category": 1.0},
        })
    )
    report = build_t0_b_report(
        rows, requested_batches=30, thresholds=T0BThresholds(),
        checkpoint_resume_passed=True, runtime_status="gpu_complete",
        resolved_config_sha256="a" * 64,
        expected_components=("existence", "category"),
    )
    assert not report["checks"]["component_mask_consistency"]["passed"]


def test_checkpoint_resume_restores_tensor_optimizer_scheduler_counter_and_rng(tmp_path):
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    contract = ResumeContract("a1o", "1" * 64, "2" * 64, "3" * 64)
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        global_step=7, epoch=2, sampler_position=11, contract=contract,
    )
    expected = (
        nested_state_digest(model.state_dict()),
        nested_state_digest(optimizer.state_dict()),
        nested_state_digest(scheduler.state_dict()),
        nested_state_digest(capture_rng_state()),
    )
    with torch.no_grad():
        model.weight.add_(99)
    torch.rand(3)
    counters = load_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        expected_contract=contract,
    )
    actual = (
        nested_state_digest(model.state_dict()),
        nested_state_digest(optimizer.state_dict()),
        nested_state_digest(scheduler.state_dict()),
        nested_state_digest(capture_rng_state()),
    )
    assert actual == expected
    assert counters == {"global_step": 7, "epoch": 2, "sampler_position": 11}


def test_t0_a_model_state_digest_matches_record_semantics_not_generic_digest():
    state = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        "bias": torch.tensor([3.0], dtype=torch.float32),
    }
    records = []
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        records.append((
            key,
            list(value.shape),
            str(value.dtype),
            __import__("hashlib").sha256(raw).hexdigest(),
        ))
    expected = stable_sha256(records)
    assert t0_a_model_state_digest(state) == expected
    assert nested_state_digest(state) != expected


def test_t0_a_model_state_restore_valid_round_trip():
    checkpoint = {"weight": torch.tensor([1.0, 2.0])}
    loaded = {"weight": checkpoint["weight"].clone()}
    expected = t0_a_model_state_digest(checkpoint)
    evidence = validate_t0_a_model_state_restore(
        checkpoint_state=checkpoint,
        loaded_state=loaded,
        expected_digest=expected,
    )
    assert set(evidence.values()) == {expected}


def test_t0_a_model_state_restore_rejects_checkpoint_tamper():
    original = {"weight": torch.tensor([1.0, 2.0])}
    tampered = {"weight": torch.tensor([1.0, 9.0])}
    with pytest.raises(ValueError, match="checkpoint payload"):
        validate_t0_a_model_state_restore(
            checkpoint_state=tampered,
            loaded_state=tampered,
            expected_digest=t0_a_model_state_digest(original),
        )


def test_t0_a_model_state_restore_rejects_loaded_state_mismatch():
    checkpoint = {"weight": torch.tensor([1.0, 2.0])}
    loaded = {"weight": torch.tensor([1.0, 9.0])}
    with pytest.raises(ValueError, match="loaded T0-A"):
        validate_t0_a_model_state_restore(
            checkpoint_state=checkpoint,
            loaded_state=loaded,
            expected_digest=t0_a_model_state_digest(checkpoint),
        )


@pytest.mark.parametrize(
    "state",
    [{}, {1: torch.tensor([1.0])}, {"weight": "not-a-tensor"}],
)
def test_t0_a_model_state_digest_rejects_malformed_state(state):
    with pytest.raises(ValueError, match="T0-A model state"):
        t0_a_model_state_digest(state)
