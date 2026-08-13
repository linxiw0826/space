import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.canonical_data import PartASample
from parta.unified_data import (
    FROZEN_SOURCE_REGISTRY,
    UNIFIED_SCHEMA_VERSION,
    PartACPUStateCollator,
    PartAUnifiedDataset,
    SplitDefaults,
    build_engineering_subset_artifact,
    build_exact_input_registry,
    build_unified_rows,
    file_sha256,
    iter_source_balanced_indices,
    load_engineering_subset_artifact,
    load_unified_rows,
    stable_scene_split,
    summarize_unified_rows,
    validate_engineering_subset_artifact,
)
from parta_data_contract import ContractError, content_sha256, stable_json


def _qa(source, scene, index):
    return {
        "source_dataset": source,
        "scene_id": scene,
        "qa_id": f"{source}:{scene}:q{index}",
        "vsi_media": f"{source}/{scene}.mp4",
        "media_kind": "video",
        "actual_frame_indices": [0, 1],
        "actual_frame_keys": ["0", "1"],
    }


def _sample(source, *, count=1, frames=1, empty=False):
    nodes = [
        {
            "object_id": f"{source}:s:o{index:03d}",
            "category": "chair",
            "center_world_m": [float(index), 0.0, 0.0],
            "extent_m": [1.0, 1.0, 1.0],
            "field_mask": {
                "category": True,
                "center": True,
                "extent": True,
                "orientation": False,
                "motion": False,
            },
        }
        for index in range(count)
    ]
    frame_rows = []
    for frame_index in range(frames):
        visible_nodes = [] if empty else [
            {
                "object_id": node["object_id"],
                "evidence_present": True,
                "visible": True,
                "field_mask": {"visibility": True},
            }
            for node in nodes
        ]
        frame_rows.append(
            {
                "source_dataset": source,
                "scene_id": "s",
                "frame_key": str(frame_index),
                "visible_nodes": visible_nodes,
            }
        )
    visible = [] if empty else [node["object_id"] for node in nodes]
    qa = {
        "qa_id": f"{source}:q",
        "source_dataset": source,
        "scene_id": "s",
        "vsi_media": f"{source}/x.jpg" if source == "hypersim" else f"{source}/x.mp4",
        "media_kind": "image" if source == "hypersim" else "video",
        "actual_frame_indices": list(range(frames)),
        "actual_frame_keys": [str(value) for value in range(frames)],
        "actual_visible_object_ids": visible,
    }
    return PartASample(
        {"source_dataset": source, "scene_id": "s", "nodes": nodes},
        tuple(frame_rows),
        qa,
    )


def _write_canonical_registry_files(root, source, qa_payload):
    (root / "qa_manifest_exact_verified.jsonl").write_text(qa_payload, encoding="utf-8")
    (root / "scene_states.jsonl").write_text(
        stable_json({"source_dataset": source, "fixture": "scene"}) + "\n",
        encoding="utf-8",
    )
    (root / "frame_states.jsonl").write_text(
        stable_json({"source_dataset": source, "fixture": "frame"}) + "\n",
        encoding="utf-8",
    )
    support_names = {
        "adt": "adt_support_certificates.jsonl",
        "scannetppv2": "scannetppv2_support_certificates.jsonl",
    }
    if source in support_names:
        (root / support_names[source]).write_text(
            stable_json({"source_dataset": source, "fixture": "support"}) + "\n",
            encoding="utf-8",
        )


def test_v2_scene_split_is_seed42_deterministic_and_train_val_disjoint():
    defaults = SplitDefaults()
    three = {
        source: [_qa(source, f"scene-{scene}", question) for scene in range(20) for question in range(2)]
        for source in ("adt", "hypersim", "scannetppv2")
    }
    rows_a = build_unified_rows(three, defaults, expected_inventory=None)
    rows_b = build_unified_rows(three, defaults, expected_inventory=None)
    assert rows_a == rows_b
    report = summarize_unified_rows(rows_a, defaults, expected_inventory=None)
    assert report["schema_version"] == "parta_unified_manifest_v2"
    assert report["seed"] == 42
    assert report["val_fraction"] == 0.10
    assert report["source_registry"] == list(FROZEN_SOURCE_REGISTRY)
    assert report["scene_intersections"] == {"train__val": 0}
    assert {row["split"] for row in rows_a} <= {"train", "val"}
    assert "smoke_fraction" not in report
    assert stable_scene_split("adt", "scene-1", defaults) == stable_scene_split(
        "adt", "scene-1", defaults
    )


@pytest.mark.parametrize(
    "defaults",
    [SplitDefaults(seed=41), SplitDefaults(val_fraction=0.11)],
)
def test_split_contract_rejects_seed_or_fraction_override(defaults):
    with pytest.raises(ContractError, match="D-62 freezes"):
        build_unified_rows(
            {source: [_qa(source, "s", 0)] for source in FROZEN_SOURCE_REGISTRY},
            defaults,
            expected_inventory=None,
        )


@pytest.mark.parametrize(
    "sources",
    [
        ("adt", "hypersim"),
        ("adt", "hypersim", "scannetppv2", "scannet"),
    ],
)
def test_registry_rejects_missing_source_or_future_scannet(sources):
    qa_by_source = {source: [_qa(source, "s", 0)] for source in sources}
    with pytest.raises(ContractError, match="must use exactly"):
        build_unified_rows(qa_by_source, SplitDefaults(), expected_inventory=None)


def test_production_inventory_is_exact_and_fixture_relaxation_is_explicit():
    qa_by_source = {
        source: [_qa(source, f"scene-{index}", 0) for index in range(2)]
        for source in FROZEN_SOURCE_REGISTRY
    }
    with pytest.raises(ContractError, match="source inventory differs from frozen contract"):
        build_unified_rows(qa_by_source, SplitDefaults())
    fixture_inventory = {
        source: {"qa": 2, "scenes": 2} for source in FROZEN_SOURCE_REGISTRY
    }
    rows = build_unified_rows(
        qa_by_source, SplitDefaults(), expected_inventory=fixture_inventory
    )
    report = summarize_unified_rows(
        rows, SplitDefaults(), expected_inventory=fixture_inventory
    )
    assert report["contract_status"] == "fixture_only_explicit_inventory"
    assert report["frozen_source_inventory"] == fixture_inventory
    assert report["frozen_total_inventory"] == {"qa": 6, "scenes": 6}


def test_manifest_loader_rejects_smoke_duplicate_and_scene_leakage(tmp_path):
    qa_by_source = {
        source: [_qa(source, f"scene-{index}", 0) for index in range(30)]
        for source in FROZEN_SOURCE_REGISTRY
    }
    rows = build_unified_rows(qa_by_source, SplitDefaults(), expected_inventory=None)
    path = tmp_path / "manifest.jsonl"

    smoke_rows = [dict(row) for row in rows]
    smoke_rows[0]["split"] = "smoke"
    path.write_text("".join(stable_json(row) + "\n" for row in smoke_rows))
    with pytest.raises(ContractError, match="unknown split: smoke"):
        load_unified_rows(path)

    duplicate_rows = rows + [dict(rows[0])]
    path.write_text("".join(stable_json(row) + "\n" for row in duplicate_rows))
    with pytest.raises(ContractError, match="duplicate qa_id"):
        load_unified_rows(path)

    leaked_rows = [dict(row) for row in rows]
    duplicate = dict(rows[0])
    duplicate["qa_id"] += ":other"
    duplicate["split"] = "val" if duplicate["split"] == "train" else "train"
    duplicate["canonical_qa_content_sha256"] = "other"
    duplicate["source_qa_count"] += 1
    leaked_rows.append(duplicate)
    path.write_text("".join(stable_json(row) + "\n" for row in leaked_rows))
    with pytest.raises(ContractError, match="scene leakage"):
        load_unified_rows(path)


def test_source_balanced_indices_are_deterministic_and_balance_sources():
    rows = [
        {"source_sampling_key": "adt"},
        {"source_sampling_key": "adt"},
        {"source_sampling_key": "hypersim"},
        {"source_sampling_key": "scannetppv2"},
    ]
    first = list(iter_source_balanced_indices(rows, seed=42))
    second = list(iter_source_balanced_indices(rows, seed=42))
    assert first == second
    assert len(first) == 6
    assert [rows[index]["source_sampling_key"] for index in first[:3]] == [
        "adt",
        "hypersim",
        "scannetppv2",
    ]


def _persist_unified_fixture(tmp_path, rows):
    roots = {}
    for source in ("adt", "hypersim", "scannetppv2"):
        root = tmp_path / source
        root.mkdir()
        _write_canonical_registry_files(
            root,
            source,
            stable_json({"source_dataset": source, "fixture": True}) + "\n",
        )
        roots[source] = root
    present_sources = {row["source_dataset"] for row in rows}
    persisted_rows = list(rows)
    for source in FROZEN_SOURCE_REGISTRY:
        if source not in present_sources:
            persisted_rows.append(
                _index_row(
                    source=source,
                    scene=f"{source}-registry-fixture",
                    qa_id=f"{source}:registry-fixture",
                )
                | {"split": "val"}
            )
    manifest = tmp_path / "unified.jsonl"
    payload = "".join(stable_json(row) + "\n" for row in persisted_rows)
    manifest.write_text(payload, encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": UNIFIED_SCHEMA_VERSION,
                "contract_status": "fixture_only_explicit_inventory",
                "source_registry": list(FROZEN_SOURCE_REGISTRY),
                "source_registry_sha256": content_sha256(list(FROZEN_SOURCE_REGISTRY)),
                "seed": 42,
                "val_fraction": 0.10,
                "manifest_rows_sha256": content_sha256(persisted_rows),
                "manifest_file_sha256": file_sha256(manifest),
                "exact_canonical_inputs": build_exact_input_registry(roots),
                "exact_canonical_inputs_registry_sha256": content_sha256(
                    build_exact_input_registry(roots)
                ),
            }
        ),
        encoding="utf-8",
    )
    return roots, manifest, report


def _index_row(source="adt", scene="s", qa_id="adt:q"):
    return {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "source_dataset": source,
        "scene_id": scene,
        "qa_id": qa_id,
        "split": "train",
        "source_sampling_key": source,
        "source_qa_count": 1,
        "source_split_qa_count": 1,
        "source_balanced_weight": 1.0,
        "canonical_qa_content_sha256": "fixture-content",
        "exact_input_binding_sha256": "fixture-input",
    }


@pytest.mark.parametrize(
    "rows,match",
    [
        ([_index_row(scene="wrong")], "identity mismatch"),
        ([_index_row(source="hypersim")], "identity mismatch"),
        ([_index_row(), _index_row()], "duplicate qa_id"),
        ([_index_row() | {"source_sampling_key": "hypersim"}], "source_sampling_key"),
    ],
)
def test_unified_dataset_rejects_wrong_or_duplicate_persisted_identity(
    tmp_path, monkeypatch, rows, match
):
    roots, manifest, report = _persist_unified_fixture(tmp_path, rows)
    sample = _sample("adt", frames=16)
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(sample,)),
    )
    with pytest.raises(ContractError, match=match):
        PartAUnifiedDataset(
            roots, manifest, split="train", report_path=report, expected_inventory=None
        )


def test_unified_dataset_rejects_ambiguous_canonical_qa_id(tmp_path, monkeypatch):
    roots, manifest, report = _persist_unified_fixture(tmp_path, [_index_row()])
    sample = _sample("adt", frames=16)
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(sample, sample)),
    )
    with pytest.raises(ContractError, match="ambiguous duplicate canonical qa_id"):
        PartAUnifiedDataset(
            roots, manifest, split="train", report_path=report, expected_inventory=None
        )


def test_unified_dataset_rejects_changed_exact_canonical_input(tmp_path, monkeypatch):
    roots, manifest, report = _persist_unified_fixture(tmp_path, [_index_row()])
    with (roots["adt"] / "qa_manifest_exact_verified.jsonl").open("a") as handle:
        handle.write("changed\n")
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(_sample("adt", frames=16),)),
    )
    with pytest.raises(ContractError, match="exact canonical input changed"):
        PartAUnifiedDataset(
            roots, manifest, split="train", report_path=report, expected_inventory=None
        )


@pytest.mark.parametrize(
    "source,filename",
    [
        ("adt", "scene_states.jsonl"),
        ("hypersim", "frame_states.jsonl"),
        ("scannetppv2", "scannetppv2_support_certificates.jsonl"),
    ],
)
def test_unified_dataset_rejects_state_frame_or_support_drift(
    tmp_path, monkeypatch, source, filename
):
    roots, manifest, report = _persist_unified_fixture(tmp_path, [_index_row()])
    with (roots[source] / filename).open("a") as handle:
        handle.write("changed\n")
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(_sample("adt", frames=16),)),
    )
    with pytest.raises(ContractError, match="exact canonical input changed"):
        PartAUnifiedDataset(
            roots, manifest, split="train", report_path=report, expected_inventory=None
        )


def test_optional_hypersim_support_certificate_presence_is_frozen(tmp_path, monkeypatch):
    roots, manifest, report = _persist_unified_fixture(tmp_path, [_index_row()])
    (roots["hypersim"] / "hypersim_support_certificates.jsonl").write_text("added\n")
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(_sample("adt", frames=16),)),
    )
    with pytest.raises(ContractError, match="exact canonical input changed"):
        PartAUnifiedDataset(
            roots, manifest, split="train", report_path=report, expected_inventory=None
        )


def test_complete_input_registry_requires_state_frame_and_source_certificates(tmp_path):
    roots, _, _ = _persist_unified_fixture(tmp_path, [_index_row()])
    registry = build_exact_input_registry(roots)
    for source in FROZEN_SOURCE_REGISTRY:
        files = registry[source]["files"]
        required = {
            "qa_manifest_exact_verified.jsonl",
            "scene_states.jsonl",
            "frame_states.jsonl",
        }
        assert required <= set(files)
        assert registry[source]["files_sha256"] == content_sha256(files)
    assert registry["adt"]["files"]["adt_support_certificates.jsonl"]["required"] is True
    assert registry["scannetppv2"]["files"][
        "scannetppv2_support_certificates.jsonl"
    ]["required"] is True
    assert registry["hypersim"]["files"][
        "hypersim_support_certificates.jsonl"
    ]["present"] is False
    (roots["adt"] / "adt_support_certificates.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="missing required support certificate"):
        build_exact_input_registry(roots)


def test_unified_dataset_accepts_exact_identity_and_registry(tmp_path, monkeypatch):
    roots, manifest, report = _persist_unified_fixture(tmp_path, [_index_row()])
    sample = _sample("adt", frames=16)
    monkeypatch.setattr(
        "parta.unified_data.PartACanonicalDataset",
        lambda *args, **kwargs: SimpleNamespace(samples=(sample,)),
    )
    dataset = PartAUnifiedDataset(
        roots, manifest, split="train", report_path=report, expected_inventory=None
    )
    assert len(dataset) == 1
    assert dataset[0] is sample


def test_production_manifest_cli_rejects_fixture_sized_inventory(tmp_path):
    roots = {}
    for source in ("adt", "hypersim", "scannetppv2"):
        root = tmp_path / source
        root.mkdir()
        rows = [_qa(source, f"{source}-scene-{index}", 0) for index in range(30)]
        _write_canonical_registry_files(
            root,
            source,
            "".join(stable_json(row) + "\n" for row in rows),
        )
        roots[source] = root
    manifest = tmp_path / "unified.jsonl"
    report = tmp_path / "report.json"
    engineering = tmp_path / "engineering.json"
    script = Path(__file__).resolve().parents[1] / "scripts/parta/build_unified_three_source_manifest.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--adt-root",
            str(roots["adt"]),
            "--hypersim-root",
            str(roots["hypersim"]),
            "--scannetppv2-root",
            str(roots["scannetppv2"]),
            "--output",
            str(manifest),
            "--report-output",
            str(report),
            "--engineering-subset-output",
            str(engineering),
            "--engineering-scenes-per-source",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "source inventory differs from frozen contract" in completed.stderr
    assert not manifest.exists()
    assert not report.exists()
    assert not engineering.exists()


def test_engineering_subset_is_deterministic_stratified_train_only_and_not_reweighted(tmp_path):
    qa_by_source = {
        source: [
            _qa(source, f"scene-{scene}", question)
            for scene in range(40)
            for question in range(2)
        ]
        for source in FROZEN_SOURCE_REGISTRY
    }
    rows = build_unified_rows(qa_by_source, SplitDefaults(), expected_inventory=None)
    roots = {}
    for source in FROZEN_SOURCE_REGISTRY:
        root = tmp_path / source
        root.mkdir()
        _write_canonical_registry_files(
            root,
            source,
            "".join(stable_json(row) + "\n" for row in qa_by_source[source]),
        )
        roots[source] = root
    registry = build_exact_input_registry(roots)
    before = [dict(row) for row in rows]
    first = build_engineering_subset_artifact(
        rows, qa_by_source, scenes_per_source=2, exact_canonical_inputs=registry
    )
    second = build_engineering_subset_artifact(
        rows, qa_by_source, scenes_per_source=2, exact_canonical_inputs=registry
    )
    assert first == second
    assert rows == before
    assert first["selected_scene_counts"] == {source: 2 for source in FROZEN_SOURCE_REGISTRY}
    assert set(first["selected_qa_counts"]) == set(FROZEN_SOURCE_REGISTRY)
    train_identities = {
        (row["source_dataset"], row["scene_id"], row["qa_id"])
        for row in rows
        if row["split"] == "train"
    }
    selected = {
        (row["source_dataset"], row["scene_id"], row["qa_id"])
        for row in first["selected_qa"]
    }
    assert selected <= train_identities
    assert first["formal_train_reuse"] == {
        "subset_rows_remain_in_train_manifest": True,
        "source_balanced_weights_unchanged": True,
        "extra_sampling_weight": False,
    }
    assert first["transaction_promotion"]["promotable_to_formal_training"] is False
    assert first["selection"]["question_loss_performance_fields_read_for_selection"] is False


def test_engineering_subset_requires_explicit_size_and_rejects_tampering(tmp_path):
    qa_by_source = {
        source: [_qa(source, f"scene-{scene}", 0) for scene in range(40)]
        for source in FROZEN_SOURCE_REGISTRY
    }
    rows = build_unified_rows(qa_by_source, SplitDefaults(), expected_inventory=None)
    roots = {}
    for source in FROZEN_SOURCE_REGISTRY:
        root = tmp_path / source
        root.mkdir()
        _write_canonical_registry_files(
            root,
            source,
            "".join(stable_json(row) + "\n" for row in qa_by_source[source]),
        )
        roots[source] = root
    registry = build_exact_input_registry(roots)
    with pytest.raises(ContractError, match="explicit positive integer"):
        build_engineering_subset_artifact(
            rows, qa_by_source, scenes_per_source=0, exact_canonical_inputs=registry
        )
    artifact = build_engineering_subset_artifact(
        rows, qa_by_source, scenes_per_source=1, exact_canonical_inputs=registry
    )
    validate_engineering_subset_artifact(
        artifact, rows, qa_by_source, exact_canonical_inputs=registry
    )
    path = tmp_path / "engineering.json"
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    expected_hash = file_sha256(path)
    loaded = load_engineering_subset_artifact(
        path,
        rows,
        qa_by_source,
        exact_canonical_inputs=registry,
        expected_file_sha256=expected_hash,
    )
    assert loaded == artifact
    tampered = dict(artifact)
    tampered["selected_scene_ids"] = dict(artifact["selected_scene_ids"])
    tampered["selected_scene_ids"]["adt"] = ["not-the-frozen-scene"]
    with pytest.raises(ContractError, match="artifact or bound inputs changed"):
        validate_engineering_subset_artifact(
            tampered, rows, qa_by_source, exact_canonical_inputs=registry
        )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ContractError, match="file hash mismatch"):
        load_engineering_subset_artifact(
            path,
            rows,
            qa_by_source,
            exact_canonical_inputs=registry,
            expected_file_sha256=expected_hash,
        )


def test_three_source_cpu_collator_enforces_k384_and_empty_gt():
    samples = [
        _sample("adt", count=385, frames=16),
        _sample("hypersim", count=2, frames=1),
        _sample("scannetppv2", count=0, frames=16, empty=True),
    ]
    batch = PartACPUStateCollator()(samples)
    assert batch.source_sampling_keys == ("adt", "hypersim", "scannetppv2")
    assert batch.targets[0].num_objects == 384
    assert len(batch.selection_audits[0].truncated_object_ids) == 1
    assert batch.targets[1].num_objects == 2
    assert batch.targets[2].num_objects == 0
    assert batch.targets[2].visibility.shape == (0, 16)
