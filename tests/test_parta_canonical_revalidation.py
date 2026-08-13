import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.canonical_revalidation import (
    CANONICAL_DATA_FILENAMES,
    publish_validation_report,
    recompute_validation_report,
)
from parta.unified_data import (
    FROZEN_SOURCE_INVENTORY,
    REQUIRED_CANONICAL_FILENAMES,
    SUPPORT_CERTIFICATE_FILENAMES,
    build_exact_input_registry,
    validate_exact_input_registry,
)
from parta_data_contract import ContractError, ValidationReport
from adt_gt_supported_clip import build_support_certificate as build_adt_certificate
from scannetppv2_support import build_support_certificate as build_scannetpp_certificate


def _write_root(root: Path, source: str, *, certificate: bool = False) -> None:
    root.mkdir()
    rows = {
        "scene_states.jsonl": [{"source_dataset": source, "scene_id": "s"}],
        "frame_states.jsonl": [
            {"source_dataset": source, "scene_id": "s", "frame_key": "f"}
        ],
        "qa_manifest_exact_verified.jsonl": [
            {"source_dataset": source, "scene_id": "s", "qa_id": f"{source}:q"}
        ],
    }
    for filename, values in rows.items():
        (root / filename).write_text(
            "".join(json.dumps(value) + "\n" for value in values)
        )
    if certificate:
        (root / SUPPORT_CERTIFICATE_FILENAMES[source]).write_text(
            json.dumps(
                {
                    "scene_id": "s",
                    "certificate_sha256": "a" * 64,
                }
            )
            + "\n"
        )


def _scannetpp_certificate(scene_id: str) -> dict:
    digest = "d" * 64
    return build_scannetpp_certificate(
        scene_id=scene_id,
        vsi_media=f"scannetppv2/{scene_id}.mp4",
        sampling_binding_sha256=digest,
        video_total_frames=16,
        video_fps=30.0,
        video_width=640,
        video_height=480,
        source_assets={
            name: {"size_bytes": 1, "sha256": digest}
            for name in ("mesh", "segments", "annotation", "pose", "exif")
        },
        video_metadata_sha256=digest,
        instance_assignment_source_sha256=digest,
        label_normalization_source_sha256=digest,
        rasterizer_source_sha256=digest,
        rasterizer_library_sha256=digest,
        frames=[
            {
                "frame_index": index,
                "frame_key": str(index),
                "pose_sha256": digest,
                "intrinsic_sha256": digest,
                "instance_mask_sha256": digest,
                "visible_instance_pixel_counts": {"1": 1},
            }
            for index in range(16)
        ],
    )


def _replace_rows(root: Path, filename: str, rows: list[dict]) -> None:
    (root / filename).write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def _set_scannetpp_binding(
    root: Path,
    certificate: dict,
    *,
    sampling_binding: str | None = None,
    indices: list[int] | None = None,
) -> None:
    certified_frames = certificate["frames"]
    frame_indices = indices or [int(row["frame_index"]) for row in certified_frames]
    frame_keys = [str(row["frame_key"]) for row in certified_frames]
    _replace_rows(
        root,
        "frame_states.jsonl",
        [
            {
                "source_dataset": "scannetppv2",
                "scene_id": certificate["scene_id"],
                "frame_key": key,
                "frame_index": index,
            }
            for key, index in zip(frame_keys, frame_indices)
        ],
    )
    _replace_rows(
        root,
        "qa_manifest_exact_verified.jsonl",
        [
            {
                "source_dataset": "scannetppv2",
                "scene_id": certificate["scene_id"],
                "qa_id": "scannetppv2:q",
                "vsi_media": certificate["vsi_media"],
                "source_sampling_binding_sha256": (
                    sampling_binding
                    if sampling_binding is not None
                    else certificate["sampling_binding_sha256"]
                ),
                "actual_frame_indices": frame_indices,
                "actual_frame_keys": frame_keys,
            }
        ],
    )


def _stub_full_validation(monkeypatch, source: str) -> list[dict]:
    calls = []

    def fake_validate(scenes, frames, qa_rows, **kwargs):
        calls.append(
            {
                "scenes": scenes,
                "frames": frames,
                "qa": qa_rows,
                "kwargs": kwargs,
            }
        )
        expected = FROZEN_SOURCE_INVENTORY[source]
        return ValidationReport(
            scenes=expected["scenes"],
            qa=expected["qa"],
            source_counts={
                source: {
                    "scenes": expected["scenes"],
                    "frames": 1,
                    "qa": expected["qa"],
                    "visible_observations": 3,
                    "visibility_evidence_observations": 4,
                }
            },
            scene_object_counts={},
            qa_coverage_counts={},
        )

    monkeypatch.setattr("parta.canonical_revalidation.validate_records", fake_validate)
    monkeypatch.setattr(
        "parta.canonical_revalidation._producer_record",
        lambda *_: {"path": "/producer", "sha256": "b" * 64, "git_revision": "c" * 40},
    )
    return calls


def test_recompute_ignores_legacy_status_and_fully_validates_inputs(tmp_path, monkeypatch):
    root = tmp_path / "hypersim"
    _write_root(root, "hypersim")
    (root / "validation_report.json").write_text(
        json.dumps({"schema_version": "parta_validation_report_v1"})
    )
    calls = _stub_full_validation(monkeypatch, "hypersim")

    payload = recompute_validation_report(
        source="hypersim",
        root=root,
        project_root=tmp_path,
        producer=__file__,
    )

    assert payload["schema_version"] == "parta_validation_report_v2"
    assert payload["status"] == "complete_passed"
    assert payload["validation_mode"] == "full_canonical_recomputation_v1"
    assert set(payload["canonical_inputs"]) == set(CANONICAL_DATA_FILENAMES)
    assert calls[0]["kwargs"] == {
        "require_fixtures": False,
        "expected_sources": ("hypersim",),
    }


def test_production_recompute_has_no_inventory_override(tmp_path):
    with pytest.raises(TypeError, match="expected_inventory"):
        recompute_validation_report(
            source="hypersim",
            root=tmp_path,
            project_root=tmp_path,
            producer=__file__,
            expected_inventory={"qa": 1, "scenes": 1},
        )


def test_recompute_rejects_invalid_data_and_source_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "hypersim"
    _write_root(root, "hypersim")

    def reject(*_args, **_kwargs):
        raise ContractError("mixed/swapped source")

    monkeypatch.setattr("parta.canonical_revalidation.validate_records", reject)
    with pytest.raises(ContractError, match="mixed/swapped"):
        recompute_validation_report(
            source="hypersim", root=root, project_root=tmp_path, producer=__file__
        )


@pytest.mark.parametrize(
    "old_report",
    [
        {"schema_version": "parta_validation_report_v1"},
        {"schema_version": "parta_validation_report_v2"},
    ],
)
def test_publish_requires_explicit_replace_and_preserves_previous_report(
    tmp_path, old_report
):
    output = tmp_path / "validation_report.json"
    old_bytes = (json.dumps(old_report) + "\n").encode()
    output.write_bytes(old_bytes)
    with pytest.raises(FileExistsError, match="replace-existing"):
        publish_validation_report({"status": "complete_passed"}, output)

    backup = publish_validation_report(
        {"schema_version": "parta_validation_report_v2", "status": "complete_passed"},
        output,
        replace_existing=True,
    )
    assert backup is not None and backup.read_bytes() == old_bytes
    upgraded = json.loads(output.read_text())
    assert upgraded["status"] == "complete_passed"
    assert upgraded["previous_validation_report"]["sha256"]
    assert not list(tmp_path.glob("*.tmp"))


def test_required_certificate_is_fail_closed_and_validated(tmp_path, monkeypatch):
    root = tmp_path / "adt"
    _write_root(root, "adt")
    _stub_full_validation(monkeypatch, "adt")
    with pytest.raises(ContractError, match="missing required support"):
        recompute_validation_report(
            source="adt", root=root, project_root=tmp_path, producer=__file__
        )

    certificate = build_adt_certificate(
        scene_id="s",
        vsi_media="adt/s.mp4",
        frame_timestamps=list(range(16)),
        fps=30.0,
        support_mask=[True] * 16,
        max_trajectory_error_ns=5_000_000,
        max_calibration_error_ns=50_000_000,
    )
    _replace_rows(
        root,
        SUPPORT_CERTIFICATE_FILENAMES["adt"],
        [certificate],
    )
    _replace_rows(
        root,
        "qa_manifest_exact_verified.jsonl",
        [
            {
                "source_dataset": "adt",
                "scene_id": "s",
                "qa_id": "adt:q",
                "vsi_media": "adt/s.mp4",
                "clip_provenance": {"support_certificate": certificate},
            }
        ],
    )
    payload = recompute_validation_report(
        source="adt", root=root, project_root=tmp_path, producer=__file__
    )
    assert payload["canonical_inputs"][SUPPORT_CERTIFICATE_FILENAMES["adt"]][
        "certificate_count"
    ] == 1
    assert payload["trusted_support_certificate_registry"] == {
        "path": SUPPORT_CERTIFICATE_FILENAMES["adt"],
        "size_bytes": (root / SUPPORT_CERTIFICATE_FILENAMES["adt"]).stat().st_size,
        "sha256": payload["canonical_inputs"][SUPPORT_CERTIFICATE_FILENAMES["adt"]][
            "sha256"
        ],
        "certificate_count": 1,
        "anchored_by": "full_canonical_recomputation_v1",
    }


def test_adt_rejects_embedded_certificate_different_from_external(tmp_path, monkeypatch):
    root = tmp_path / "adt"
    _write_root(root, "adt")
    certificate = build_adt_certificate(
        scene_id="s",
        vsi_media="adt/s.mp4",
        frame_timestamps=list(range(16)),
        fps=30.0,
        support_mask=[True] * 16,
        max_trajectory_error_ns=5_000_000,
        max_calibration_error_ns=50_000_000,
    )
    changed = dict(certificate)
    changed["certificate_sha256"] = "0" * 64
    _replace_rows(root, SUPPORT_CERTIFICATE_FILENAMES["adt"], [certificate])
    _replace_rows(
        root,
        "qa_manifest_exact_verified.jsonl",
        [{"source_dataset": "adt", "scene_id": "s", "qa_id": "adt:q",
          "vsi_media": "adt/s.mp4", "clip_provenance": {"support_certificate": changed}}],
    )
    _stub_full_validation(monkeypatch, "adt")
    with pytest.raises(ContractError, match="embedded/external"):
        recompute_validation_report(
            source="adt", root=root, project_root=tmp_path, producer=__file__
        )


def test_scannetpp_rejects_sampling_binding_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "scannetppv2"
    _write_root(root, "scannetppv2")
    certificate = _scannetpp_certificate("s")
    _replace_rows(root, SUPPORT_CERTIFICATE_FILENAMES["scannetppv2"], [certificate])
    _set_scannetpp_binding(root, certificate, sampling_binding="0" * 64)
    _stub_full_validation(monkeypatch, "scannetppv2")
    with pytest.raises(ContractError, match="sampling binding mismatch"):
        recompute_validation_report(
            source="scannetppv2", root=root, project_root=tmp_path, producer=__file__
        )


def test_scannetpp_rejects_certified_frame_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "scannetppv2"
    _write_root(root, "scannetppv2")
    certificate = _scannetpp_certificate("s")
    _replace_rows(root, SUPPORT_CERTIFICATE_FILENAMES["scannetppv2"], [certificate])
    _set_scannetpp_binding(root, certificate, indices=list(range(1, 17)))
    _stub_full_validation(monkeypatch, "scannetppv2")
    with pytest.raises(ContractError, match="certified frames differ"):
        recompute_validation_report(
            source="scannetppv2", root=root, project_root=tmp_path, producer=__file__
        )


def test_scannetpp_rejects_canonical_frame_table_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "scannetppv2"
    _write_root(root, "scannetppv2")
    certificate = _scannetpp_certificate("s")
    _replace_rows(root, SUPPORT_CERTIFICATE_FILENAMES["scannetppv2"], [certificate])
    _set_scannetpp_binding(root, certificate)
    frames = [
        {
            "source_dataset": "scannetppv2",
            "scene_id": "s",
            "frame_key": str(index),
            "frame_index": index,
        }
        for index in range(16)
    ]
    frames[-1]["frame_index"] = 99
    _replace_rows(root, "frame_states.jsonl", frames)
    _stub_full_validation(monkeypatch, "scannetppv2")
    with pytest.raises(ContractError, match="canonical frame table"):
        recompute_validation_report(
            source="scannetppv2", root=root, project_root=tmp_path, producer=__file__
        )


def test_scannetpp_real_certificate_path_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / "scannetppv2"
    _write_root(root, "scannetppv2")
    certificate = _scannetpp_certificate("s")
    _replace_rows(root, SUPPORT_CERTIFICATE_FILENAMES["scannetppv2"], [certificate])
    _set_scannetpp_binding(root, certificate)
    _stub_full_validation(monkeypatch, "scannetppv2")
    payload = recompute_validation_report(
        source="scannetppv2", root=root, project_root=tmp_path, producer=__file__
    )
    assert payload["trusted_support_certificate_registry"]["certificate_count"] == 1


def test_atomic_publish_failure_keeps_previous_report_and_cleans_temp(
    tmp_path, monkeypatch
):
    output = tmp_path / "validation_report.json"
    output.write_text("old\n")
    real_replace = __import__("os").replace
    replace_calls = 0

    def fail_final_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected final publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr("parta.canonical_revalidation.os.replace", fail_final_replace)
    with pytest.raises(OSError, match="injected"):
        publish_validation_report(
            {"status": "complete_passed"}, output, replace_existing=True
        )
    assert output.read_text() == "old\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_validation_report_replacement_does_not_invalidate_exact_registry(tmp_path):
    roots = {}
    for source in ("adt", "hypersim", "scannetppv2"):
        root = tmp_path / source
        root.mkdir()
        for filename in REQUIRED_CANONICAL_FILENAMES:
            (root / filename).write_text(f"{source}:{filename}\n")
        if source != "hypersim":
            (root / SUPPORT_CERTIFICATE_FILENAMES[source]).write_text(
                f"{source}:certificate\n"
            )
        (root / "validation_report.json").write_text("old\n")
        roots[source] = root

    assert "validation_report.json" not in REQUIRED_CANONICAL_FILENAMES
    registry = build_exact_input_registry(roots)
    for root in roots.values():
        (root / "validation_report.json").write_text("new\n")
    validate_exact_input_registry(roots, registry)
