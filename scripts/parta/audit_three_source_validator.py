#!/usr/bin/env python3
"""Recompute the three-source canonical/manifest validation evidence fail closed."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from parta.provenance import atomic_json_dump, sha256_file  # noqa: E402
from parta.unified_data import (FROZEN_SOURCE_INVENTORY, FROZEN_SOURCE_REGISTRY,
                                SplitDefaults, file_sha256, load_unified_rows,
                                summarize_unified_rows, validate_exact_input_registry,
                                validate_unified_report)  # noqa: E402

RECOMPUTED_KEYS = ("source_registry", "manifest_rows_sha256", "total_scenes", "total_qa",
                   "frozen_source_inventory", "frozen_total_inventory",
                   "scene_intersections", "sources")

def validate_recomputed_summary(recomputed: dict, manifest_report: dict) -> None:
    if any(recomputed.get(key) != manifest_report.get(key) for key in RECOMPUTED_KEYS):
        raise ValueError("unified manifest recomputation differs from its report")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-validation", action="append", required=True,
                        metavar="SOURCE=PATH")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validations = {}
    for value in args.source_validation:
        source, raw_path = value.split("=", 1)
        if source in validations or source not in FROZEN_SOURCE_REGISTRY:
            raise ValueError("source validation registry is duplicate or unknown")
        path = Path(raw_path).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        expected = dict(FROZEN_SOURCE_INVENTORY[source])
        observed = {"qa": report.get("qa"), "scenes": report.get("scenes")}
        if (report.get("schema_version") != "parta_validation_report_v2"
                or report.get("status") != "complete_passed" or observed != expected
                or report.get("source_counts", {}).get(source, {}).get("qa") != expected["qa"]
                or report.get("source_counts", {}).get(source, {}).get("scenes") != expected["scenes"]):
            raise ValueError(f"invalid canonical validation report: {source}")
        validations[source] = {"path": str(path), "sha256": sha256_file(path),
                               "schema_version": report["schema_version"], "status": report["status"]}
    if set(validations) != set(FROZEN_SOURCE_REGISTRY):
        raise ValueError("canonical validation reports must cover exactly three sources")
    manifest_report = json.loads(args.manifest_report.read_text(encoding="utf-8"))
    validate_unified_report(manifest_report, expected_inventory=FROZEN_SOURCE_INVENTORY)
    registry = manifest_report["exact_canonical_inputs"]
    roots = {source: Path(item["root"]).resolve() for source, item in registry.items()}
    validate_exact_input_registry(roots, registry)
    for source, root in roots.items():
        if Path(validations[source]["path"]) != root / "validation_report.json":
            raise ValueError(f"{source} validation report is not owned by its canonical root")
    rows = load_unified_rows(args.manifest)
    recomputed = summarize_unified_rows(rows, SplitDefaults(),
                                        expected_inventory=FROZEN_SOURCE_INVENTORY)
    validate_recomputed_summary(recomputed, manifest_report)
    if file_sha256(args.manifest) != manifest_report.get("manifest_file_sha256"):
        raise ValueError("unified manifest file hash mismatch")
    producer = Path(__file__).resolve()
    atomic_json_dump({"schema_version": "parta_three_source_validator_audit_v2",
                      "status": "complete_passed",
                      "producer": {"path": str(producer), "sha256": sha256_file(producer),
                                   "git_revision": subprocess.check_output(
                                       ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
                                   ).strip()},
                      "manifest": {"path": str(args.manifest.resolve()),
                                   "sha256": file_sha256(args.manifest)},
                      "manifest_report": {"path": str(args.manifest_report.resolve()),
                                          "sha256": sha256_file(args.manifest_report)},
                      "exact_registry": registry,
                      "exact_registry_sha256": manifest_report["exact_canonical_inputs_registry_sha256"],
                      "source_validations": validations,
                      "recomputed_summary": recomputed}, args.output)

if __name__ == "__main__":
    main()
