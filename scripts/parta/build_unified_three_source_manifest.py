#!/usr/bin/env python3
"""Build the stable ADT/Hypersim/ScanNet++ Part A QA split index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from parta_data_contract import content_sha256, read_jsonl, stable_json  # noqa: E402
from parta.unified_data import (  # noqa: E402
    SplitDefaults,
    build_engineering_subset_artifact,
    build_exact_input_registry,
    build_unified_rows,
    summarize_unified_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adt-root", type=Path, required=True)
    parser.add_argument("--hypersim-root", type=Path, required=True)
    parser.add_argument("--scannetppv2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--engineering-subset-output", type=Path, required=True)
    parser.add_argument(
        "--engineering-scenes-per-source",
        type=int,
        required=True,
        help="Explicit D-62 engineering-subset size; no implicit default is permitted.",
    )
    return parser.parse_args()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    roots = {
        "adt": args.adt_root,
        "hypersim": args.hypersim_root,
        "scannetppv2": args.scannetppv2_root,
    }
    qa_by_source = {}
    for source, root in roots.items():
        path = root / "qa_manifest_exact_verified.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing exact canonical manifest for {source}: {path}")
        qa_by_source[source] = list(read_jsonl(path))
    defaults = SplitDefaults()
    rows = build_unified_rows(qa_by_source, defaults)
    report = summarize_unified_rows(rows, defaults)
    report["canonical_roots"] = {source: str(path.resolve()) for source, path in roots.items()}
    exact_inputs = build_exact_input_registry(roots)
    report["exact_canonical_inputs"] = exact_inputs
    report["exact_canonical_inputs_registry_sha256"] = content_sha256(exact_inputs)
    report["note"] = "D-62 frozen three-source train/val contract; no smoke split."
    manifest_payload = "".join(stable_json(row) + "\n" for row in rows)
    report["manifest_file_sha256"] = hashlib.sha256(
        manifest_payload.encode("utf-8")
    ).hexdigest()
    engineering_artifact = build_engineering_subset_artifact(
        rows,
        qa_by_source,
        scenes_per_source=args.engineering_scenes_per_source,
        exact_canonical_inputs=exact_inputs,
    )
    engineering_payload = json.dumps(
        engineering_artifact, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    atomic_write(args.output, manifest_payload)
    atomic_write(args.engineering_subset_output, engineering_payload)
    report["engineering_subset"] = {
        "path": str(args.engineering_subset_output.resolve()),
        "size_bytes": args.engineering_subset_output.stat().st_size,
        "sha256": hashlib.sha256(engineering_payload.encode("utf-8")).hexdigest(),
        "payload_sha256": engineering_artifact["artifact_payload_sha256"],
        "scenes_per_source": args.engineering_scenes_per_source,
        "promotable_to_formal_training": False,
    }
    atomic_write(args.report_output, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
