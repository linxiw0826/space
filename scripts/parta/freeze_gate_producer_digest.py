#!/usr/bin/env python3
"""Freeze the reviewed gate-producer digest on the execution server.

This command is intentionally not run on the code server.  It converts the
post-transfer file content into the trust registry consumed by PhaseCommand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--review-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer = args.producer.resolve()
    review = json.loads(args.review_record.read_text(encoding="utf-8"))
    if review.get("status") != "static_review_passed":
        raise ValueError("producer trust requires a static ReviewAgent PASS record")
    if review.get("code_revision") != __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=producer.parents[2], text=True
    ).strip():
        raise ValueError("review record is not bound to the execution-server revision")
    payload = {
        "schema_version": "parta_gate_producer_trust_v1",
        "producer_name": producer.name,
        "producer_path": str(producer),
        "producer_sha256": _sha256(producer),
        "review_status": "static_review_passed",
        "review_record_path": str(args.review_record.resolve()),
        "review_record_sha256": _sha256(args.review_record),
        "code_revision": review["code_revision"],
        "frozen_on_execution_server": True,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
