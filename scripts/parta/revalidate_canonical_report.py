#!/usr/bin/env python3
"""Recompute a v2 canonical validation report from immutable canonical JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.canonical_revalidation import (  # noqa: E402
    publish_validation_report,
    recompute_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=("adt", "hypersim", "scannetppv2"), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: ROOT/validation_report.json. Existing files are never replaced implicitly.",
    )
    parser.add_argument("--replace-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.root / "validation_report.json"
    payload = recompute_validation_report(
        source=args.source,
        root=args.root,
        project_root=PROJECT,
        producer=Path(__file__),
    )
    backup = publish_validation_report(
        payload, output, replace_existing=args.replace_existing
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source": args.source,
                "output": str(output.resolve()),
                "backup": str(backup.resolve()) if backup is not None else None,
                "canonical_inputs": payload["canonical_inputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
