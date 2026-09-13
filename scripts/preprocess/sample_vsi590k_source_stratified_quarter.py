#!/usr/bin/env python3
"""Create a deterministic source-stratified quarter VSI-590K manifest."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def source_of(row: dict) -> str:
    parts = Path(row["mope_video"]).parts
    try:
        return parts[parts.index("VSI-590K") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot infer source from mope_video: {row.get('mope_video')}") from exc


def sample(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[source_of(row)].append(row)
    selected: list[dict] = []
    stats = {}
    for source in sorted(groups):
        source_rows = groups[source]
        target = round(len(source_rows) * fraction)
        rng = random.Random(f"{seed}:{source}")
        chosen = [source_rows[i] for i in sorted(rng.sample(range(len(source_rows)), target))]
        selected.extend(chosen)
        stats[source] = {
            "original_rows": len(source_rows),
            "selected_rows": len(chosen),
            "original_videos": len({r["mope_video"] for r in source_rows}),
            "selected_videos": len({r["mope_video"] for r in chosen}),
        }
    random.Random(seed).shuffle(selected)
    for item in stats.values():
        item["fraction"] = item["selected_rows"] / item["original_rows"]
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise SystemExit("fraction must be in (0, 1]")
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("output must differ from input")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit("input must be a non-empty JSON list")
    selected, stats = sample(rows, args.fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps({
        "input": str(args.input), "output": str(args.output),
        "fraction": args.fraction, "seed": args.seed,
        "sampling_unit": "row", "original_rows": len(rows),
        "selected_rows": len(selected), "sources": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS rows={len(rows)} -> {len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
