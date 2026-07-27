#!/usr/bin/env python3
"""Build minimal source download lists for VSI-590K Gold-ID subsets.

This script reads only the released VSI-590K JSONL.  It does not download data.
ScanNet is intentionally excluded because its source annotations are supplied
separately for this project.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


GOLD_SOURCES = ("adt", "scannetppv2", "hypersim", "procthor")


def media_path(row: dict) -> str:
    return str(row.get("video") or row.get("image") or "")


def normalize_adt_sequence(stem: str) -> str:
    if stem.startswith("ADT_"):
        stem = stem[4:]
    for suffix in ("_preview_rgb", "_synthetic_video", "_video"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def source_item(source: str, media: str) -> str:
    path = PurePosixPath(media)
    relative = PurePosixPath(*path.parts[1:])
    if source == "adt":
        return normalize_adt_sequence(relative.stem)
    if source == "scannetppv2":
        return relative.stem
    if source == "procthor":
        # VSI paths are typically procthor/<house_id>/<video>.mp4.
        return relative.parts[0] if relative.parts else ""
    # Keep the complete relative Hypersim image path: scene, camera and frame
    # are all required to select files from its per-scene archives.
    return str(relative.with_suffix(""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    qa_counts: Counter[str] = Counter()
    media_by_source: dict[str, set[str]] = defaultdict(set)
    items_by_source: dict[str, set[str]] = defaultdict(set)

    with args.jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            media = media_path(row)
            source = media.split("/", 1)[0] if "/" in media else ""
            if source not in GOLD_SOURCES:
                continue
            qa_counts[source] += 1
            media_by_source[source].add(media)
            item = source_item(source, media)
            if item:
                items_by_source[source].add(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "adt": "adt_sequences.txt",
        "scannetppv2": "scannetppv2_scenes.txt",
        "hypersim": "hypersim_items.txt",
        "procthor": "procthor_house_ids.txt",
    }

    summary = {
        "schema_version": "vsi590k_gold_source_lists_v1",
        "jsonl": str(args.jsonl),
        "scanNet_excluded": True,
        "sources": {},
    }
    for source in GOLD_SOURCES:
        values = sorted(items_by_source[source])
        output_path = args.output_dir / output_names[source]
        output_path.write_text(
            "".join(f"{value}\n" for value in values),
            encoding="utf-8",
        )
        summary["sources"][source] = {
            "qa_rows": qa_counts[source],
            "unique_media": len(media_by_source[source]),
            "unique_download_items": len(values),
            "list": str(output_path),
            "examples": values[:10],
        }

    summary_path = args.output_dir / "gold_source_lists_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
