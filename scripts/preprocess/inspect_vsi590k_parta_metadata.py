#!/usr/bin/env python3
"""Read-only, bounded schema audit for VSI-590K Part A metadata."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


TARGET_TERMS = (
    "scene", "video", "frame", "time", "camera", "pose", "intrinsic",
    "extrinsic", "depth", "mesh", "point", "object", "instance", "track",
    "bbox", "box", "mask", "category", "semantic", "visibility", "occlu",
)


def summarize(value, depth=0):
    if depth >= 3:
        return {"type": type(value).__name__}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value)[:100],
            "values": {str(k): summarize(v, depth + 1)
                       for k, v in list(value.items())[:20]},
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first": summarize(value[0], depth + 1) if value else None,
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "first": summarize(value.flat[0], depth + 1) if value.size else None,
        }
    if isinstance(value, np.generic):
        return {"type": type(value).__name__, "value": value.item()}
    return {"type": type(value).__name__, "value": str(value)[:300]}


def find_candidate_keys(value, prefix="", depth=0):
    if depth >= 5:
        return []
    found = []
    if isinstance(value, dict):
        for key, child in list(value.items())[:500]:
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(term in str(key).lower() for term in TARGET_TERMS):
                found.append(path)
            found.extend(find_candidate_keys(child, path, depth + 1))
    elif isinstance(value, (list, tuple)) and value:
        found.extend(find_candidate_keys(value[0], f"{prefix}[0]", depth + 1))
    elif isinstance(value, np.ndarray) and value.dtype == object and value.size:
        found.extend(find_candidate_keys(value.flat[0], f"{prefix}[0]", depth + 1))
    return sorted(set(found))


def load_sample(path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return json.loads(handle.readline())
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True, mmap_mode=None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {"root": str(args.root), "files": []}
    for path in sorted(args.root.rglob("*")):
        if not path.is_file() or "/.cache/" in str(path):
            continue
        item = {"path": str(path.relative_to(args.root)), "bytes": path.stat().st_size}
        try:
            if path.suffix.lower() in {".json", ".jsonl", ".npy"}:
                sample = load_sample(path)
                item["sample_schema"] = summarize(sample)
                item["candidate_keys"] = find_candidate_keys(sample)
            elif path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                item["archive_members"] = names[:200]
                item["archive_member_count"] = len(names)
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        report["files"].append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
