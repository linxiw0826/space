#!/usr/bin/env bash

# Shared helpers for the final515k E-02c evaluation wrappers.  This file is
# sourced by wrappers that already enable `set -euo pipefail`.

mope_resolve_complete_hf_checkpoint() {
  local requested="${1:?requested checkpoint path is required}"
  python - "${requested}" <<'PY'
import json
import sys
from pathlib import Path

requested = Path(sys.argv[1])


def complete_hf_checkpoint(path):
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        return False
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (KeyError, OSError, json.JSONDecodeError):
        return False
    shards = set(weight_map.values())
    return bool(shards) and all(
        (path / shard).is_file() and (path / shard).stat().st_size > 0
        for shard in shards
    )


candidates = [requested]
if requested.is_dir():
    candidates.extend(
        sorted(
            requested.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1])
            if path.name.rsplit("-", 1)[-1].isdigit()
            else -1,
            reverse=True,
        )
    )

for candidate in candidates:
    if complete_hf_checkpoint(candidate):
        print(candidate)
        break
else:
    raise SystemExit(
        f"No complete HF checkpoint (config + index + all shards) under {requested}"
    )
PY
}

mope_assert_smoke_output_isolated() {
  local results_dir="${1:?smoke results path is required}"
  local formal_dir="${2:?formal results path is required}"
  local smoke_root="${3:?smoke root path is required}"
  python - "${results_dir}" "${formal_dir}" "${smoke_root}" <<'PY'
import sys
from pathlib import Path

results, formal, smoke_root = (Path(value).resolve() for value in sys.argv[1:])
overlaps_formal = (
    results == formal
    or results in formal.parents
    or formal in results.parents
)
inside_smoke_root = results != smoke_root and smoke_root in results.parents
if overlaps_formal or not inside_smoke_root:
    raise SystemExit(
        "Smoke output must be an isolated child of "
        f"{smoke_root} and must not overlap formal results {formal}; got {results}"
    )
PY
}

mope_validate_jsonl_count() {
  local jsonl_path="${1:?samples JSONL path is required}"
  local expected_count="${2:?expected sample count is required}"
  python - "${jsonl_path}" "${expected_count}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
count = 0
with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            raise SystemExit(f"blank samples JSONL line at {line_number}: {path}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid samples JSONL line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise SystemExit(f"samples JSONL line {line_number} is not an object")
        count += 1
if count != expected:
    raise SystemExit(
        f"samples JSONL count mismatch: expected={expected}, actual={count}, path={path}"
    )
PY
}
