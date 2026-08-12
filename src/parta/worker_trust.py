"""Fail-closed declarations for repository Python workers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TRAIN_WORKER_VALUE_FLAGS = frozenset({
    "--arm", "--manifest", "--manifest-report", "--source-root", "--media-root",
    "--model-path", "--vggt-path", "--output-dir", "--matched-contract", "--seed",
    "--learning-rate", "--weight-decay", "--gradient-accumulation-steps",
    "--max-steps", "--save-steps", "--dtype", "--device", "--distributed-strategy",
    "--num-workers", "--engineering-subset", "--engineering-mode",
    "--required-frame-count", "--max-grad-norm",
})
TRAIN_WORKER_SWITCH_FLAGS = frozenset({"--dry-run", "--gradient-checkpointing"})


def train_worker_flag_contract(engineering_mode: str) -> dict[str, str | None]:
    contract = {name: None for name in TRAIN_WORKER_VALUE_FLAGS}
    contract["--engineering-mode"] = engineering_mode
    return contract


def validate_python_worker(record: Mapping[str, Any], argv: Sequence[str], *,
                           script: Path, script_sha256: str, git_revision: str,
                           engineering_mode: str,
                           allowed_value_flags: Mapping[str, str | None],
                           allowed_switch_flags: Sequence[str] = (),
                           source_registry: Sequence[str] = ("adt", "hypersim", "scannetppv2")) -> None:
    canonical = script.resolve()
    if (not isinstance(argv, list) or len(argv) < 2
            or argv[0] != sys.executable or argv[1] != str(canonical)
            or sum(str(value) == str(canonical) for value in argv) != 1
            or record.get("python_executable") != sys.executable
            or Path(str(record.get("script_path", ""))).resolve() != canonical
            or record.get("script_sha256") != script_sha256
            or record.get("git_revision") != git_revision):
        raise ValueError("untrusted repository worker declaration")
    switches = set(allowed_switch_flags)
    if set(allowed_value_flags) & switches:
        raise ValueError("worker flag schema overlaps")
    observed: dict[str, str | bool] = {}
    source_roots: dict[str, str] = {}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ValueError("untrusted repository worker declaration")
        if flag in observed and flag != "--source-root":
            raise ValueError("untrusted repository worker declaration")
        if flag in switches:
            observed[flag] = True
            index += 1
            continue
        if flag not in allowed_value_flags or index + 1 >= len(argv):
            raise ValueError("untrusted repository worker declaration")
        value = argv[index + 1]
        if not isinstance(value, str) or value.startswith("--"):
            raise ValueError("untrusted repository worker declaration")
        if flag == "--source-root":
            if "=" not in value:
                raise ValueError("untrusted repository worker declaration")
            source, raw_path = value.split("=", 1)
            if source in source_roots or source not in source_registry or not raw_path:
                raise ValueError("untrusted repository worker declaration")
            source_roots[source] = str(Path(raw_path).resolve())
            observed[flag] = "<three-source-registry>"
        else:
            observed[flag] = value
        index += 2
    expected_mode = observed.get("--engineering-mode")
    if expected_mode != engineering_mode:
        raise ValueError("untrusted repository worker declaration")
    if set(source_roots) != set(source_registry):
        raise ValueError("untrusted repository worker declaration")
    manifest_report_value = observed.get("--manifest-report")
    if source_registry and not isinstance(manifest_report_value, str):
        raise ValueError("untrusted repository worker declaration")
    if source_registry:
        report_path = Path(manifest_report_value).resolve()
        if not report_path.is_file():
            raise ValueError("untrusted repository worker declaration")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        registry = report.get("exact_canonical_inputs", {})
        bound_roots = {source: str(Path(str(item.get("root", ""))).resolve())
                       for source, item in registry.items()}
        if set(registry) != set(source_registry) or bound_roots != source_roots:
            raise ValueError("untrusted repository worker declaration")
    for flag, expected in allowed_value_flags.items():
        if expected is not None and observed.get(flag) != expected:
            raise ValueError("untrusted repository worker declaration")
