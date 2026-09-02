#!/usr/bin/env python3
"""Verify that E-04a changed only its fresh MoPE CrossAttn projector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# This verifier is intentionally runnable from a fresh login shell, separate
# from the training wrapper that exported PYTHONPATH for its own child process.
SPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPACE_ROOT / "src"))

import torch
from safetensors import safe_open

from model.mope_new_encoder import PROJECTOR_KEYS


EXPECTED_CONTRACT = [2, 12, 8, 8, 1, 1, 16, 4, 4, 224, 1]
MOPE_PREFIX = "model._mope_"
PROJECTOR_PREFIX = "model._mope_projector."
ENCODER_CONTRACT_KEY = "model._mope_encoder.contract"


def _weight_map(root: Path) -> dict[str, str]:
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise RuntimeError(f"incomplete HF checkpoint metadata: {root}")
    try:
        mapping = json.loads(index.read_text())["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid HF checkpoint index: {root}") from exc
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"empty HF checkpoint index: {root}")
    for shard in set(mapping.values()):
        path = root / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty HF shard: {path}")
    return {str(key): str(value) for key, value in mapping.items()}


def verify(base: Path, candidate: Path) -> dict[str, object]:
    base = base.resolve()
    candidate = candidate.resolve()
    base_map = _weight_map(base)
    candidate_map = _weight_map(candidate)
    if any(key.startswith(MOPE_PREFIX) for key in base_map):
        raise RuntimeError("E-01 base unexpectedly contains MoPE tensors")

    missing_base = sorted(set(base_map) - set(candidate_map))
    unexpected_non_mope = sorted(
        key for key in set(candidate_map) - set(base_map)
        if not key.startswith(MOPE_PREFIX)
    )
    if missing_base or unexpected_non_mope:
        raise RuntimeError(
            f"backbone key mismatch: missing={missing_base[:8]}, "
            f"unexpected={unexpected_non_mope[:8]}"
        )

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, base_shard in base_map.items():
        groups[(base_shard, candidate_map[key])].append(key)

    compared = 0
    for (base_shard, candidate_shard), keys in groups.items():
        with safe_open(str(base / base_shard), framework="pt", device="cpu") as left, safe_open(
            str(candidate / candidate_shard), framework="pt", device="cpu"
        ) as right:
            for key in keys:
                base_tensor = left.get_tensor(key)
                candidate_tensor = right.get_tensor(key)
                if base_tensor.dtype != candidate_tensor.dtype or not torch.equal(
                    base_tensor, candidate_tensor
                ):
                    raise RuntimeError(f"frozen E-01 tensor changed: {key}")
                compared += 1

    projector_names = {
        key[len(PROJECTOR_PREFIX):]
        for key in candidate_map
        if key.startswith(PROJECTOR_PREFIX)
    }
    if projector_names != set(PROJECTOR_KEYS):
        raise RuntimeError(
            f"projector keys mismatch: missing={sorted(set(PROJECTOR_KEYS) - projector_names)}, "
            f"unexpected={sorted(projector_names - set(PROJECTOR_KEYS))}"
        )

    projector_norms = {}
    for name in PROJECTOR_KEYS:
        key = PROJECTOR_PREFIX + name
        with safe_open(
            str(candidate / candidate_map[key]), framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(key)
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite projector tensor: {key}")
        projector_norms[name] = float(tensor.float().norm().item())
    if projector_norms["out_proj.weight"] == 0.0:
        raise RuntimeError("projector out_proj.weight is still zero after training")

    if ENCODER_CONTRACT_KEY not in candidate_map:
        raise RuntimeError("candidate has no final515k contract tensor")
    with safe_open(
        str(candidate / candidate_map[ENCODER_CONTRACT_KEY]), framework="pt", device="cpu"
    ) as handle:
        contract = handle.get_tensor(ENCODER_CONTRACT_KEY).cpu().tolist()
    if contract != EXPECTED_CONTRACT:
        raise RuntimeError(f"final515k contract mismatch: {contract}")

    return {
        "schema_version": "e04a_checkpoint_verification_v1",
        "status": "complete_passed",
        "base": str(base),
        "candidate": str(candidate),
        "frozen_tensors_compared_exactly": compared,
        "projector_keys": sorted(projector_names),
        "projector_norms": projector_norms,
        "final515k_contract": contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.base, args.candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    print(
        f"STATUS=complete_passed frozen={report['frozen_tensors_compared_exactly']} "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
