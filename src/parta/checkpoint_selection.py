"""D-62 validation-only checkpoint selection contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .provenance import sha256_file


RULE = "minimum_source_balanced_val_qa_loss_tie_earliest_step_v1"


def select_validation_checkpoint(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("checkpoint selection requires validation candidates")
    normalized = []
    for item in candidates:
        losses = item.get("source_val_qa_loss", {})
        if set(losses) != {"adt", "hypersim", "scannetppv2"}:
            raise ValueError("validation loss must cover the exact D-62 source registry")
        values = [float(losses[source]) for source in sorted(losses)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("validation loss contains non-finite values")
        normalized.append({
            "step": int(item["step"]),
            "checkpoint_sha256": str(item["checkpoint_sha256"]),
            "source_val_qa_loss": dict(losses),
            "source_balanced_val_qa_loss": sum(values) / len(values),
        })
    if len({item["step"] for item in normalized}) != len(normalized):
        raise ValueError("duplicate validation checkpoint step")
    selected = min(normalized, key=lambda item: (item["source_balanced_val_qa_loss"], item["step"]))
    return {
        "schema_version": "parta_checkpoint_selection_v1",
        "selection_rule": RULE,
        "metric_source": "validation_only",
        "vsibench_used_for_selection": False,
        "selected": selected,
        "candidates": sorted(normalized, key=lambda item: item["step"]),
    }


def assert_matched_selection_rule(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for payload in (left, right):
        if payload.get("selection_rule") != RULE or payload.get("vsibench_used_for_selection") is not False:
            raise ValueError("matched arms must use the frozen validation-only selection rule")


def validate_selection_report(payload: Mapping[str, Any], *, arm: str,
                              checkpoint: Path, frozen_config_sha256: str) -> None:
    selected = payload.get("selected", {})
    if (payload.get("schema_version") != "parta_checkpoint_selection_v1"
            or payload.get("status") != "complete_selected"
            or payload.get("arm") != arm
            or payload.get("selection_rule") != RULE
            or payload.get("metric_source") != "validation_only"
            or payload.get("vsibench_used_for_selection") is not False
            or tuple(payload.get("source_registry", ())) != ("adt", "hypersim", "scannetppv2")
            or payload.get("frozen_config_artifact_sha256") != frozen_config_sha256
            or selected.get("checkpoint_sha256") != sha256_file(checkpoint)
            or Path(str(selected.get("checkpoint_path", ""))).resolve() != checkpoint.resolve()):
        raise ValueError(f"invalid formal checkpoint selection report for {arm}")
