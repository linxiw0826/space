"""Machine-readable diagnostics for matched Part A training."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingStepRecord:
    step: int
    arm: str
    source_dataset: str
    qa_loss: float
    state_loss: float
    total_loss: float
    loss_existence: float
    loss_category: float
    loss_center: float
    loss_extent: float
    loss_visibility: float
    grad_norm: float
    shared_grad_norm: float
    state_head_grad_norm: float
    effective_gt_objects: int
    matched_slots: int
    unmatched_slots: int
    matching_mean_cost: float
    matching_valid_pairs: int
    actual_frame_count: int
    learning_rate: float
    step_seconds: float
    samples_per_second: float
    frames_per_second: float
    peak_cuda_memory_bytes: int | None
    lambda_state: float
    schema_version: str = "parta_training_step_v1"

    def validate(self) -> None:
        if self.arm not in {"a0", "a1o"}:
            raise ValueError(f"unknown arm: {self.arm}")
        if self.step < 0 or self.actual_frame_count < 1 or self.effective_gt_objects < 0:
            raise ValueError("invalid non-negative diagnostic counter")
        if self.matched_slots < 0 or self.unmatched_slots < 0:
            raise ValueError("invalid slot diagnostic counter")
        if self.arm == "a0" and (self.state_loss != 0.0 or self.lambda_state != 0.0):
            raise ValueError("A0 diagnostic must have zero state loss and lambda")


class JsonlTrainingLogger:
    """Append and fsync each record so monitor processes see committed steps."""

    def __init__(self, path: str | Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TrainingStepRecord, *, extra: Mapping[str, Any] | None = None) -> None:
        record.validate()
        if not self.enabled:
            return
        payload = asdict(record)
        if extra:
            overlap = set(payload) & set(extra)
            if overlap:
                raise ValueError(f"extra diagnostics overwrite schema fields: {sorted(overlap)}")
            payload.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
