"""A1-O-drop checkpoint filtering and load-audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

STATE_HEAD_PREFIXES = ("parta_state_head.", "module.parta_state_head.")


@dataclass(frozen=True)
class HeadFreeLoadAudit:
    dropped_state_head_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    loaded_shared_keys: int
    expected_state_head_keys: tuple[str, ...] = ()
    minimum_dropped_state_head_keys: int = 1

    @property
    def passed(self) -> bool:
        expected_ok = (
            set(self.dropped_state_head_keys) == set(self.expected_state_head_keys)
            if self.expected_state_head_keys
            else len(self.dropped_state_head_keys) >= self.minimum_dropped_state_head_keys
        )
        return expected_ok and not self.missing_keys and not self.unexpected_keys


def is_state_head_key(key: str) -> bool:
    return key.startswith(STATE_HEAD_PREFIXES)


def filter_head_free_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Remove only the training-side state head from an A1-O checkpoint."""
    kept = {}
    dropped = []
    for key, value in state_dict.items():
        if is_state_head_key(key):
            dropped.append(key)
        else:
            kept[key] = value
    return kept, tuple(sorted(dropped))


def load_head_free_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    expected_state_head_keys: tuple[str, ...] = (),
    minimum_dropped_state_head_keys: int = 1,
) -> HeadFreeLoadAudit:
    """Load common QA/backbone weights without ever constructing the head."""
    filtered, dropped = filter_head_free_state_dict(state_dict)
    incompatible = model.load_state_dict(filtered, strict=False)
    missing = tuple(sorted(key for key in incompatible.missing_keys if not is_state_head_key(key)))
    unexpected = tuple(sorted(key for key in incompatible.unexpected_keys if not is_state_head_key(key)))
    audit = HeadFreeLoadAudit(
        dropped_state_head_keys=dropped,
        missing_keys=missing,
        unexpected_keys=unexpected,
        loaded_shared_keys=len(filtered) - len(unexpected),
        expected_state_head_keys=tuple(sorted(expected_state_head_keys)),
        minimum_dropped_state_head_keys=minimum_dropped_state_head_keys,
    )
    return audit
