"""Thin A1-O training integration; A0 does not import or instantiate this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .state_head import (
    ObjectStatePredictions,
    SetSlotStateHead,
    StateHeadConfig,
    StateTapOutput,
    build_state_tap_from_packed,
)
from .state_loss import ObjectStateSetLoss, StateLossConfig, StateTargets


@dataclass
class A1OSideBranchOutput:
    tap: StateTapOutput
    predictions: ObjectStatePredictions
    losses: dict[str, object]


def attach_a1o_state_head(model: nn.Module, config: StateHeadConfig) -> SetSlotStateHead:
    """Attach the side head under the checkpoint-stable key ``parta_state_head``.

    The function fails closed if a MoPE module is present: D-57 freezes both A0
    and A1-O to no MoPE. A0 must never call this function.
    """
    inner = getattr(model, "model", model)
    if getattr(inner, "_mope_encoder", None) is not None or getattr(inner, "_mope_projector", None) is not None:
        raise ValueError("D-57: A1-O must not instantiate/input/train MoPE")
    if hasattr(model, "parta_state_head"):
        raise ValueError("parta_state_head is already attached")
    head = SetSlotStateHead(config)
    model.add_module("parta_state_head", head)
    return head


def run_a1o_side_branch(
    model: nn.Module,
    visual_state_hidden: torch.Tensor,
    visual_state_valid_mask: torch.Tensor,
    frame_token_counts: Sequence[Sequence[int]],
    frame_ids: Sequence[Sequence[int]],
    targets: Sequence[StateTargets],
    loss_config: StateLossConfig,
) -> A1OSideBranchOutput:
    """Run the training-only side path after the shared QA forward."""
    inner = getattr(model, "model", model)
    if getattr(inner, "_mope_encoder", None) is not None or getattr(inner, "_mope_projector", None) is not None:
        raise ValueError("D-57: A1-O runtime must remain MoPE-free")
    head = getattr(model, "parta_state_head", None)
    if head is None:
        raise ValueError("A1-O side branch requested without an attached state head")
    tap = build_state_tap_from_packed(
        visual_state_hidden,
        visual_state_valid_mask,
        frame_token_counts,
        frame_ids,
    )
    predictions = head(tap)
    losses = ObjectStateSetLoss(loss_config)(predictions, targets)
    return A1OSideBranchOutput(tap=tap, predictions=predictions, losses=losses)
