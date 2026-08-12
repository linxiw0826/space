"""Thin A1-O training integration; A0 does not import or instantiate this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

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


def install_a1o_forward_integration(model: nn.Module) -> None:
    """Execute the A1-O branch inside ``model.forward`` (and hence FSDP).

    The transient request/result are ordinary Python attributes and are never
    checkpointed.  The state head itself remains a registered child module, so
    wrapping the parent includes its parameters in optimizer/FSDP state.
    """
    if not hasattr(model, "parta_state_head"):
        raise ValueError("forward integration requires an attached A1-O head")
    if getattr(model, "_parta_forward_hook_handle", None) is not None:
        raise ValueError("A1-O forward integration is already installed")
    model._parta_side_request = None
    model._parta_side_result = None

    def _hook(module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any):
        request = module._parta_side_request
        if request is None:
            return output
        if module._parta_side_result is not None:
            raise RuntimeError("unconsumed A1-O forward result")
        hidden = getattr(output, "visual_state_hidden", None)
        valid = getattr(output, "visual_state_valid_mask", None)
        if hidden is None or valid is None:
            raise ValueError("A1-O forward lacks authoritative visual state tap")
        module._parta_side_result = _run_a1o_side_branch_unwrapped(
            module, hidden, valid, **request
        )
        return output

    model._parta_forward_hook_handle = model.register_forward_hook(_hook, with_kwargs=True)


def prepare_a1o_forward_request(model: nn.Module, **request: Any) -> None:
    inner = _unwrap_parallel(model)
    if getattr(inner, "_parta_forward_hook_handle", None) is None:
        raise RuntimeError("A1-O side branch is not integrated into model.forward")
    if inner._parta_side_request is not None or inner._parta_side_result is not None:
        raise RuntimeError("stale A1-O forward request/result")
    inner._parta_side_request = request


def consume_a1o_forward_result(model: nn.Module) -> A1OSideBranchOutput:
    inner = _unwrap_parallel(model)
    result = inner._parta_side_result
    inner._parta_side_request = None
    inner._parta_side_result = None
    if result is None:
        raise RuntimeError("A1-O model.forward did not produce its integrated side result")
    return result


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


def _unwrap_parallel(model: nn.Module) -> nn.Module:
    while True:
        wrapped = getattr(model, "module", None)
        if not isinstance(wrapped, nn.Module):
            wrapped = getattr(model, "_fsdp_wrapped_module", None)
        if not isinstance(wrapped, nn.Module) or wrapped is model:
            return model
        model = wrapped


def _run_a1o_side_branch_unwrapped(
    model: nn.Module,
    visual_state_hidden: torch.Tensor,
    visual_state_valid_mask: torch.Tensor,
    frame_token_counts: Sequence[Sequence[int]],
    frame_ids: Sequence[Sequence[int]],
    media_kinds: Sequence[str],
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
        media_kinds,
    )
    predictions = head(tap)
    losses = ObjectStateSetLoss(loss_config)(predictions, targets)
    return A1OSideBranchOutput(tap=tap, predictions=predictions, losses=losses)


def run_a1o_side_branch(
    model: nn.Module,
    visual_state_hidden: torch.Tensor,
    visual_state_valid_mask: torch.Tensor,
    frame_token_counts: Sequence[Sequence[int]],
    frame_ids: Sequence[Sequence[int]],
    media_kinds: Sequence[str],
    targets: Sequence[StateTargets],
    loss_config: StateLossConfig,
) -> A1OSideBranchOutput:
    """Legacy non-parallel seam used by T0-A and CPU unit tests only.

    Formal training installs :func:`install_a1o_forward_integration`; calling
    this function on any parallel wrapper is forbidden because FSDP may already
    have resharded the head parameters.
    """
    if getattr(model, "module", None) is not None or getattr(
        model, "_fsdp_wrapped_module", None
    ) is not None:
        raise RuntimeError("parallel A1-O side branch must execute inside wrapped model.forward")
    return _run_a1o_side_branch_unwrapped(
        model, visual_state_hidden, visual_state_valid_mask, frame_token_counts,
        frame_ids, media_kinds, targets, loss_config,
    )
