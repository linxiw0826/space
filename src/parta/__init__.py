"""Part A A1-O training-only object-state supervision components."""

from .state_head import (
    ObjectStatePredictions,
    SetSlotStateHead,
    StateHeadConfig,
    StateTapOutput,
    build_state_tap_from_packed,
    extract_visual_prefix_hidden,
)
from .state_loss import ObjectStateSetLoss, StateLossConfig, StateTargets
from .training import attach_a1o_state_head, run_a1o_side_branch

__all__ = [
    "ObjectStatePredictions",
    "ObjectStateSetLoss",
    "SetSlotStateHead",
    "StateHeadConfig",
    "StateLossConfig",
    "StateTapOutput",
    "StateTargets",
    "attach_a1o_state_head",
    "build_state_tap_from_packed",
    "extract_visual_prefix_hidden",
    "run_a1o_side_branch",
]
