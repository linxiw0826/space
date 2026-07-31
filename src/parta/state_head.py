"""Question-independent set-slot side head for Part A A1-O.

The module is deliberately independent of the QA sequence. Callers must pass
the final LLM hidden states and the *same* ``visual_pos_masks`` used by the
multimodal scatter in Qwen3-VL. The returned slots never re-enter the QA path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class StateHeadConfig:
    hidden_size: int
    num_categories: int
    num_slots: int = 384
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.0
    max_frames: int = 32

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_slots != 384:
            raise ValueError("D-58 freezes A1-O v1 to K=384 slots")
        if self.max_frames != 32:
            raise ValueError("D-56 freezes the Part A maximum to 32 frames")


@dataclass
class StateTapOutput:
    hidden: torch.Tensor
    valid_mask: torch.Tensor
    frame_ids: torch.Tensor
    frame_valid_mask: torch.Tensor
    token_frame_index: torch.Tensor
    frame_token_spans: torch.Tensor
    tap_layer: str = "final"


@dataclass
class ObjectStatePredictions:
    existence_logits: torch.Tensor
    category_logits: torch.Tensor
    center_world_normalized: torch.Tensor
    extent_normalized: torch.Tensor
    visibility_logits: torch.Tensor
    slots: torch.Tensor


def extract_visual_prefix_hidden(
    last_hidden_state: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    frame_token_counts: Sequence[Sequence[int]],
    frame_ids: Sequence[Sequence[int]],
) -> StateTapOutput:
    """Restore flat visual positions to a padded, frame-aware batch.

    Args:
        last_hidden_state: Final normalized LLM hidden, ``[B, L, D]``.
        visual_pos_masks: Authoritative Qwen forward visual mask, ``[B, L]``.
        frame_token_counts: Visual placeholder count for every actual frame.
        frame_ids: Exact sampled source frame IDs in temporal order.
    """
    if last_hidden_state.ndim != 3 or visual_pos_masks.ndim != 2:
        raise ValueError("expected hidden [B,L,D] and visual_pos_masks [B,L]")
    if last_hidden_state.shape[:2] != visual_pos_masks.shape:
        raise ValueError("hidden and visual_pos_masks sequence shapes differ")
    batch_size, _, hidden_size = last_hidden_state.shape
    if len(frame_token_counts) != batch_size or len(frame_ids) != batch_size:
        raise ValueError("frame metadata batch size differs from hidden batch")

    per_sample = []
    max_tokens = 0
    max_frames = max((len(x) for x in frame_ids), default=0)
    for batch_index in range(batch_size):
        counts = [int(value) for value in frame_token_counts[batch_index]]
        ids = [int(value) for value in frame_ids[batch_index]]
        if len(counts) != len(ids):
            raise ValueError(f"sample {batch_index}: frame count/ID length mismatch")
        if not 16 <= len(ids) <= 32:
            raise ValueError(f"sample {batch_index}: expected 16-32 frames, got {len(ids)}")
        visual = last_hidden_state[batch_index, visual_pos_masks[batch_index].bool()]
        if sum(counts) != visual.shape[0]:
            raise ValueError(
                f"sample {batch_index}: frame token total {sum(counts)} "
                f"!= visual mask total {visual.shape[0]}"
            )
        per_sample.append((visual, counts, ids))
        max_tokens = max(max_tokens, visual.shape[0])

    hidden = last_hidden_state.new_zeros((batch_size, max_tokens, hidden_size))
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=last_hidden_state.device)
    token_frame_index = torch.full(
        (batch_size, max_tokens), -1, dtype=torch.long, device=last_hidden_state.device
    )
    padded_frame_ids = torch.full(
        (batch_size, max_frames), -1, dtype=torch.long, device=last_hidden_state.device
    )
    frame_valid_mask = torch.zeros(
        (batch_size, max_frames), dtype=torch.bool, device=last_hidden_state.device
    )
    frame_token_spans = torch.full(
        (batch_size, max_frames, 2), -1, dtype=torch.long, device=last_hidden_state.device
    )

    for batch_index, (visual, counts, ids) in enumerate(per_sample):
        hidden[batch_index, : visual.shape[0]] = visual
        valid_mask[batch_index, : visual.shape[0]] = True
        offset = 0
        for frame_index, (count, frame_id) in enumerate(zip(counts, ids)):
            end = offset + count
            token_frame_index[batch_index, offset:end] = frame_index
            padded_frame_ids[batch_index, frame_index] = frame_id
            frame_valid_mask[batch_index, frame_index] = True
            frame_token_spans[batch_index, frame_index] = torch.tensor(
                [offset, end], device=last_hidden_state.device
            )
            offset = end

    return StateTapOutput(
        hidden=hidden,
        valid_mask=valid_mask,
        frame_ids=padded_frame_ids,
        frame_valid_mask=frame_valid_mask,
        token_frame_index=token_frame_index,
        frame_token_spans=frame_token_spans,
    )


def build_state_tap_from_packed(
    visual_state_hidden: torch.Tensor,
    visual_state_valid_mask: torch.Tensor,
    frame_token_counts: Sequence[Sequence[int]],
    frame_ids: Sequence[Sequence[int]],
) -> StateTapOutput:
    """Add exact frame metadata to the packed side output of Qwen3-VL."""
    if visual_state_hidden.ndim != 3 or visual_state_valid_mask.shape != visual_state_hidden.shape[:2]:
        raise ValueError("invalid packed visual state hidden/mask")
    batch_size, max_tokens, _ = visual_state_hidden.shape
    if len(frame_ids) != batch_size or len(frame_token_counts) != batch_size:
        raise ValueError("packed visual state metadata batch mismatch")
    max_frames = max(len(ids) for ids in frame_ids)
    padded_ids = torch.full(
        (batch_size, max_frames), -1, dtype=torch.long, device=visual_state_hidden.device
    )
    frame_valid = torch.zeros(
        (batch_size, max_frames), dtype=torch.bool, device=visual_state_hidden.device
    )
    token_frame_index = torch.full(
        (batch_size, max_tokens), -1, dtype=torch.long, device=visual_state_hidden.device
    )
    spans = torch.full(
        (batch_size, max_frames, 2), -1, dtype=torch.long, device=visual_state_hidden.device
    )
    for batch_index, (counts_raw, ids_raw) in enumerate(zip(frame_token_counts, frame_ids)):
        counts = [int(value) for value in counts_raw]
        ids = [int(value) for value in ids_raw]
        if len(counts) != len(ids) or not 16 <= len(ids) <= 32:
            raise ValueError(f"sample {batch_index}: invalid 16-32 frame metadata")
        actual_tokens = int(visual_state_valid_mask[batch_index].sum().item())
        if sum(counts) != actual_tokens:
            raise ValueError(
                f"sample {batch_index}: frame token total {sum(counts)} != packed valid total {actual_tokens}"
            )
        offset = 0
        for frame_index, (count, frame_id) in enumerate(zip(counts, ids)):
            end = offset + count
            padded_ids[batch_index, frame_index] = frame_id
            frame_valid[batch_index, frame_index] = True
            token_frame_index[batch_index, offset:end] = frame_index
            spans[batch_index, frame_index] = torch.tensor(
                [offset, end], device=visual_state_hidden.device
            )
            offset = end
    return StateTapOutput(
        hidden=visual_state_hidden,
        valid_mask=visual_state_valid_mask,
        frame_ids=padded_ids,
        frame_valid_mask=frame_valid,
        token_frame_index=token_frame_index,
        frame_token_spans=spans,
    )


class SetSlotStateHead(nn.Module):
    """Independent cross-attention decoder producing an unordered object set."""

    def __init__(self, config: StateHeadConfig):
        super().__init__()
        self.config = config
        self.slot_queries = nn.Parameter(torch.empty(config.num_slots, config.hidden_size))
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.existence = nn.Linear(config.hidden_size, 1)
        self.category = nn.Linear(config.hidden_size, config.num_categories)
        self.center = nn.Linear(config.hidden_size, 3)
        self.extent = nn.Linear(config.hidden_size, 3)
        self.visibility = nn.Linear(config.hidden_size, config.max_frames)
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, tap: StateTapOutput) -> ObjectStatePredictions:
        if tap.hidden.ndim != 3 or tap.valid_mask.shape != tap.hidden.shape[:2]:
            raise ValueError("invalid StateTapOutput hidden/valid_mask")
        if not tap.valid_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one visual token")
        batch_size = tap.hidden.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        slots = self.decoder(
            tgt=queries,
            memory=tap.hidden,
            memory_key_padding_mask=~tap.valid_mask,
        )
        slots = self.output_norm(slots)
        return ObjectStatePredictions(
            existence_logits=self.existence(slots).squeeze(-1),
            category_logits=self.category(slots),
            center_world_normalized=self.center(slots),
            extent_normalized=self.extent(slots),
            visibility_logits=self.visibility(slots),
            slots=slots,
        )
