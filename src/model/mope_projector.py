# MoPE projector - maps MoPE patch features to LLM embedding space
# Interface: MoPEProjector(mope_dim=768, llm_dim=3584).forward(mope_features) -> [B, 1, 3584]
#
# Design:
#   1. Global average pool over N_patches    -> [B, 768]
#   2. LayerNorm                             -> [B, 768]
#   3. Linear projection                     -> [B, 3584]
#   4. unsqueeze(1)                          -> [B, 1, 3584]
#
# The output [B, 1, 3584] is broadcastable over the full visual token sequence
# [B, N_tokens, 3584] via .expand_as(image_embeds), enabling per-clip bias
# injection without requiring spatial alignment with visual patch tokens.
#
# Only this module is trainable in the MoPE integration; MoPEEncoder is frozen.

import torch
import torch.nn as nn


class MoPEProjector(nn.Module):
    """
    Lightweight projector that condenses MoPE patch features into a single
    LLM-dimensional embedding vector per clip.

    Args:
        mope_dim: Dimensionality of MoPE patch features (default: 768, ViT-B).
        llm_dim:  LLM hidden dimension to project into (default: 3584, Qwen3-VL-7B).
    """

    def __init__(self, mope_dim: int = 768, llm_dim: int = 3584):
        super().__init__()
        self.mope_dim = mope_dim
        self.llm_dim = llm_dim

        self.norm = nn.LayerNorm(mope_dim)
        self.proj = nn.Linear(mope_dim, llm_dim, bias=True)

        # Zero-initialize so MoPE contribution is strictly zero at training
        # start, preserving GUIDE's learned geometric priors as the baseline.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, mope_features: torch.Tensor) -> torch.Tensor:
        """
        Project MoPE patch features to LLM embedding space.

        Args:
            mope_features: Float tensor [B, N_patches, 768] from MoPEEncoder.

        Returns:
            mope_embeds: Float tensor [B, 1, llm_dim] ready to broadcast
                         over visual tokens via .expand_as(image_embeds).
        """
        # Step 1: global average pool over N_patches -> [B, 768]
        x = mope_features.mean(dim=1)

        # Step 2: layer norm -> [B, 768]
        x = self.norm(x)

        # Step 3: linear projection -> [B, llm_dim]
        x = self.proj(x)

        # Step 4: add token dimension -> [B, 1, llm_dim]
        x = x.unsqueeze(1)

        return x
