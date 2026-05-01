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


class MoPEProjectorConcat(nn.Module):
    """E-02b per-token projector — no global avg pool.

    Projects each MoPE patch token independently to LLM embedding space.
    Output shape [B, N_patches, llm_dim] is concatenated into the LLM input
    sequence rather than broadcast-added.

    Args:
        mope_dim: MoPE feature dimension (default: 768, ViT-B).
        llm_dim:  LLM hidden dimension (default: 3584, Qwen3-VL-4B).
    """

    def __init__(self, mope_dim: int = 768, llm_dim: int = 3584):
        super().__init__()
        self.mope_dim = mope_dim
        self.llm_dim = llm_dim

        self.norm = nn.LayerNorm(mope_dim)
        self.proj = nn.Linear(mope_dim, llm_dim, bias=True)

        # Zero-init: MoPE contribution is zero at training start.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, mope_features: torch.Tensor) -> torch.Tensor:
        """Project each MoPE patch token to LLM embedding space.

        Args:
            mope_features: Float tensor [B, N_patches, mope_dim].

        Returns:
            mope_embeds: Float tensor [B, N_patches, llm_dim] for sequence concat.
        """
        x = self.norm(mope_features)   # [B, N_patches, mope_dim]
        x = self.proj(x)               # [B, N_patches, llm_dim]
        return x


class MoPEProjectorCrossAttn(nn.Module):
    """E-02c single-head single-layer cross-attention projector.

    Fuses MoPE features into each image token via residual cross-attention.
    MoPE tokens are keys/values; image tokens are queries.

    Args:
        mope_dim: MoPE feature dimension (default: 768, ViT-B).
        llm_dim:  LLM hidden dimension (default: 3584, Qwen3-VL-4B).

    Forward:
        mope_features: [B, N_mope, mope_dim]  — MoPE encoder output (N_mope=784)
        image_embeds:  [B, N_img,  llm_dim]   — one image-embed shard
        returns:       [B, N_img,  llm_dim]   — residual-updated image_embeds
    """

    def __init__(self, mope_dim: int = 768, llm_dim: int = 3584):
        super().__init__()
        self.mope_dim = mope_dim
        self.llm_dim = llm_dim

        self.norm = nn.LayerNorm(mope_dim)
        self.k_proj = nn.Linear(mope_dim, llm_dim)
        self.v_proj = nn.Linear(mope_dim, llm_dim)

        # Zero-init: out_proj weight and bias both zero so MoPE contribution
        # is strictly zero at training start, preserving GUIDE's geometric
        # priors as the baseline starting point.
        self.out_proj = nn.Linear(llm_dim, llm_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        mope_features: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Apply single-head cross-attention and residual-add to image_embeds.

        Args:
            mope_features: Float tensor [B, N_mope, mope_dim].
            image_embeds:  Float tensor [B, N_img, llm_dim].

        Returns:
            updated_embeds: Float tensor [B, N_img, llm_dim].
        """
        x = self.norm(mope_features)                                   # [B, N_mope, mope_dim]
        K = self.k_proj(x)                                             # [B, N_mope, llm_dim]
        V = self.v_proj(x)                                             # [B, N_mope, llm_dim]
        Q = image_embeds                                               # [B, N_img,  llm_dim]

        scale = Q.shape[-1] ** -0.5
        attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) # [B, N_img, N_mope]
        out = attn @ V                                                 # [B, N_img, llm_dim]
        out = self.out_proj(out)                                       # [B, N_img, llm_dim]
        return image_embeds + out.to(image_embeds.dtype)


class MoPEProjectorQFormer(nn.Module):
    """E-02d Q-Former projector — compresses 784 MoPE tokens to num_queries tokens.

    Uses num_queries learnable query vectors as Q in single-head cross-attention
    against MoPE features. Output tokens are concatenated to the LLM input sequence.

    Args:
        mope_dim:    MoPE feature dimension (default: 768, ViT-B).
        llm_dim:     LLM hidden dimension (default: 3584, Qwen3-VL-4B).
        num_queries: Number of compressed output tokens (default: 32).

    Forward:
        mope_features: [B, N_mope, mope_dim]   — MoPE encoder output (N_mope=784)
        returns:       [B, num_queries, llm_dim] — compressed tokens for LLM concat
    """

    def __init__(self, mope_dim: int = 768, llm_dim: int = 3584, num_queries: int = 32):
        super().__init__()
        self.mope_dim = mope_dim
        self.llm_dim = llm_dim
        self.num_queries = num_queries

        self.norm = nn.LayerNorm(mope_dim)
        self.k_proj = nn.Linear(mope_dim, llm_dim)
        self.v_proj = nn.Linear(mope_dim, llm_dim)

        # Zero-init: queries start at zero and out_proj maps to zero so the
        # 32 prepended tokens are zero vectors at training start, introducing
        # no perturbation to the LLM until the model begins to learn.
        self.out_proj = nn.Linear(llm_dim, llm_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.queries = nn.Parameter(torch.zeros(1, num_queries, llm_dim))

    def forward(self, mope_features: torch.Tensor) -> torch.Tensor:
        """Compress MoPE features to num_queries tokens via learned cross-attention.

        Args:
            mope_features: Float tensor [B, N_mope, mope_dim].

        Returns:
            compressed: Float tensor [B, num_queries, llm_dim] for LLM sequence concat.
        """
        B = mope_features.shape[0]
        x = self.norm(mope_features)                                   # [B, N_mope, mope_dim]
        K = self.k_proj(x)                                             # [B, N_mope, llm_dim]
        V = self.v_proj(x)                                             # [B, N_mope, llm_dim]
        Q = self.queries.expand(B, -1, -1)                             # [B, num_queries, llm_dim]

        scale = Q.shape[-1] ** -0.5
        attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1) # [B, num_queries, N_mope]
        out = attn @ V                                                 # [B, num_queries, llm_dim]
        out = self.out_proj(out)                                       # [B, num_queries, llm_dim]
        return out
