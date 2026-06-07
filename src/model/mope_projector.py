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

from typing import Optional

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

    E-10 (Router v1) extension
    --------------------------
    When ``use_gate=True`` an optional learned, content-driven scalar gate
    ``g in (0, 1)`` modulates the MoPE residual::

        return image_embeds + g * out          # vs E-03a: image_embeds + out

    The gate is a small MLP that maps a pooled *text* representation
    ``cond_text`` (shape [B, llm_dim], supplied by the caller — see
    ``mope_patch._patch_model_for_mope_crossattn``) to one scalar per sample
    (``per-sample`` granularity, broadcast over the [N_img, llm_dim] residual).
    A scalar gate is the simplest, most interpretable router head: the gate
    value of each question can be read out directly to check whether dynamic
    questions get high g and static questions low g.

    Warm-start gate init (E-10): the gate-MLP final-layer bias is initialised
    to a large positive value (``gate_init_bias``, default +4.0 → sigmoid ≈
    0.982) so g starts ≈ 1 and the E-10 starting point matches E-03a
    (full-strength MoPE residual).  The gate-MLP weights are zero-initialised
    so g is initially a constant determined purely by the bias.  This is the
    correct warm-start semantics: E-03a's out_proj is already trained (non-
    zero), so g must start ≈ 1, NOT inherit out_proj's zero-init.

    When ``use_gate=False`` (default) this module is byte-for-byte identical to
    the original E-02c/E-03a projector: no gate submodule is created and the
    forward returns ``image_embeds + out`` exactly as before.

    Args:
        mope_dim:       MoPE feature dimension (default: 768, ViT-B).
        llm_dim:        LLM hidden dimension (default: 3584, Qwen3-VL-4B).
        use_gate:       Enable the E-10 content-driven gate (default: False).
        gate_mode:      "learned" (default) — g = sigmoid(MLP(cond_text)).
                        # PENDING[D-15]: "oracle_task" (真值静/动标签硬门控 =
                        # E-10-oracle 路由上界) 留第二步实现, 本任务只做 learned 主路径.
        gate_hidden:    Hidden width of the gate MLP (default: 256).
        gate_init_bias: Final-layer bias of the gate MLP (default: +4.0 →
                        g ≈ 0.982). Use +2.2 for g ≈ 0.9.

    Forward:
        mope_features: [B, N_mope, mope_dim]  — MoPE encoder output (N_mope=784)
        image_embeds:  [B, N_img,  llm_dim]   — one image-embed shard
        cond_text:     [B, llm_dim] or None   — pooled text repr for the gate
        returns:       [B, N_img,  llm_dim]   — residual-updated image_embeds
    """

    def __init__(
        self,
        mope_dim: int = 768,
        llm_dim: int = 3584,
        use_gate: bool = False,
        gate_mode: str = "learned",
        gate_hidden: int = 256,
        gate_init_bias: float = 4.0,
    ):
        super().__init__()
        self.mope_dim = mope_dim
        self.llm_dim = llm_dim
        self.use_gate = use_gate
        self.gate_mode = gate_mode
        self.gate_init_bias = gate_init_bias

        self.norm = nn.LayerNorm(mope_dim)
        self.k_proj = nn.Linear(mope_dim, llm_dim)
        self.v_proj = nn.Linear(mope_dim, llm_dim)

        # Zero-init: out_proj weight and bias both zero so MoPE contribution
        # is strictly zero at training start, preserving GUIDE's geometric
        # priors as the baseline starting point.
        self.out_proj = nn.Linear(llm_dim, llm_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # ---------------------------------------------------------------
        # E-10 content-driven gate (only when use_gate=True).
        # Input  = pooled text repr cond_text [B, llm_dim]
        # Output = scalar logit per sample -> sigmoid -> g in (0,1)
        # The gate submodule lives INSIDE the projector so that it is
        # discovered by model.model._mope_projector.parameters() and is
        # automatically managed by --freeze_mope_projector in train_space.py
        # (no separate optimizer/parameter group needed; trainable in both
        # 2-stage stages whenever the projector is trainable).
        # ---------------------------------------------------------------
        # E-10 gate-value process logging buffer (GateStatsCallback reads/clears
        # this). A plain Python list of detached fp32 1-D tensors, appended once
        # per forward ONLY when use_gate and self.training (eval / use_gate=False
        # never touch it). It is NOT a registered buffer/parameter, so it is
        # invisible to state_dict / optimizer / FSDP and adds zero overhead on
        # the disabled path. B=1 per-sample calls each append their own slice;
        # the callback concatenates the window and clears it (all ranks clear
        # their own buf to avoid unbounded growth).
        self._gate_buf = []
        if self.use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(llm_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            # Warm-start init with NO zero-gradient deadzone (cf. the QFormer
            # queries zero-init bug, status M-1):
            #
            #   g = sigmoid( W2 · GELU(W1·x + b1) + b2 )
            #
            #   - b2 = gate_init_bias  (large positive) → g ≈ sigmoid(b2) ≈ 1 at
            #     init, so the E-10 starting point matches E-03a (full MoPE).
            #   - W2 (final-layer weight) = SMALL normal (std 1e-3), NOT zero:
            #     if W2 were zero, W1's weight grad would flow back through W2=0
            #     and be identically zero on every step → the gate could never
            #     become content-driven (it would stay a constant from b2 alone).
            #     A tiny non-zero W2 keeps g ≈ sigmoid(b2) at init (the small
            #     content term is negligible) while letting gradients reach W1.
            #   - W1 (hidden-layer weight) = small normal (std 0.02): breaks
            #     symmetry across hidden units and lets W2 receive non-zero grad
            #     (∝ hidden activations) from step 0.
            #   - b1 = 0.
            nn.init.normal_(self.gate_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.gate_mlp[0].bias)
            nn.init.normal_(self.gate_mlp[-1].weight, std=1e-3)
            nn.init.constant_(self.gate_mlp[-1].bias, gate_init_bias)

    def _compute_gate(
        self,
        cond_text: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gate g and return it shaped [B, 1, 1] for broadcasting.

        Args:
            cond_text: [B, llm_dim] pooled text repr, or None.  When None
                       (no usable text condition) we fall back to g = 1, i.e.
                       behave like ungated E-03a (safe fallback).
            ref:       reference tensor used to derive batch size / device /
                       dtype for the fallback path (the ``out`` tensor).

        Returns:
            g: Float tensor [B, 1, 1] broadcastable over out [B, N_img, llm_dim].
        """
        B = ref.shape[0]
        if cond_text is None:
            # Safe fallback: g = 1 -> identical to ungated E-03a residual.
            return torch.ones(B, 1, 1, device=ref.device, dtype=ref.dtype)
        # Gate MLP forward + sigmoid run in float32 for numerical stability,
        # then cast g back to the residual dtype. We run the whole MLP in
        # float32 (input + each linear's weight/bias cast to fp32 per-call) so
        # the matmuls, GELU and sigmoid are all fp32, not just the sigmoid.
        # The casts are functional (do NOT mutate self.gate_mlp), so the stored
        # gate_mlp dtype/weights and the warm-start init (g ≈ 0.98) are
        # unchanged; gradients flow back to the bf16 params as before.
        fc1, act, fc2 = self.gate_mlp[0], self.gate_mlp[1], self.gate_mlp[2]
        h = torch.nn.functional.linear(
            cond_text.float(), fc1.weight.float(), fc1.bias.float()
        )                                                                   # fp32
        h = act(h)                                                          # GELU fp32
        logit = torch.nn.functional.linear(
            h, fc2.weight.float(), fc2.bias.float()
        )                                                                   # [B, 1] fp32
        g = torch.sigmoid(logit)                                            # [B, 1] fp32
        g = g.view(B, 1, 1).to(ref.dtype)                                   # [B, 1, 1]
        return g

    def forward(
        self,
        mope_features: torch.Tensor,
        image_embeds: torch.Tensor,
        cond_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply single-head cross-attention and residual-add to image_embeds.

        Args:
            mope_features: Float tensor [B, N_mope, mope_dim].
            image_embeds:  Float tensor [B, N_img, llm_dim].
            cond_text:     Float tensor [B, llm_dim] pooled text repr for the
                           E-10 gate, or None.  Ignored when use_gate=False.

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
        out = out.to(image_embeds.dtype)

        if self.use_gate:
            # E-10: modulate the MoPE residual by a content-driven scalar gate.
            g = self._compute_gate(cond_text, out)                     # [B, 1, 1]
            # Stash g for the process-logging callback. Only accumulate during
            # training (eval never grows the buffer). detach()+float()+reshape
            # to a flat fp32 1-D tensor keeps it cheap and dtype/device-agnostic;
            # B=1 per-sample calls each append their own slice.
            if self.training:
                self._gate_buf.append(g.detach().float().reshape(-1))
            out = g * out

        return image_embeds + out


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

        self.queries = nn.Parameter(torch.empty(1, num_queries, llm_dim))
        nn.init.normal_(self.queries, std=0.02)

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
