# LFP (Latent Future Prediction) auxiliary head for paper-2 Stage 1.
#
# Implements the auxiliary prediction-head paradigm (Cambrian-P / Cambrian-S
# LFP template): a small 2-layer MLP that maps a single LLM hidden-state
# vector to the MoPE motion-latent space, trained with MSE + cosine loss
# alongside the NTP objective. The head is TRAINING-ONLY: at inference the
# outer LFP forward-patch is skipped (no labels) and the head is never
# called, so the eval path is byte-for-byte identical to E-03a.
#
# Reference: state/analyses/20260619_stage1_lfp_head_integration.md (S1)
#
# Design notes
# ------------
# - 2-layer MLP: Linear(llm_dim -> hidden) -> GELU -> Linear(hidden -> target_dim).
#   Matches the Cambrian-S LFP head shape (2-layer MLP + GELU). The default
#   hidden=2048 / target_dim=768 (= MoPE embed_dim) come from the Stage 1
#   spec; both are configurable via MoPEArguments.
# - Decoupled input projection (risk-6, spec Q6/S5): when the feed-features
#   path is OFF (mope_feed_features=False, the "prediction-only" E-15 arm)
#   the cross-attn projector's k/v/out_proj are no longer exercised by the
#   feed path. To keep the three E-03a / E-15 / E-16 arms strictly orthogonal
#   the LFP head has its OWN small input projection (a single Linear), so the
#   prediction target/pred never depends on the feed-features projector. When
#   feed-features is ON the pred source is the LLM hidden state directly, so
#   the input projection is still applied (it just maps hidden -> a stable
#   head input space); this keeps one code path for both arms.
# - The head lives on ``model.model`` as ``_mope_lfp_head`` (mirroring
#   ``_mope_projector``) so it is picked up by ``.to(device)`` /
#   ``.parameters()`` / FSDP / state_dict automatically. It is trainable
#   whenever the projector is (managed jointly with --freeze_mope_projector
#   so both 2-stage stages train it).
#
# implements D-12 (论文2 辅助信号 = MoPE latent future-feature prediction;
# Stage 1 activation per decisions.md 2026-06-19). NOT a PENDING marker: D-12
# is activated as the paper-2 main method by the 2026-06-19 user decision,
# and the distillation-style target (time-split of the current 8 frames,
# not a separate future-frame data pipeline) is the locked Stage 1 choice
# (paper2_design.md §2.3, decisions.md 2026-06-20).

import torch
import torch.nn as nn


class MoPELFPHead(nn.Module):
    """Auxiliary Latent Future Prediction head (paper-2 Stage 1).

    A 2-layer MLP that maps one LLM hidden-state vector ``h [B, llm_dim]``
    to a prediction of the (mean-pooled) frozen-MoPE future-frame latent
    ``target [B, target_dim]``. Trained with MSE + (1 - cosine) alongside NTP.

    The head carries its own decoupled input projection so the "prediction
    only" arm (mope_feed_features=False) is strictly orthogonal to the
    feed-features projector (risk-6).

    Args:
        llm_dim:     LLM hidden dim of the pred-source hidden state (e.g. 3584).
        hidden:      MLP hidden width (default 2048, Cambrian-S scale).
        target_dim:  MoPE latent dim to predict (default 768 = embed_dim).

    Forward:
        h:        Float tensor [B, llm_dim]  (LLM hidden state at pred-source
                  position, typically the last non -100 label token).
        returns:  Float tensor [B, target_dim] prediction of the MoPE latent.
    """

    def __init__(self, llm_dim: int, hidden: int = 2048, target_dim: int = 768):
        super().__init__()
        self.llm_dim = llm_dim
        self.hidden = hidden
        self.target_dim = target_dim

        # Decoupled input projection: maps the raw LLM hidden state into a
        # stable head-input space. This is what keeps the prediction-only arm
        # (no feed features) orthogonal to the cross-attn projector: the head
        # reads the LLM hidden state through its OWN projection, never the
        # feed-features projector output.
        self.input_proj = nn.Linear(llm_dim, hidden)

        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """[B, llm_dim] -> [B, target_dim]."""
        x = self.input_proj(h)   # [B, hidden]
        x = self.net(x)          # [B, target_dim]
        return x
