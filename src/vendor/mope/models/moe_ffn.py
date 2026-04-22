# --------------------------------------------------------
# models/moe_ffn.py  ── MoPE 物理专家 FFN
#
# 对齐 WISA 17 类物理标签：
#   Dynamics      (0-5):  Collision, RigidBodyMotion, ElasticMotion,
#                          LiquidMotion, GasMotion, Deformation
#   Thermodynamics(6-11): Melting, Solidification, Vaporization,
#                          Liquefaction, Explosion, Combustion
#   Optics        (12-16): Reflection, Refraction, Scattering,
#                           InterferenceAndDiffraction, UnnaturalLightSources
#
# Shared expert 4个（始终激活）
# Routable expert 17个（video-level 路由，top_k=1）
# Router 可用软标签分布（JSON label_distribution）做 soft cross-entropy 监督
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

LABEL_TO_EXPERT: Dict[int, int] = {i: i for i in range(17)}
NUM_PHYSICS_CLASSES = 17

# 与 Qwen 打分 JSON 中 label_distribution 的 key 顺序一致（对应 expert 0..16）
PHYSICS_SOFT_LABEL_KEYS = (
    "collision",
    "rigid_body_motion",
    "elastic_motion",
    "liquid_motion",
    "gas_motion",
    "deformation",
    "melting",
    "solidification",
    "vaporization",
    "liquefaction",
    "explosion",
    "combustion",
    "reflection",
    "refraction",
    "scattering",
    "interference_diffraction",
    "unnatural_light_sources",
)


def distribution_dict_to_tensor(
    d: Dict,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """将 JSON 的 label_distribution 转为长度 17 的向量。"""
    vec = [float(d.get(k, 0.0)) for k in PHYSICS_SOFT_LABEL_KEYS]
    t = torch.tensor(vec, device=device, dtype=dtype)
    s = t.sum()
    if s > 0:
        t = t / s
    return t


# ── 1. 单个 Expert FFN ────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ── 2. 物理感知 Router ────────────────────────────────────────────────────────

class PhysicsRouter(nn.Module):
    """Video-level 路由器：输入为 time-aware summary"""

    def __init__(self, dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        # x_video = concat(g_app, g_key) -> [B, 2C]
        self.gate        = nn.Linear(dim * 2, num_experts, bias=False)

    def forward(self, x_video: torch.Tensor):
        # x_video: [B, 2C]（g_app 与 g_key 拼接）
        logits = self.gate(x_video)                                      # [B, E]
        scores = F.softmax(logits, dim=-1)                               # [B, E]
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)     # [B, top_k]
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
        return scores, topk_scores, topk_indices


# ── 3. Expert Fusion Network ──────────────────────────────────────────────────

class ExpertFusionNetwork(nn.Module):
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim + num_experts, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, aggregated: torch.Tensor,
                dispatch_weights: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([aggregated, dispatch_weights], dim=-1)
        return self.proj(fused) + aggregated


# ── 4. 纯稀疏 MoE ────────────────────────────────────────────────────────────

class PhysicsSparseMoE(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 num_experts: int = 17, top_k: int = 1, drop: float = 0.0,
                 key_top_m: int = 3, key_alpha: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.key_top_m   = key_top_m
        self.key_alpha   = key_alpha
        hidden_dim       = int(dim * mlp_ratio)

        self.router  = PhysicsRouter(dim, num_experts, top_k)
        self.token_router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_experts)
        ])
        self.fusion  = ExpertFusionNetwork(dim, num_experts)

    def _build_time_aware_summary(
            self, x: torch.Tensor,
            time_ids: Optional[torch.Tensor],
            num_time_bins: Optional[int] = None) -> torch.Tensor:
        """
        x: [B, N_vis, C]
        time_ids: [B, N_vis], visible token 对应时间片 id
        return: x_video [B, 2C] = concat(g_app, g_key)
        """
        B, N, C = x.shape
        g_app_fallback = x.mean(dim=1)  # [B, C]

        if time_ids is None:
            # 兼容旧调用链：退化为原始 mean summary
            return torch.cat([g_app_fallback, g_app_fallback], dim=-1)

        if time_ids.shape != (B, N):
            raise ValueError(
                f"time_ids shape must be [B, N_vis], got {tuple(time_ids.shape)}"
            )
        time_ids = time_ids.long()
        # 固定 T，避免 time_ids.max().item() 在 compile/checkpoint 下 graph break，
        # 且与 encoder 的 tubelet 时间维一致。
        if num_time_bins is not None:
            T = int(num_time_bins)
        else:
            T = int(time_ids.max().item()) + 1

        # 使用非原位版 scatter_add（而非 scatter_add_），避免对含梯度的 x 做
        # in-place 写入，防止 gradient checkpoint 重算时 autograd 图节点
        # 序号偏移导致的 CheckpointError。
        idx_C = time_ids.unsqueeze(-1).expand(-1, -1, C)   # [B, N, C]
        token_sum = torch.zeros(
            B, T, C, device=x.device, dtype=x.dtype).scatter_add(1, idx_C, x)

        idx_1 = time_ids.unsqueeze(-1)                     # [B, N, 1]
        ones  = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        token_cnt = torch.zeros(
            B, T, 1, device=x.device, dtype=x.dtype).scatter_add(1, idx_1, ones)

        h_t = token_sum / (token_cnt + 1e-6)  # [B, T, C]
        valid = token_cnt.squeeze(-1) > 0     # [B, T]
        valid_f = valid.to(dtype=x.dtype).unsqueeze(-1)

        g_app = (h_t * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        s_global = (h_t - g_app.unsqueeze(1)).abs().sum(dim=-1)  # [B, T]

        h_prev      = torch.roll(h_t,    shifts=1, dims=1)
        valid_prev_raw = torch.roll(valid, shifts=1, dims=1)
        # 用 concat 替代 valid_prev[:, 0] = False（in-place 赋值），
        # 避免 checkpoint 重算路径与正向路径的 autograd 图不一致。
        false_col  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        valid_prev = torch.cat([false_col, valid_prev_raw[:, 1:]], dim=1)

        s_diff = (h_t - h_prev).abs().sum(dim=-1)
        s_diff = s_diff * (valid & valid_prev).to(dtype=x.dtype)

        s = self.key_alpha * s_global + (1.0 - self.key_alpha) * s_diff
        s = s.masked_fill(~valid, -1e9)

        top_m = max(1, min(self.key_top_m, T))
        top_scores, top_idx = s.topk(top_m, dim=1)  # [B, m]
        top_w = F.softmax(top_scores, dim=-1)       # [B, m]

        h_top = h_t.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, C))
        g_key = (top_w.unsqueeze(-1) * h_top).sum(dim=1)  # [B, C]

        return torch.cat([g_app, g_key], dim=-1)  # [B, 2C]

    def forward(self, x: torch.Tensor,
                physics_label: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None,
                physics_label_soft: Optional[torch.Tensor] = None,
                num_time_bins: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        # video-level 路由：time-aware summary → gate 打分
        x_video = self._build_time_aware_summary(
            x, time_ids, num_time_bins=num_time_bins)            # [B, 2C]
        scores, topk_scores, topk_indices = self.router(x_video)
        # scores: [B, 17], topk_indices: [B, 1]

        # ── video router → top-k 候选集合 ──────────────────────────────
        # top_k 建议改成3，给 token router 候选空间
        candidate_mask = torch.zeros(B, self.num_experts,
                                     dtype=torch.bool, device=x.device)
        candidate_mask.scatter_(1, topk_indices, True)  # [B, E]

        # ── token router：在候选内每个 token 独立选专家（无监督）──────
        token_logits = self.token_router(x)                          # [B, N, E]
        mask_exp = candidate_mask.unsqueeze(1).expand_as(token_logits)
        token_logits = token_logits.masked_fill(~mask_exp, float('-inf'))
        token_scores = F.softmax(token_logits, dim=-1)               # [B, N, E]
        token_expert = token_scores.argmax(dim=-1)                   # [B, N]

        # ── 构造 dispatch_weights（token-level）────────────────────────
        token_expert_flat = token_expert.reshape(B * N)              # [B*N]
        token_scores_flat = token_scores.reshape(B * N, self.num_experts)
        dispatch_weights = torch.zeros(B * N, self.num_experts,
                                       device=x.device, dtype=x.dtype)
        dispatch_weights.scatter_(
            1,
            token_expert_flat.unsqueeze(1),
            token_scores_flat.gather(1, token_expert_flat.unsqueeze(1))
        )
        # dispatch_weights: [B*N, E]，每行只有一个非零值

        # router 监督：soft cross-entropy  L = -sum_i y_i log p_i
        # physics_label_soft: [B, E]；若未提供则用 one-hot(physics_label) 退化
        router_loss = x.new_zeros(())
        if physics_label_soft is not None:
            y = physics_label_soft.to(dtype=scores.dtype, device=scores.device)
            y = y.clamp_min(0.0)
            y = y / (y.sum(dim=-1, keepdim=True) + 1e-8)
            p = scores.clamp_min(1e-8)
            router_loss = -(y * p.log()).sum(dim=-1).mean()
        elif physics_label is not None:
            y = F.one_hot(
                physics_label, num_classes=self.num_experts).to(dtype=scores.dtype)
            p = scores.clamp_min(1e-8)
            router_loss = -(y * p.log()).sum(dim=-1).mean()

        # balance loss：鼓励各专家负载均衡
        # balance_loss：改用 token-level 负载统计
        E = self.num_experts
        expert_counts = torch.zeros(E, device=x.device, dtype=x.dtype)
        expert_counts.scatter_add_(
            0, token_expert_flat,
            torch.ones(B * N, device=x.device, dtype=x.dtype))
        f_e = expert_counts / (B * N + 1e-6)
        p_e = scores.mean(0)
        balance_loss = E * (f_e * p_e).sum()

        # 专家计算
        expert_out = torch.stack(
            [exp(x_flat) for exp in self.experts], dim=0)                # [E, B*N, C]
        weights    = dispatch_weights.T.unsqueeze(-1)                    # [E, B*N, 1]
        aggregated = (expert_out * weights).sum(0)                       # [B*N, C]

        out = self.fusion(aggregated, dispatch_weights)
        return out.reshape(B, N, C), {
            "balance_loss":  balance_loss,
            "router_loss":   router_loss,
            "router_scores": scores.detach(),
            "expert_load":   dispatch_weights.detach().mean(0),
            "token_scores": token_scores,  # [B, N, 17]
        }


# ── 5. SharedExpertMoE（替换 Block.self.mlp）────────────────────────────────

class SharedExpertMoE(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 num_routable_experts: int = 17, num_shared_experts: int = 4,
                 top_k: int = 1, drop: float = 0.0,
                 key_top_m: int = 3, key_alpha: float = 0.5):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.shared_experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, drop=drop)
            for _ in range(num_shared_experts)
        ])
        self.shared_weight = nn.Parameter(
            torch.ones(num_shared_experts) / num_shared_experts)

        self.sparse_moe = PhysicsSparseMoE(
            dim=dim, mlp_ratio=mlp_ratio,
            num_experts=num_routable_experts, top_k=top_k, drop=drop,
            key_top_m=key_top_m, key_alpha=key_alpha)

    def forward(self, x: torch.Tensor,
                physics_label: Optional[torch.Tensor] = None,
                time_ids: Optional[torch.Tensor] = None,
                physics_label_soft: Optional[torch.Tensor] = None,
                num_time_bins: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
        w = F.softmax(self.shared_weight, dim=0)
        shared_out = torch.stack(
            [wi * exp(x) for wi, exp in zip(w, self.shared_experts)],
            dim=0).sum(dim=0)

        sparse_out, aux = self.sparse_moe(
            x, physics_label, time_ids, physics_label_soft,
            num_time_bins=num_time_bins)
        return shared_out + sparse_out, aux