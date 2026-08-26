import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class NativeRouter(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, score_func="sigmoid",
                 bias_update_speed=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.bias_update_speed = float(bias_update_speed)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, x):
        logits = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=-1)

        k = max(1, min(int(self.top_k), self.num_experts))
        select_scores = scores + self.expert_bias.view(1, 1, -1).to(scores.dtype)
        _, expert_id = torch.topk(select_scores, k=k, dim=-1)
        expert_weight = scores.gather(-1, expert_id)
        expert_weight = expert_weight / expert_weight.sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        return scores, expert_id, expert_weight

    @torch.no_grad()
    def update_bias(self, select_frac):
        if self.bias_update_speed <= 0:
            return
        select_frac = select_frac.to(self.expert_bias.device, dtype=torch.float32)
        target = float(self.top_k) / float(self.num_experts)
        delta = torch.sign(torch.full_like(select_frac, target) - select_frac)
        self.expert_bias.add_(self.bias_update_speed * delta)
        self.expert_bias.sub_(self.expert_bias.mean())


class NativeMoEFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, num_experts=4, top_k=2,
                 num_shared_experts=1, drop=0.0, router_score_func="sigmoid",
                 router_bias_update_speed=0.001):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = NativeRouter(
            dim, num_experts=num_experts, top_k=top_k,
            score_func=router_score_func,
            bias_update_speed=router_bias_update_speed)
        self.routed_experts = nn.ModuleList(
            [ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList(
            [ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_shared_experts)])
        self.shared_weight = nn.Parameter(
            torch.ones(num_shared_experts) / max(1, num_shared_experts))
        self.last_route_stats = None

    def _dispatch(self, x_flat, expert_id_flat, weight_flat):
        out = torch.zeros_like(x_flat)
        active = torch.ones_like(expert_id_flat, dtype=torch.bool)
        for expert_idx, expert in enumerate(self.routed_experts):
            sel = active & (expert_id_flat == expert_idx)
            idx = sel.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                y_dummy = expert(x_flat[:1]) * 0.0
                out = out.index_add(
                    0, torch.zeros(1, dtype=torch.long, device=x_flat.device),
                    y_dummy.to(out.dtype))
                continue
            y = expert(x_flat.index_select(0, idx))
            y = y * weight_flat.index_select(0, idx).unsqueeze(-1).to(y.dtype)
            out = out.index_add(0, idx, y.to(out.dtype))
        return out

    def forward(self, x, record_stats=True):
        bsz, n_tokens, dim = x.shape
        x_flat = x.reshape(bsz * n_tokens, dim)

        shared = None
        shared_w = F.softmax(self.shared_weight, dim=0)
        for wi, expert in zip(shared_w, self.shared_experts):
            y = wi * expert(x_flat)
            shared = y if shared is None else shared + y

        scores, expert_id, expert_weight = self.router(x)
        route_k = expert_id.shape[-1]
        routed_in = x_flat.repeat_interleave(route_k, dim=0)
        routed = self._dispatch(
            routed_in, expert_id.reshape(-1), expert_weight.reshape(-1))
        routed = routed.reshape(bsz * n_tokens, route_k, dim).sum(dim=1)

        if record_stats:
            with torch.no_grad():
                selected = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
                selected.scatter_add_(
                    0, expert_id.reshape(-1),
                    torch.ones(bsz * n_tokens * route_k, device=x.device, dtype=x.dtype))
                select_frac = selected / (bsz * n_tokens + 1e-6)

                top1 = scores.argmax(dim=-1)
                top1_frac = torch.zeros_like(selected)
                top1_frac.scatter_add_(
                    0, top1.reshape(-1),
                    torch.ones(bsz * n_tokens, device=x.device, dtype=x.dtype))
                top1_frac = top1_frac / (bsz * n_tokens + 1e-6)

                weight_sum = torch.zeros_like(selected)
                weight_sum.scatter_add_(0, expert_id.reshape(-1), expert_weight.reshape(-1))
                weight_frac = weight_sum / (bsz * n_tokens + 1e-6)
                avg_selected_weight = weight_sum / selected.clamp_min(1e-6)
                probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1).mean()
                self.last_route_stats = {
                    "select_frac": select_frac.detach(),
                    "top1_frac": top1_frac.detach(),
                    "weight_frac": weight_frac.detach(),
                    "avg_selected_weight": avg_selected_weight.detach(),
                    "router_entropy": entropy.detach(),
                    "expert_bias": self.router.expert_bias.detach().clone(),
                }

        return (shared + routed.to(shared.dtype)).reshape(bsz, n_tokens, dim)
