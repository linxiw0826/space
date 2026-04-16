# --------------------------------------------------------
# models/modeling_pretrain.py  ── MoPE-JEPA 版本
# 删除 MAE decoder，替换为 JEPA predictor
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from .modeling_finetune import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
from .jepa_predictor import MoPEJEPAPredictor


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    # ── 完全不变，原样保留 ──────────────────────────────
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None,
                 tubelet_size=2, use_learnable_pos_emb=False,
                 with_cp=False, all_frames=16, cos_attn=False,
                 moe_layer_indices=None,
                 num_routable_experts=17, num_shared_experts=4, top_k=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        if moe_layer_indices is None:
            moe_layer_indices = list(range(depth * 2 // 3, depth))
        self.moe_layer_indices = moe_layer_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                cos_attn=cos_attn,
                use_moe=(i in moe_layer_indices),
                num_routable_experts=num_routable_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) \
            if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, mask, physics_label=None,
                         physics_label_soft=None):
        num_time_bins = x.shape[2] // self.patch_embed.tubelet_size
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)

        num_spatial = self.patch_embed.num_patches // num_time_bins
        full_time_ids = torch.arange(
            num_time_bins, device=x.device, dtype=torch.long
        ).repeat_interleave(num_spatial)
        full_time_ids = full_time_ids.unsqueeze(0).expand(B, -1)
        x_vis_time_ids = full_time_ids[~mask].reshape(B, -1)

        all_balance_loss = []
        all_router_loss  = []
        self._last_token_scores = None

        for blk in self.blocks:
            if self.with_cp:
                x_vis = cp.checkpoint(
                    lambda t, _blk=blk: _blk(
                        t, physics_label, x_vis_time_ids, physics_label_soft,
                        num_time_bins=num_time_bins),
                    x_vis, use_reentrant=False)
            else:
                x_vis = blk(x_vis, physics_label, x_vis_time_ids,
                            physics_label_soft, num_time_bins=num_time_bins)

            if getattr(blk, 'use_moe', False):
                aux = getattr(blk, '_moe_aux', {})
                if 'balance_loss' in aux:
                    all_balance_loss.append(aux['balance_loss'])
                if 'router_loss' in aux:
                    all_router_loss.append(aux['router_loss'])
                if 'token_scores' in aux:
                    self._last_token_scores = aux['token_scores']

        x_vis = self.norm(x_vis)
        self._balance_loss = (
            torch.stack(all_balance_loss).mean() if all_balance_loss else None)
        self._router_loss = (
            torch.stack(all_router_loss).mean() if all_router_loss else None)
        self._last_x_vis = x_vis
        return x_vis

    def forward(self, x, mask, physics_label=None, physics_label_soft=None):
        x = self.forward_features(x, mask, physics_label, physics_label_soft)
        x = self.head(x)
        return x


class PretrainVisionTransformer(nn.Module):
    """MoPE-JEPA：用 JEPA predictor 替换 MAE decoder"""

    def __init__(
        self,
        img_size=224, patch_size=16,
        encoder_in_chans=3, encoder_num_classes=0,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        # JEPA predictor 参数（替换原来的 decoder 参数）
        predictor_dim=384, predictor_depth=6, predictor_num_heads=6,
        mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, init_values=0.,
        use_learnable_pos_emb=False, tubelet_size=2,
        num_classes=0, in_chans=0,  # timm 兼容
        with_cp=False, all_frames=16, cos_attn=False,
        moe_layer_indices=None,
        num_routable_experts=17, num_shared_experts=4, top_k=5,
    ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size,
            in_chans=encoder_in_chans, num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp, all_frames=all_frames, cos_attn=cos_attn,
            moe_layer_indices=moe_layer_indices,
            num_routable_experts=num_routable_experts,
            num_shared_experts=num_shared_experts, top_k=top_k)

        num_patches = self.encoder.patch_embed.num_patches

        # JEPA predictor（替换 MAE decoder）
        self.predictor = MoPEJEPAPredictor(
            num_patches=num_patches,
            encoder_dim=encoder_embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, mask, physics_label=None, physics_label_soft=None):
        """
        x:    [B, 3, T, H, W]
        mask: [B, N_all]  bool，True=被mask

        return:
            pred:   [B, N_mask, encoder_dim]  ← predictor 预测
            target: [B, N_mask, encoder_dim]  ← target encoder 输出（detach）
        """
        B = x.shape[0]
        N_all = self.encoder.patch_embed.num_patches

        # ── context encoder：只处理 visible tokens ──────────────────────
        x_vis = self.encoder(x, mask, physics_label, physics_label_soft)
        # x_vis: [B, N_vis, encoder_dim]
        # 立刻保存 context path 的 aux loss，target path 会覆盖这些属性
        _balance_loss  = self.encoder._balance_loss
        _router_loss   = self.encoder._router_loss
        _token_scores  = self.encoder._last_token_scores
        _last_x_vis    = self.encoder._last_x_vis    # ← 新增

        # ── target encoder：处理全部 token，取 mask 位置，stop_gradient ──
        # 方案B：同一个 encoder，mask 全为 False（全部可见）
        full_mask = torch.zeros(B, N_all, dtype=torch.bool, device=x.device)
        with torch.no_grad():
            x_all = self.encoder(x, full_mask)  # [B, N_all, encoder_dim]
        # 取 mask 位置的 latent 作为预测目标
        target = x_all[mask].reshape(B, -1, x_all.shape[-1])  # [B, N_mask, encoder_dim]
        target = target.detach()
        # 恢复 context path 的 aux loss
        self.encoder._balance_loss      = _balance_loss
        self.encoder._router_loss       = _router_loss
        self.encoder._last_token_scores = _token_scores
        self.encoder._last_x_vis        = _last_x_vis  # ← 新增

        # ── predictor：预测 mask 位置的 latent ──────────────────────────
        pred = self.predictor(x_vis, mask)  # [B, N_mask, encoder_dim]

        return pred, target


@register_model
def pretrain_mope_jepa_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        encoder_num_classes=0,
        predictor_dim=384, predictor_depth=6, predictor_num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model