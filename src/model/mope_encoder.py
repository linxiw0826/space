# MoPE encoder wrapper - frozen ViT-B for dynamic video features
# Interface: MoPEEncoder(checkpoint_path).forward(frames) -> [B, N_patches, 768]
#
# Architecture: VideoMAEv2-based ViT-B with MoE blocks in top 1/3 of layers.
# Model name: 'pretrain_mope_jepa_base_patch16_224'
# All parameters are frozen after loading. Only MoPEProjector is trainable.

from typing import Optional

import torch
import torch.nn as nn
import sys
import os

# Add MoPE codebase to sys.path at import time so that 'models' and 'dataset'
# packages inside the MoPE repo can be imported without modification.
_MOPE_ROOT = os.path.join(os.path.dirname(__file__), '../vendor/mope')
_MOPE_ROOT = os.path.abspath(_MOPE_ROOT)
if _MOPE_ROOT not in sys.path:
    sys.path.insert(0, _MOPE_ROOT)


class MoPEEncoder(nn.Module):
    """
    Frozen wrapper around the MoPE ViT-B encoder (VideoMAEv2 + MoE routing).

    After construction, ALL parameters are permanently frozen (requires_grad=False).
    This module is used in eval mode for feature extraction only.

    Args:
        checkpoint_path: Path to the MoPE .pth checkpoint file.
                         If None, only the model architecture is built (weights
                         will be loaded from the E-02a checkpoint externally).
        all_frames: Number of video frames the model expects (default: 8).
                    Must match T dimension of input tensors.

    MoPE model config (ViT-B/16 base):
        num_routable_experts = 17  (one per physics class)
        num_shared_experts   = 4
        top_k                = 5
        tubelet_size         = 2
        embed_dim            = 768
    """

    def __init__(self, checkpoint_path: Optional[str] = None, all_frames: int = 8):
        super().__init__()

        # Import here so that the sys.path insertion above takes effect first.
        from timm.models import create_model
        import models  # noqa: F401 - registers MoPE model variants with timm

        # Build the full pretrain model (encoder + JEPA predictor).
        # We only use the .encoder sub-module at inference.
        pretrain_model = create_model(
            'pretrain_mope_jepa_base_patch16_224',
            pretrained=False,
            drop_path_rate=0.0,
            all_frames=all_frames,
            tubelet_size=2,
            with_cp=False,
            num_routable_experts=17,
            num_shared_experts=4,
            top_k=5,
        )

        if checkpoint_path is not None:
            print(f"[MoPEEncoder] Loading pretrain_mope_jepa_base_patch16_224 "
                  f"(all_frames={all_frames}) from {checkpoint_path}")

            # Load checkpoint.
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = None
            for key in ['model', 'module', 'state_dict']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    print(f"[MoPEEncoder] Loaded state_dict via key='{key}'")
                    break
            if state_dict is None:
                state_dict = ckpt
                print("[MoPEEncoder] Loaded state_dict directly")

            # Strip _orig_mod. prefix from torch.compile artefacts.
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            # Exclude predictor weights: predictor_pos_embed size depends on the
            # all_frames the checkpoint was trained with, which may differ from the
            # all_frames used here.  We only need the encoder sub-module anyway.
            encoder_state_dict = {k: v for k, v in state_dict.items()
                                  if not k.startswith('predictor.')}
            skipped = len(state_dict) - len(encoder_state_dict)
            if skipped:
                print(f"[MoPEEncoder] Skipped {skipped} predictor.* keys (not used at inference)")

            msg = pretrain_model.load_state_dict(encoder_state_dict, strict=False)
            print(f"[MoPEEncoder] Missing keys: {len(msg.missing_keys)}, "
                  f"Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"[MoPEEncoder] Missing (first 5): {msg.missing_keys[:5]}")
        else:
            print("[MoPEEncoder] No checkpoint provided — architecture only (weights to be loaded from E-02a ckpt)")

        # Keep only the encoder sub-module.
        self.encoder = pretrain_model.encoder
        self.encoder.eval()

        # Freeze ALL encoder parameters permanently.
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Store for reference (used to build the full visible mask at forward time).
        self.num_patches = self.encoder.patch_embed.num_patches
        print(f"[MoPEEncoder] Frozen encoder loaded. num_patches={self.num_patches}, "
              f"embed_dim={self.encoder.embed_dim}")

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from a batch of video clips.

        Args:
            frames: Float tensor of shape [B, C, T, H, W] where
                    C=3 (RGB), T=all_frames, H=W=224.
                    Should be normalized with ImageNet mean/std.

        Returns:
            x_vis: Float tensor of shape [B, N_vis, 768] where N_vis equals
                   num_patches when mask is all-False (fully visible).
                   With all_frames=8, tubelet_size=2, patch_size=16, 224x224:
                   N_vis = (8/2) * (224/16)^2 = 4 * 196 = 784.
        """
        B = frames.shape[0]
        device = frames.device

        # Full-visible mask: all False => no tokens masked out.
        mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)

        x_vis = self.encoder.forward_features(
            frames, mask,
            physics_label=None,
            physics_label_soft=None,
        )  # [B, N_vis, 768]

        return x_vis


class VideoMAEEncoder(nn.Module):
    """Frozen plain-VideoMAE ViT-B/16 encoder — a drop-in for MoPEEncoder.

    Paper-2 D-26/D-27 encoder swap: while the physics MoPE encoder is not
    finished training, a plain (no-MoE) VideoMAE ViT-B/16 tubelet-2 encoder is
    used as a frozen stand-in (experiments E-03a_vmae / E-16_vmae). It MIRRORS
    the ``MoPEEncoder`` interface exactly:

        __init__(checkpoint_path=None, all_frames=8)
        forward(frames [B,3,T,224,224]) -> x_vis [B, 784, 768]   (time-major)

    so the projector / feed cross-attn / LFP head are byte-for-byte unchanged —
    the encoder is the ONLY variable vs the MoPE arm.

    Architecture: the SAME ``PretrainVisionTransformerEncoder`` that backs MoPE,
    but with ``moe_layer_indices=[]`` so every one of the 12 blocks is a plain
    ViT block (no SharedExpertMoE). ``forward_features`` therefore returns the
    full ``[B, N_vis, 768]`` token sequence in the identical time-major layout
    (token j in bin j//196; N_vis = (T/2)*(224/16)^2 = 4*196 = 784 for T=8),
    guaranteed by the shared Conv3d patch-embed + flatten(t,h,w) ordering.

    Weight loading (training only, checkpoint_path set): the on-disk
    ``videomaev2_base.pth`` state_dict has keys ``backbone.*`` (+ an unused
    ``cls_head.*``). We strip the ``backbone.`` prefix, drop ``cls_head.*``, and
    rename ``fc_norm.*`` -> ``norm.*`` (VideoMAE's final classification LayerNorm
    becomes our per-token final norm, applied identically per 768-vector). This
    loads cleanly (0 missing / 0 unexpected) into the no-MoE ViT-B encoder.

    At eval, ``checkpoint_path=None`` (architecture only): the frozen encoder
    weights are restored from the fine-tuned checkpoint's ``_mope_encoder.*``
    keys by the eval wrapper, exactly like MoPEEncoder. The encoder type MUST
    therefore match between train and eval (env MOPE_ENCODER_TYPE=videomae), else
    the plain-MLP keys silently fail to load against MoE keys.

    ALL parameters are permanently frozen (requires_grad=False), eval mode.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, all_frames: int = 8):
        super().__init__()

        # Import here so the sys.path insertion at module top takes effect first.
        from functools import partial
        from models.modeling_pretrain import PretrainVisionTransformerEncoder  # noqa: E402

        # Build a no-MoE ViT-B/16 video encoder mirroring MoPE's encoder config
        # (qkv_bias=True, LayerNorm eps=1e-6, tubelet_size=2) EXCEPT MoE is
        # disabled via moe_layer_indices=[] so all 12 blocks are plain ViT blocks.
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            tubelet_size=2,
            all_frames=all_frames,
            moe_layer_indices=[],
        )

        if checkpoint_path is not None:
            print(f"[VideoMAEEncoder] Loading plain VideoMAE ViT-B/16 "
                  f"(all_frames={all_frames}) from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = None
            for key in ['model', 'module', 'state_dict']:
                if isinstance(ckpt, dict) and key in ckpt:
                    state_dict = ckpt[key]
                    print(f"[VideoMAEEncoder] Loaded state_dict via key='{key}'")
                    break
            if state_dict is None:
                state_dict = ckpt
                print("[VideoMAEEncoder] Loaded state_dict directly")

            # Strip torch.compile artefact prefix if present.
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            # Map mmaction2 VideoMAE keys -> our encoder keys:
            #   strip 'backbone.'  /  drop 'cls_head.*'  /  'fc_norm.*' -> 'norm.*'
            mapped = {}
            dropped_clshead = 0
            for k, v in state_dict.items():
                if k.startswith('cls_head.'):
                    dropped_clshead += 1
                    continue
                k2 = k[len('backbone.'):] if k.startswith('backbone.') else k
                if k2.startswith('fc_norm.'):
                    k2 = 'norm.' + k2[len('fc_norm.'):]
                mapped[k2] = v
            if dropped_clshead:
                print(f"[VideoMAEEncoder] Dropped {dropped_clshead} cls_head.* keys "
                      f"(classification head, not used for feature extraction)")

            msg = self.encoder.load_state_dict(mapped, strict=False)
            print(f"[VideoMAEEncoder] Missing keys: {len(msg.missing_keys)}, "
                  f"Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"[VideoMAEEncoder] Missing (first 5): {msg.missing_keys[:5]}")
            if msg.unexpected_keys:
                print(f"[VideoMAEEncoder] Unexpected (first 5): {msg.unexpected_keys[:5]}")
        else:
            print("[VideoMAEEncoder] No checkpoint provided — architecture only "
                  "(weights to be restored from the fine-tuned _mope_encoder.* keys)")

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.num_patches = self.encoder.patch_embed.num_patches
        print(f"[VideoMAEEncoder] Frozen encoder loaded. num_patches={self.num_patches}, "
              f"embed_dim={self.encoder.embed_dim}")

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features. Same contract as MoPEEncoder.forward.

        Args:
            frames: Float tensor [B, C=3, T, H=224, W=224], ImageNet-normalized.

        Returns:
            x_vis: Float tensor [B, N_vis, 768], time-major (token j in bin
                   j//196). N_vis = 784 for T=8 (tubelet_size=2, ViT-B/16).
        """
        B = frames.shape[0]
        device = frames.device

        mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)

        x_vis = self.encoder.forward_features(
            frames, mask,
            physics_label=None,
            physics_label_soft=None,
        )  # [B, N_vis, 768]

        # R2 (scoper): assert the token layout is [n_bins x 196] time-major, so
        # the LFP head (//196) and the E-16t temporal PE bin split stay aligned.
        assert x_vis.dim() == 3 and x_vis.shape[0] == B and x_vis.shape[-1] == 768, (
            f"[VideoMAEEncoder] unexpected output shape {tuple(x_vis.shape)}; "
            f"expected [B, N_vis, 768]."
        )
        assert x_vis.shape[1] % 196 == 0, (
            f"[VideoMAEEncoder] N_vis={x_vis.shape[1]} not divisible by 196 "
            f"(spatial tokens/bin); token order is not [n_bins x 196] time-major "
            f"— LFP / temporal-PE bins would be misaligned."
        )

        return x_vis
