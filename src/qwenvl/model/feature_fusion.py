"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def get_2d_sincos_pos_embed(self, height: int, width: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.
        
        Args:
            height: Height of the grid
            width: Width of the grid  
            embed_dim: Embedding dimension
            device: Device to create tensor on
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        # Generate grid coordinates
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # [2, height, width]
        
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed
        
    def get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings from grid.
        
        Args:
            embed_dim: Embedding dimension
            grid: Grid coordinates of shape [2, height, width]
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [height*width, embed_dim//2]
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_h, emb_w], dim=1)  # [height*width, embed_dim]
        return emb
        
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        """
        Generate 1D sinusoidal position embeddings.
        
        Args:
            embed_dim: Embedding dimension
            pos: Position tensor of shape [height, width]
            
        Returns:
            emb: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # [embed_dim//2]
        
        pos = pos.flatten()
        out = torch.einsum('m,d->md', pos, omega)  # [height*width, embed_dim//2], outer product
        
        emb_sin = torch.sin(out)  # [height*width, embed_dim//2]
        emb_cos = torch.cos(out)  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # [height*width, embed_dim]
        return emb
        
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
        # Normalize features
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)
        
        # Add batch dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Generate 2D position embeddings
        pos_embed = self.get_2d_sincos_pos_embed(h_grid, w_grid, self.hidden_size, query.device).to(query.dtype)  # [h_grid*w_grid, hidden_size]

        # Add position embeddings to query and key
        # Assuming features are organized as [batch_size, h_grid*w_grid, hidden_size]
        query = query + pos_embed.unsqueeze(0)  # Broadcast across batch dimension
        key = key + pos_embed.unsqueeze(0)
            
        # Cross-attention: 2D features as query, 3D features as key/value
        attn_output, _ = self.cross_attention(query, key, value)
        
        if squeeze_output:
            attn_output = attn_output.squeeze(0)
            
        # First residual connection
        x = features_2d + attn_output
        
        # MLP with second residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x


class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.5))
            self.weight_3d = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features
        """

        _, h_grid, w_grid, _ = features_3d.shape
        if self.fusion_method == "add":
            return features_2d + features_3d
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            return self.projection(concat_features)
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            return x
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            return gate * features_2d + (1 - gate) * features_3d
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            return norm_weight_2d * features_2d + norm_weight_3d * features_3d
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp"):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x


class CameraGuidedModalityFusion(nn.Module):
    """
    Camera-Guided Modality Fusion (CGMF) module.
    
    Fuses visual features (fv) with spatial/3D features (fs) using camera latent (fc) as guidance.
    
    Inputs:
        fv: Visual features, shape (N, M_v, d_v)
        fs: Spatial features, shape (N, M_s, d_s)
        fc: Camera latent, shape (N, 1, d_s)
    
    Output:
        ffused: Fused features, shape (N, M_v, d_v)
    """
    
    def __init__(
        self,
        dv: int,
        ds: int,
        da: int,
        hidden_dim: int = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        enable_bias: bool = True,
        enable_importance: bool = True,
        enable_camera_memory: bool = True,
        enable_gate: bool = True,
    ):
        """
        Args:
            dv: Visual feature dimension (e.g., 3584 for Qwen2.5-VL)
            ds: Spatial feature dimension (e.g., 2048 for VGGT)
            da: Attention dimension (shared space for attention)
            hidden_dim: Hidden dimension for MLPs (default: da)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dv = dv
        self.ds = ds
        self.da = da
        self.hidden_dim = hidden_dim if hidden_dim is not None else da
        
        # 1. Projection Layers (Eq 5-7)
        self.ln_fv = nn.LayerNorm(dv)
        self.proj_q = nn.Linear(dv, da)  # PQ: Visual -> Attention space
        
        self.ln_fs = nn.LayerNorm(ds)
        self.proj_k = nn.Linear(ds, da)  # PK: Spatial -> Attention space
        self.proj_v = nn.Linear(ds, da)  # PV: Spatial -> Attention space
        
        self.proj_c = nn.Linear(ds, da)  # PC: Camera -> Attention space
        
        self.enable_bias = enable_bias
        self.enable_importance = enable_importance
        self.enable_camera_memory = enable_camera_memory
        self.enable_gate = enable_gate

        # 2. Camera-Conditioned Spatial Bias (Eq 8-9)
        # Input dim is ds (spatial) + ds (camera) = 2*ds
        if self.enable_bias:
            self.bias_mlp = nn.Sequential(
                nn.Linear(ds + ds, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, da),
            )
        
        # 3. Per-token Spatial Importance (Eq 11-12)
        if self.enable_importance:
            self.importance_mlp = nn.Sequential(
                nn.Linear(ds, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),  # Eq 11
            )
        
        # 4. Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=da,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Output Projection & SwiGLU Gate (Eq 16-20)
        if self.enable_gate:
            self.proj_out = nn.Linear(da, dv)  # PO: Attention space -> Visual space
            self.ln_out = nn.LayerNorm(dv)
            # SwiGLU components for Camera Gate
            self.gate_proj_1 = nn.Linear(da, dv)  # Pg,1
            self.gate_proj_2 = nn.Linear(da, dv)  # Pg,2
            self.final_proj = nn.Linear(dv, dv)  # PL
        else:
            self.proj_out = nn.Linear(da, dv)
            self.ln_out = nn.LayerNorm(dv)
    
    def forward(self, fv: torch.Tensor, fs: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CGMF.
        
        Args:
            fv: Visual features, shape (N, Mv, dv)
            fs: Spatial features, shape (N, Ms, ds)
            fc: Camera latent, shape (N, 1, ds)
        
        Returns:
            ffused: Fused features, shape (N, Mv, dv)
        """
        # --- Step 1: Project inputs to shared attention space (Eq 5-7) ---
        q = self.proj_q(self.ln_fv(fv))       # [N, Mv, da]
        k = self.proj_k(self.ln_fs(fs))       # [N, Ms, da]
        v = self.proj_v(self.ln_fs(fs))       # [N, Ms, da]
        c = self.proj_c(fc)                   # [N, 1, da]
        
        # --- Step 2: Camera-Conditioned Spatial Bias (Eq 8-10) ---
        if self.enable_bias:
            fc_expanded = fc.expand(-1, fs.shape[1], -1)  # [N, Ms, ds]
            cat_fs_fc = torch.cat([fs, fc_expanded], dim=-1)  # [N, Ms, 2*ds]
            bg = self.bias_mlp(cat_fs_fc)  # [N, Ms, da]
            k = k + bg
            v = v + bg
        
        # --- Step 3: Per-token Spatial Importance (Eq 11-13) ---
        if self.enable_importance:
            wt = self.importance_mlp(fs)  # [N, Ms, 1]
            v = v * wt  # Eq 13: Rescale V
        
        # --- Step 4: Construct Memory (Eq 14) ---
        if self.enable_camera_memory:
            k_prime = torch.cat([c, k], dim=1)  # [N, 1+Ms, da]
            v_prime = torch.cat([c, v], dim=1)  # [N, 1+Ms, da]
        else:
            k_prime = k
            v_prime = v
        
        # --- Step 5: Cross Attention (Eq 15) ---
        # Q attends to K', V'
        attn_out, _ = self.attn(query=q, key=k_prime, value=v_prime)  # [N, Mv, da]
        
        # --- Step 6: SwiGLU Gating & Residual (Eq 16-21) ---
        f_proj = self.ln_out(self.proj_out(attn_out))  # [N, Mv, dv]
        if self.enable_gate:
            c_squeezed = c.squeeze(1)  # [N, da]
            u = self.gate_proj_1(c_squeezed)  # [N, dv]
            v_gate = self.gate_proj_2(c_squeezed)  # [N, dv]
            swish_u = u * torch.sigmoid(u)
            g = swish_u * v_gate  # [N, dv]
            g_expanded = g.unsqueeze(1)
            ffused = (self.final_proj(f_proj) * g_expanded) + fv
        else:
            ffused = f_proj + fv
        
        return ffused


class GeometryMLPGate(nn.Module):
    """MLP-based residual gate for geometry features.

    Predicts a per-position gate delta from raw VGGT features:
        gate = tanh(MLP(raw_geo))         # range [-1, 1], initialized to 0
        output = (1 + gate) * geo_features

    At init the last linear is zero-initialized, so gate=tanh(0)=0 and
    the effective multiplier is 1 (identity).  The tanh bounds the
    multiplier to [0, 2], preventing feature explosion.
    """

    def __init__(self, geo_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.gate = nn.Sequential(
            nn.LayerNorm(geo_dim, eps=1e-6),
            nn.Linear(geo_dim, geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, 1),
            nn.Tanh(),
        )
        self._zero_init()

    def _zero_init(self):
        """Zero-init the last linear (before Tanh) so gate starts at 0."""
        last_linear = self.gate[-2]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, raw_geo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_geo_features: [N, h_patch, w_patch, geo_dim] raw VGGT features
        Returns:
            gate: [N, h_grid, w_grid, 1] spatially-merged gate signal
        """
        gate = self.gate(raw_geo_features)             # [N, h_patch, w_patch, 1]

        merge = self.spatial_merge_size
        N, h_patch, w_patch, _ = gate.shape
        h_grid = h_patch // merge
        w_grid = w_patch // merge
        gate = gate[:, :h_grid * merge, :w_grid * merge, :]
        gate = gate.reshape(N, h_grid, merge, w_grid, merge, 1)
        gate = gate.mean(dim=(2, 4))                   # [N, h_grid, w_grid, 1]

        return gate


class VGLLMStyleMLPGate(nn.Module):
    """MLP-based gate for geometry features, replicating VG-LLM's design.

    Uses RMSNorm + MLP + Sigmoid to produce a per-position gate in [0, 1]:
        gate = Sigmoid(MLP(RMSNorm(raw_geo)))
        output = gate * geo_features

    No zero-init — Sigmoid's default ~0.5 output lets the gate learn
    immediately, and the [0, 1] range only attenuates (never amplifies).
    """

    def __init__(self, geo_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        from .modeling_qwen3_vl import Qwen3VLTextRMSNorm

        self.gate = nn.Sequential(
            Qwen3VLTextRMSNorm(geo_dim, eps=1e-6),
            nn.Linear(geo_dim, geo_dim),
            nn.GELU(),
            nn.Linear(geo_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, raw_geo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_geo_features: [N, h_patch, w_patch, geo_dim] raw VGGT features
        Returns:
            gate: [N, h_grid, w_grid, 1] spatially-merged gate signal
        """
        gate = self.gate(raw_geo_features)  # [N, h_patch, w_patch, 1]

        merge = self.spatial_merge_size
        N, h_patch, w_patch, _ = gate.shape
        h_grid = h_patch // merge
        w_grid = w_patch // merge
        gate = gate[:, :h_grid * merge, :w_grid * merge, :]
        gate = gate.reshape(N, h_grid, merge, w_grid, merge, 1)
        gate = gate.mean(dim=(2, 4))  # [N, h_grid, w_grid, 1]

        return gate


class CameraAdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on camera token (FiLM / AdaLN).

    Applies LayerNorm to the input, then uses the global camera token to predict
    per-channel scale (γ) and shift (β), and modulates:
        x_norm = LayerNorm(x)
        [γ, β] = MLP(z_camera)
        out = (1 + γ) ⊙ x_norm + β

    The last linear is zero-initialized so that γ=0, β=0 at the start of
    training, giving out = x_norm (identity modulation), preserving pretrained
    feature distributions.
    """

    def __init__(self, hidden_size: int, camera_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.modulation = nn.Sequential(
            nn.Linear(camera_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self._zero_init_modulation()

    def _zero_init_modulation(self):
        """Zero-init the last linear so γ=0, β=0 → identity on first forward."""
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, camera_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: fused spatial features [N, h, w, D] or [N, L, D]
            camera_token: global camera feature [N, 1, camera_dim]
        Returns:
            modulated features, same shape as input x
        """
        orig_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() == 4 else x
        x_flat = self.norm(x_flat)

        params = self.modulation(camera_token.squeeze(1))       # [N, 2*D]
        gamma, beta = params.chunk(2, dim=-1)                   # each [N, D]

        out = (1 + gamma.unsqueeze(1)) * x_flat + beta.unsqueeze(1)

        return out.reshape(orig_shape)


class DualCameraAdaLN(nn.Module):
    """Camera AdaLN with true zero-initialization via residual formulation.

    Unlike CameraAdaLN whose base state is LN(x) (not x), this module uses:
        x_norm = LayerNorm(x)
        [γ, β] = MLP(z_camera)
        out = x + γ ⊙ x_norm + β

    At zero init (γ=0, β=0): out = x  ← true identity, no feature distortion.

    This is critical when the modulator is applied **before** fusion (adaLN_dual),
    where distorting features at init would hurt the pretrained model immediately.
    The (1+γ)/LN formulation in CameraAdaLN is fine post-fusion but not pre-fusion.
    """

    def __init__(self, hidden_size: int, camera_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.modulation = nn.Sequential(
            nn.Linear(camera_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self._zero_init_modulation()

    def _zero_init_modulation(self):
        """Zero-init the last linear → γ=0, β=0 → out = x at init (true identity)."""
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, camera_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: features [N, h, w, D] or [N, L, D]
            camera_token: global camera feature [N, 1, camera_dim]
        Returns:
            modulated features, same shape as input x
        """
        orig_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, x.shape[-1]) if x.dim() == 4 else x
        x_norm = self.norm(x_flat)                               # [N, L, D]

        params = self.modulation(camera_token.squeeze(1))        # [N, 2*D]
        gamma, beta = params.chunk(2, dim=-1)                    # each [N, D]

        # Residual: at zero init (γ=0, β=0) → out = x_flat (true identity)
        out = x_flat + gamma.unsqueeze(1) * x_norm + beta.unsqueeze(1)

        return out.reshape(orig_shape)


class CameraGeoAdaLN(nn.Module):
    """Spatially-varying AdaLN conditioned on raw geometry features + broadcast camera token.

    Uses raw (pre-merger) VGGT spatial features concatenated with the broadcast
    camera token to produce **per-position** scale (γ) and shift (β):

        geo_aligned  = avg_pool_2x2(raw_geo)           # [N, h, w, geo_dim]
        cam_spatial  = broadcast(camera_token)          # [N, h, w, cam_dim]
        condition    = concat(geo_aligned, cam_spatial) # [N, h, w, geo_dim+cam_dim]
        [γ, β]      = MLP(condition)                    # [N, h, w, D] each
        out          = (1 + γ) ⊙ LN(x) + β             # spatially-varying

    The last linear is zero-initialized for identity modulation at init.
    """

    def __init__(self, hidden_size: int, camera_dim: int, geo_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.norm = nn.LayerNorm(hidden_size)
        condition_dim = camera_dim + geo_dim
        self.modulation = nn.Sequential(
            nn.Linear(condition_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self._zero_init_modulation()

    def _zero_init_modulation(self):
        """Zero-init the last linear so γ=0, β=0 → identity on first forward."""
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, camera_token: torch.Tensor, raw_geo_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: fused spatial features [N, h_grid, w_grid, D]
            camera_token: global camera feature [N, 1, camera_dim]
            raw_geo_features: raw VGGT features before merger [N, h_patch, w_patch, geo_dim]
        Returns:
            modulated features, same shape as input x
        """
        orig_shape = x.shape
        N = x.shape[0]
        merge = self.spatial_merge_size

        _, h_patch, w_patch, geo_dim = raw_geo_features.shape
        h_grid = h_patch // merge
        w_grid = w_patch // merge
        geo_aligned = raw_geo_features[:, :h_grid * merge, :w_grid * merge, :]
        geo_aligned = geo_aligned.reshape(N, h_grid, merge, w_grid, merge, geo_dim)
        geo_aligned = geo_aligned.mean(dim=(2, 4))                   # [N, h_grid, w_grid, geo_dim]

        cam_spatial = camera_token.squeeze(1)[:, None, None, :]      # [N, 1, 1, cam_dim]
        cam_spatial = cam_spatial.expand(N, h_grid, w_grid, -1)      # [N, h_grid, w_grid, cam_dim]

        condition = torch.cat([geo_aligned, cam_spatial], dim=-1)     # [N, h_grid, w_grid, geo+cam]

        x_flat = x.reshape(N, -1, x.shape[-1])
        x_flat = self.norm(x_flat)

        cond_flat = condition.reshape(N, -1, condition.shape[-1])     # [N, h*w, geo+cam]
        params = self.modulation(cond_flat)                           # [N, h*w, 2*D]
        gamma, beta = params.chunk(2, dim=-1)                        # each [N, h*w, D]

        out = (1 + gamma) * x_flat + beta

        return out.reshape(orig_shape)


class DeepStackImportanceGate(nn.Module):
    """Per-token importance gate learned from LLM hidden states.

    At injection time, extracts the LLM hidden states at visual-token
    positions and predicts a per-token scalar gate g_i ∈ [0, 1]:

        g = Sigmoid(MLP(RMSNorm(h_visual)))   # [num_visual_tokens, 1]
        injected_feature = g * geo_feature

    The gate is conditioned on the LLM's own representation at each layer,
    so it learns which spatial positions are semantically relevant for the
    current task—suppressing uninformative regions (background, sky) while
    preserving geometry-rich ones (objects, surfaces).

    Inspired by GeoThinker's importance_net but adapted for additive
    deepstack injection (multiplicative gate on the feature, not an
    attention bias).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # from .modeling_qwen3_vl import Qwen3VLTextRMSNorm

        self.gate_net = nn.Sequential(
            # Qwen3VLTextRMSNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: full LLM hidden states [B, seq_len, D]
            visual_pos_masks: bool mask [B, seq_len] indicating visual positions
        Returns:
            gate: [num_visual_tokens, 1] per-token importance in [0, 1]
        """
        visual_hidden = hidden_states[visual_pos_masks]  # [num_visual_tokens, D]
        return self.gate_net(visual_hidden)               # [num_visual_tokens, 1]
