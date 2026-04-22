"""VGGT geometry encoder implementation."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class VGGTEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        
        # Lazy import to avoid circular dependencies
        from ..vggt.models.vggt import VGGT

        # Initialize VGGT model
        self.vggt = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
        
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame    
        self.patch_size = 14
        
    
    def encode(self, images: torch.Tensor, geometry_deepstack_indexes=None) -> tuple:
        """Encode images using VGGT.
        
        Args:
            images: Input images tensor.
            geometry_deepstack_indexes: Optional list of VGGT layer indices to extract
                deepstack features from, e.g. [1, 8, 20].
        
        Returns:
            Tuple of (features, camera_features, deepstack_features, deepstack_camera_features).
            deepstack_features and deepstack_camera_features are None when
            geometry_deepstack_indexes is not provided.
        """
        self.vggt.eval()

        # Apply reference frame transformation
        images = self._apply_reference_frame_transform(images)
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images[None])
                features = aggregated_tokens_list[-2][0, :, patch_start_idx:]
                camera_features = aggregated_tokens_list[-2][0, :, 0]

                deepstack_features = None
                deepstack_camera_features = None
                if geometry_deepstack_indexes is not None:
                    deepstack_features = []
                    deepstack_camera_features = []
                    for idx in geometry_deepstack_indexes:
                        layer_tokens = aggregated_tokens_list[idx][0]
                        feat = layer_tokens[:, patch_start_idx:]
                        cam = layer_tokens[:, 0]
                        deepstack_features.append(feat)
                        deepstack_camera_features.append(cam)

        features = self._apply_inverse_reference_frame_transform(features)
        if deepstack_features is not None:
            deepstack_features = [
                self._apply_inverse_reference_frame_transform(f) for f in deepstack_features
            ]
        if deepstack_camera_features is not None:
            deepstack_camera_features = [
                self._apply_inverse_reference_frame_transform(c) for c in deepstack_camera_features
            ]
        
        return features, camera_features, deepstack_features, deepstack_camera_features
    
    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048  # VGGT feature dimension
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)
    
    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images
    
    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    
    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT model."""
        from ..vggt.models.vggt import VGGT
        self.vggt = VGGT.from_pretrained(model_path, enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
                
        # Freeze parameters if required
        if self.freeze_encoder:
            for param in self.vggt.parameters():
                param.requires_grad = False
