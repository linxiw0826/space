import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder (model configuration)
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gate")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")
    use_geometry_loss: bool = field(default=False)  # Whether to compute and add geometry loss during training
    use_object_geometry_loss: bool = field(default=False)  # Whether to compute and add object-based geometry loss during training
    use_proj_3d: bool = field(default=False)  # Whether to use MLP projection layer to align image_embeds_output with VGGT features (2048)
    use_mlp_gate: bool = field(default=False)  # Whether to use MLP-based gate (multi-layer) instead of simple linear gate when fusion_method is "gated"
    use_camera_gate: bool = field(default=False)  # Whether to use camera features as gate signal: g = Swish(P_g1(C̄)) ⊙ P_g2(C̄)
    use_feature_fusion_module: bool = field(default=False)  # Use FeatureFusionModule for 2D/3D fusion
    use_camera_method: Optional[str] = field(default=None)  # "adaLN": FiLM-style modulation with camera token (γ⊙LN(x)+β); "cat": prepend camera token
    use_geometry_deepstack_only: bool = field(default=False, metadata={"help": "If enabled, VGGT features are only used through geometry deepstack injection. The main VGGT-visual fusion path before entering the LLM is skipped."})
    geometry_deepstack_indexes: Optional[str] = field(default=None, metadata={"help": "Comma-separated VGGT layer indices for geometry deepstack, e.g. '1,8,20'. Must match the number of visual deepstack layers. Features are added to visual deepstack."})
    geometry_deepstack_indexes_pro: Optional[str] = field(default=None, metadata={"help": "VGGT-to-LLM independent deepstack mapping. Format: 'vggt:llm[-llm-...],...'. One VGGT layer can target multiple LLM layers (shared merger), e.g. '7:0-1-2-3,14:4-5-6-7'. Single target also works: '7:0,14:5'. Mutually exclusive with geometry_deepstack_indexes."})
    use_deepstack_importance_gate: Optional[str] = field(default=None, metadata={"help": "Importance gating on geometry deepstack (pro mode only). 'all' = gate all target LLM layers; comma-separated LLM layer indices (e.g. '0,5,10') = gate only those layers."})
    use_deepstack_global_gate: Optional[str] = field(default=None, metadata={"help": "Per-layer learnable scalar gate (tanh) on geometry deepstack (pro mode only). Initialized to 0 so injection starts silent. 'all' = all target LLM layers; comma-separated LLM layer indices (e.g. '0,5,10') = gate only those layers."})
    use_deepstack_camera_adaln: Optional[str] = field(default=None, metadata={"help": "Per-layer DualCameraAdaLN modulation on geometry deepstack (pro mode only). Uses the corresponding VGGT deepstack camera token to modulate both the VGGT deepstack feature and the target LLM hidden states before addition. 'all' = all target LLM layers; comma-separated LLM layer indices (e.g. '0,5,10') = only those layers."})

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    # VG-LLM style resize before image_processor patchification (only affects local processor)
    vgllm_resize: bool = field(default=False)
    vgllm_resize_mode: str = field(default="crop")  # "crop" or "pad"
    vgllm_target_size: int = field(default=512)
    # Geometry inputs support (for VGGT encoder inputs in data preprocessing)
    use_geometry_inputs: bool = field(default=False)  # Whether to return geometry_encoder_inputs for VGGT encoder
    use_patch_size_alin: bool = field(default=False)  # Align geometry inputs to image_grid_thw * 14
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
