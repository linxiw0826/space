"""
Training argument dataclasses for the Space Sensing framework.

Forked from:
  refs/Let_Geometry_GUIDE/qwen-vl-finetune/qwenvl/train/argument.py

Changes vs. upstream:
  - Added ``MoPEArguments`` dataclass with all MoPE-specific fields.
  - ``ModelArguments`` is unchanged from GUIDE so checkpoints remain compatible.
  - ``DataArguments`` and ``TrainingArguments`` are unchanged from GUIDE.

Usage in train_space.py:
    parser = transformers.HfArgumentParser(
        (ModelArguments, MoPEArguments, DataArguments, TrainingArguments)
    )
    model_args, mope_args, data_args, training_args = \
        parser.parse_args_into_dataclasses()
"""

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


# ---------------------------------------------------------------------------
# ModelArguments  (verbatim copy from GUIDE — do not modify)
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)
    geometry_encoder_type: str = field(default="vggt")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")
    reference_frame: str = field(default="first")
    feature_fusion_method: str = field(default="add")
    fusion_num_layers: int = field(default=1)
    geometry_merger_type: str = field(default="mlp")
    use_geometry_loss: bool = field(default=False)
    use_object_geometry_loss: bool = field(default=False)
    use_proj_3d: bool = field(default=False)
    use_mlp_gate: bool = field(default=False)
    use_camera_gate: bool = field(default=False)
    use_feature_fusion_module: bool = field(default=False)
    use_camera_method: Optional[str] = field(default=None)
    use_geometry_deepstack_only: bool = field(
        default=False,
        metadata={"help": "If enabled, VGGT features are only used through geometry deepstack "
                          "injection. The main VGGT-visual fusion path before entering the LLM "
                          "is skipped."},
    )
    geometry_deepstack_indexes: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated VGGT layer indices for geometry deepstack, e.g. "
                          "'1,8,20'. Must match the number of visual deepstack layers."},
    )
    geometry_deepstack_indexes_pro: Optional[str] = field(
        default=None,
        metadata={"help": "VGGT-to-LLM independent deepstack mapping. Format: "
                          "'vggt:llm[-llm-...],...'. One VGGT layer can target multiple LLM "
                          "layers (shared merger), e.g. '7:0-1-2-3,14:4-5-6-7'."},
    )
    use_deepstack_importance_gate: Optional[str] = field(
        default=None,
        metadata={"help": "Importance gating on geometry deepstack (pro mode only). "
                          "'all' = gate all target LLM layers; comma-separated LLM layer "
                          "indices (e.g. '0,5,10') = gate only those layers."},
    )
    use_deepstack_global_gate: Optional[str] = field(
        default=None,
        metadata={"help": "Per-layer learnable scalar gate (tanh) on geometry deepstack "
                          "(pro mode only). Initialized to 0 so injection starts silent."},
    )
    use_deepstack_camera_adaln: Optional[str] = field(
        default=None,
        metadata={"help": "Per-layer DualCameraAdaLN modulation on geometry deepstack "
                          "(pro mode only)."},
    )


# ---------------------------------------------------------------------------
# MoPEArguments  (new — Space Sensing extension)
# ---------------------------------------------------------------------------

@dataclass
class MoPEArguments:
    """Arguments for the MoPE (VideoMAEv2-based ViT-B) dynamic video encoder.

    Analogous to the geometry encoder flags in ``ModelArguments``.
    ``use_mope`` is the primary enable flag; when False all other fields are
    ignored at model initialisation time.
    """

    use_mope: bool = field(
        default=False,
        metadata={"help": "Enable MoPE dynamic encoder. When True, MoPEEncoder is attached "
                          "to the model (frozen) and MoPEProjector (trainable) is added."},
    )
    mope_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to MoPE codebase root directory (must contain models/__init__.py "
                          "and dataset/). Added to sys.path at initialisation."},
    )
    mope_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to MoPE checkpoint .pth file "
                          "(e.g. checkpoint-199.pth from a MoPE training run)."},
    )
    mope_llm_dim: int = field(
        default=-1,
        metadata={"help": "LLM embedding dimension for MoPEProjector output. "
                          "Set to -1 (default) to auto-detect from model.config.hidden_size. "
                          "Manual override useful for debugging or special architectures."},
    )
    mope_all_frames: int = field(
        default=8,
        metadata={"help": "Number of video frames passed to MoPEEncoder. "
                          "Must be consistent with data preprocessing."},
    )


# ---------------------------------------------------------------------------
# DataArguments  (verbatim copy from GUIDE — do not modify)
# ---------------------------------------------------------------------------

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    # VG-LLM style resize before image_processor patchification
    vgllm_resize: bool = field(default=False)
    vgllm_resize_mode: str = field(default="crop")
    vgllm_target_size: int = field(default=512)
    # Geometry inputs support
    use_geometry_inputs: bool = field(default=False)
    use_patch_size_alin: bool = field(default=False)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


# ---------------------------------------------------------------------------
# TrainingArguments  (verbatim copy from GUIDE — do not modify)
# ---------------------------------------------------------------------------

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
