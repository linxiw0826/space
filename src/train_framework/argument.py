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
    mope_fusion_mode: str = field(
        default="add",
        metadata={"help": "MoPE fusion strategy. "
                          "'add': E-02a — global avg pool → broadcast-add bias to image_embeds. "
                          "'concat': E-02b — per-token projection, prepend to LLM input sequence. "
                          "'crossattn': E-02c — single-head cross-attention, image tokens as Q, MoPE as K/V, residual add, sequence length unchanged. "
                          "'qformer': E-02d — Q-Former with learnable queries, output concat to LLM sequence."},
    )
    mope_concat_num_tokens: int = field(
        default=784,
        metadata={"help": "Number of MoPE patch tokens for concat fusion (mope_fusion_mode=concat). "
                          "Used to prepend -100 labels so loss shape matches extended logits. "
                          "Default 784 = VideoMAEv2 ViT-B, 8 frames, 224×224, tubelet_size=2."},
    )
    mope_qformer_num_queries: int = field(
        default=32,
        metadata={"help": "Number of learnable queries for Q-Former fusion (mope_fusion_mode=qformer). "
                          "Output shape: [B, mope_qformer_num_queries, llm_dim]. "
                          "Default 32."},
    )
    freeze_mope_projector: bool = field(
        default=False,
        metadata={"help": "Freeze MoPE projector weights during training. Used in Phase 2 two-stage "
                          "experiments (E-03a, E-03b) where projector was pre-trained in Stage 1 "
                          "(E-00b / E-00c) and LLM is trained in Stage 2."},
    )
    mope_use_gate: bool = field(
        default=False,
        metadata={"help": "E-10 (Router v1): enable the learned content-driven scalar gate g that "
                          "modulates the MoPE cross-attention residual (image_embeds + g*out). "
                          "Only applies when mope_fusion_mode=crossattn. Default False keeps "
                          "E-02c/E-03a behavior byte-for-byte unchanged. The gate submodule lives "
                          "inside MoPEProjectorCrossAttn and is trainable whenever the projector is "
                          "(managed by --freeze_mope_projector; no separate parameter group needed)."},
    )
    mope_gate_mode: str = field(
        default="learned",
        metadata={"help": "E-10 gate mode (only used when mope_use_gate=True). "
                          "'learned': g = sigmoid(MLP(pooled_question_text)) — content-driven main "
                          "path (D-15 b). 'oracle_task': true static/dynamic task-label hard gate "
                          "(E-10-oracle routing upper bound, D-15 a) — NOT implemented in this step."},
    )
    gate_log_interval: int = field(
        default=25,
        metadata={"help": "E-10 (Router v1): step interval for the GateStatsCallback process log. "
                          "Logs g_mean/g_std/g_min/g_max/n + gate_grad_norm every N steps (plus the "
                          "first few steps), rank0 only, via rank0_print into the tee'd LOG_FILE. "
                          "Only active when mope_use_gate=True; no effect otherwise (callback not added). "
                          "Watch for [E10-gate] lines: g_std>0 and rising = the router is learning."},
    )

    # -----------------------------------------------------------------------
    # E-10b (Router v1.1): gate anti-collapse three-piece fix + diagnostics.
    # All default to the E-10 status quo, so existing E-03a/E-02c/E-10 scripts
    # are byte-for-byte unchanged. The E-10b train script turns them on
    # explicitly via --mope_gate_anticollapse True (+ the three coefs).
    # PENDING[D-15]: gate input is still content-driven (b); the anti-collapse
    # terms only regularise that learned scalar — no task-label supervision.
    # -----------------------------------------------------------------------
    mope_gate_anticollapse: bool = field(
        default=False,
        metadata={"help": "E-10b: enable the gate anti-collapse loss bundle (z-loss + Bernoulli "
                          "entropy with linear warm-up). Default False = E-10 status quo (no extra "
                          "loss). Only takes effect when mope_use_gate=True. The E-10b train script "
                          "sets this True; E-03a/E-02c/E-10 leave it False (unchanged behaviour)."},
    )
    mope_gate_init_bias: float = field(
        default=0.0,
        metadata={"help": "E-10b A1: final-layer bias init of the gate MLP. Default 0.0 -> g starts "
                          "≈0.5 (sigmoid slope maximal, non-saturated). The E-10 collapse-prone value "
                          "was +4.0 (g≈0.98 saturated); set --mope_gate_init_bias 4.0 to reproduce the "
                          "E-10 control. Used at projector construction; no effect when mope_use_gate=False."},
    )
    mope_gate_zloss_coef: float = field(
        default=1e-3,
        metadata={"help": "E-10b A2: coefficient of the gate logit z-loss L_z = mean(logit^2), which "
                          "penalises the pre-sigmoid scalar drifting to ±large (re-saturation). Added "
                          "to the task loss only when mope_gate_anticollapse=True. Default 1e-3."},
    )
    mope_gate_entropy_coef: float = field(
        default=1e-2,
        metadata={"help": "E-10b A3: max coefficient lambda_max of the Bernoulli batch-usage entropy "
                          "anti-collapse term. Loss adds -lambda_t * H(g_bar) where g_bar is the batch "
                          "mean gate and H is the binary entropy (maximising H spreads usage off the "
                          "0/1 corners). Linearly warmed up over mope_gate_entropy_warmup_steps. "
                          "Default 0.01. Only active when mope_gate_anticollapse=True."},
    )
    mope_gate_entropy_warmup_steps: int = field(
        default=500,
        metadata={"help": "E-10b A3: linear warm-up length T_warm for the entropy coefficient: "
                          "lambda_t = lambda_max * min(1, step / T_warm). Default 500 (~10%% of a "
                          "typical single-epoch VSI-590K run). Set 0 to apply lambda_max from step 0."},
    )
    mope_gate_diag_every: int = field(
        default=20,
        metadata={"help": "E-10b change B: step interval for the rich [gate-diag] training log "
                          "(gate value histogram, logit saturation, gate_mlp weight/grad norms, loss "
                          "decomposition, residual ratio, optional static/dynamic g split). rank0 only, "
                          "into the tee'd LOG_FILE. Only active when mope_use_gate=True. Default 20."},
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
