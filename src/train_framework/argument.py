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
    load_mope_projector_from_ckpt: bool = field(
        default=False,
        metadata={"help": "E-10b warm-gate probe: load the MoPE projector (k/v/out_proj) weights from "
                          "--model_name_or_path WITHOUT freezing it. Unlike --freeze_mope_projector (which "
                          "loads AND freezes for Stage-1→Stage-2 warm-start), this loads the projector but "
                          "keeps it trainable, and tolerates a checkpoint that lacks gate_mlp keys "
                          "(strict=False -> the freshly-built gate_mlp keeps its constructor MI-seed init). "
                          "Purpose: start a gated crossattn run from a NON-gated warm checkpoint (e.g. E-03a "
                          "checkpoint-4000, out_proj≈1.147) so the dynamic branch is non-zero from step 0 and "
                          "the gate has a live d(loss)/d(g) signal ('doubly-dead gate' falsification probe). "
                          "Default False = unchanged behaviour for E-03a/E-02c/E-10/E-10b gatefix. Ignored "
                          "when --freeze_mope_projector is True (that path already loads the projector)."},
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
    mope_gate_lastw_std: float = field(
        default=1e-3,
        metadata={"help": "E-10b v2.2: gate MLP final-layer weight init std = the gate's content-seed "
                          "magnitude. Set a larger value (e.g. 0.5) so the init gate g spreads to "
                          "~0.3–0.7, giving the MI objective a content seed to amplify and reopening "
                          "the gradient path back to the content-read layer W1. Default 1e-3 = E-10 "
                          "status quo (near-zero W2; paired with bias 4.0 -> a constant g≈0.98). "
                          "Used at projector construction; no effect when mope_use_gate=False."},
    )
    mope_gate_zloss_coef: float = field(
        default=0.0,
        metadata={"help": "E-10b v2.1 A2: coefficient of the gate logit z-loss L_z = mean(logit^2). "
                          "DEFAULT 0.0 = z-loss OFF. v2 added it always, but its unique minimum at "
                          "logit=0 (g=0.5) actively pulled the gate toward the constant 0.5 and fought "
                          "the per-sample confidence the MI objective needs, so it is now off by "
                          "default; the MI objective is self-saturating/self-limiting. Set a small "
                          "positive value (e.g. 1e-3) only if you observe logit runaway. Still computed "
                          "for diagnostics (l_z_raw); added to the loss only when this coef > 0 AND "
                          "mope_gate_anticollapse=True."},
    )
    mope_gate_entropy_coef: float = field(
        default=1e-2,
        metadata={"help": "E-10b v2.1 A3: max coefficient lambda_max of the mutual-information (MI) "
                          "anti-collapse term (RIM/IMSAT). Loss adds -lambda_t * MI where "
                          "MI = H_marg - H_cond (marginal batch-usage entropy minus mean per-sample "
                          "entropy); maximising MI drives content-dependent divergence and makes the "
                          "all-0.5 collapse point MI's global MINIMUM (unstable saddle) instead of the "
                          "stable max the v2 batch-mean entropy term had. Linearly warmed up over "
                          "mope_gate_entropy_warmup_steps. Default 0.01. Only active when "
                          "mope_gate_anticollapse=True."},
    )
    mope_gate_entropy_warmup_steps: int = field(
        default=500,
        metadata={"help": "E-10b A3: linear warm-up length T_warm for the entropy coefficient: "
                          "lambda_t = lambda_max * min(1, step / T_warm). Default 500 (~10%% of a "
                          "typical single-epoch VSI-590K run). Set 0 to apply lambda_max from step 0."},
    )
    mope_gate_diag_every: int = field(
        default=10,
        metadata={"help": "E-10b change B: step interval for the rich [gate-diag] training log "
                          "(gate value histogram, logit saturation, gate_mlp weight norms, real "
                          "gate_grad_norm, MI loss decomposition H_marg/H_cond/MI, residual ratio). "
                          "rank0 only, into the tee'd LOG_FILE. Only active when mope_use_gate=True. "
                          "Default 10 (print every 10 steps)."},
    )

    # -----------------------------------------------------------------------
    # paper-2 Stage 1: LFP (Latent Future Prediction) auxiliary head.
    # ALL fields default to OFF / zero-impact (the same opt-in pattern as
    # ``load_mope_projector_from_ckpt`` and ``mope_use_gate``). When
    # ``mope_lfp_enable=False`` (the default) the LFP head is NOT built, the
    # outer LFP forward-patch is NOT installed, and the E-03a / E-10 / E-10b
    # paths are byte-for-byte unchanged. implements D-12 (论文2 辅助信号 =
    # MoPE latent future-feature prediction; Stage 1 activation per
    # decisions.md 2026-06-19, distillation-style target per paper2_design.md §2.3).
    # NOT a PENDING marker: D-12 is activated as the paper-2 main method.
    # -----------------------------------------------------------------------
    mope_lfp_enable: bool = field(
        default=False,
        metadata={"help": "paper-2 Stage 1 MAIN SWITCH: enable the LFP auxiliary prediction head "
                          "(LLM hidden state -> predict frozen-MoPE future-frame latent). When False "
                          "(default) the head is not built, the outer LFP forward-patch is not "
                          "installed, and the training path is byte-for-byte identical to E-03a. "
                          "Only meaningful when mope_fusion_mode=crossattn (the inner crossattn patch "
                          "must run to cache the frozen-MoPE latent for the LFP target). "
                          "implements D-12."},
    )
    mope_lfp_weight: float = field(
        default=0.1,
        metadata={"help": "LFP auxiliary loss weight lambda in L = L_NTP + lambda * L_lfp. "
                          "Default 0.1 (Cambrian-S scale). Monitor |L_lfp| vs |L_NTP|: the "
                          "auxiliary term must stay ~1 order of magnitude below NTP so it "
                          "reshapes the representation without swamping the LM objective "
                          "(risk-7). lambda=0.1 is the starting value; tune per run. "
                          "Before a full run, read the rank0 '[LFP DIAG]' print (first 3 "
                          "steps) and adjust lambda by the ratio_L_lfp/L_ntp it reports: "
                          "target lambda*L_lfp ~1 order of magnitude below L_ntp "
                          "(ratio ~0.05-0.2); >0.5 -> lower lambda, <0.01 -> raise it."},
    )
    mope_lfp_mse_weight: float = field(
        default=1.0,
        metadata={"help": "LFP MSE term weight in L_lfp = mse_w*MSE + cos_w*(1-cos). Default 1.0."},
    )
    mope_lfp_cos_weight: float = field(
        default=1.0,
        metadata={"help": "LFP cosine term weight in L_lfp = mse_w*MSE + cos_w*(1-cos). Default 1.0."},
    )
    mope_lfp_hidden: int = field(
        default=2048,
        metadata={"help": "LFP head MLP hidden width (Cambrian-S scale). Default 2048. "
                          "Used at head construction; no effect when mope_lfp_enable=False."},
    )
    mope_lfp_target_dim: int = field(
        default=768,
        metadata={"help": "LFP target dimension (= MoPE embed_dim, fixed 768 for ViT-B). "
                          "Used at head construction; no effect when mope_lfp_enable=False."},
    )
    mope_lfp_context_frames: int = field(
        default=4,
        metadata={"help": "DEPRECATED (kept for back-compat, no effect on the default "
                          "token-shift path). Was the leading-context frame count for the "
                          "old distillation-style target (方案 4-A). The FINAL token-shift "
                          "target (decisions.md 2026-06-27, pred_source=per_frame_video) does "
                          "NOT use a context/future split — it derives 3 predict-next-bin pairs "
                          "from the 4 latent time-bins directly. Only consulted when "
                          "pred_source != 'per_frame_video' (legacy distillation path)."},
    )
    mope_lfp_target_pool: str = field(
        default="mean",
        metadata={"help": "LFP target aggregation: 'mean' (default) -> [B, 768] (single-vector "
                          "prediction, the Cambrian-S 2-layer-MLP template, most stable); "
                          "'token' -> [B, N_future, 768] (per-token prediction, reserved for a "
                          "later Stage-1 extension). Stage 1 uses 'mean'."},
    )
    mope_lfp_pred_source: str = field(
        default="per_frame_video",
        metadata={"help": "Selects the LFP target/source construction. 'per_frame_video' "
                          "(DEFAULT, FINAL token-shift path, decisions.md 2026-06-27): pool the "
                          "LLM last hidden state PER VIDEO FRAME (one mean-pool per contiguous "
                          "run of video_token_id, temporal order), uniformly group the F frames "
                          "into the MoPE 4 time-bins, and predict bin t+1's frozen-MoPE latent "
                          "from bin t's causal hidden (3 predict-next-bin pairs). This is the "
                          "Cambrian-S-faithful TRUE-future objective with no leakage (LLM is "
                          "strictly causal). 'last_answer' (legacy, BLOCK-1 distillation): read "
                          "the hidden state at the last non -100 label token — kept only to "
                          "reproduce old runs; it leaks future info and is no longer the default."},
    )
    mope_lfp_align_strategy: str = field(
        default="uniform",
        metadata={"help": "How to align the F LLM video frames to the MoPE 4 time-bins for the "
                          "token-shift target (pred_source=per_frame_video). 'uniform' (default): "
                          "bin b covers LLM frames [b*F//4, (b+1)*F//4) (time-segment alignment, "
                          "robust to F != 8 since the LLM frame count is set by the video_processor "
                          "while MoPE always uses 8 frames). Reserved knob; only 'uniform' is "
                          "currently implemented."},
    )
    # -----------------------------------------------------------------------
    # E-16b: anti-collapse / decorrelation regularizer (VICReg) on top of the
    # token-shift LFP objective. ALL fields default to OFF / zero-impact (the
    # same opt-in pattern as the E-10b anti-collapse bundle). When
    # ``mope_lfp_reg_weight=0.0`` (the default) the regularizer branch is NOT
    # entered, no reg tensor is built, and the loss is BYTE-FOR-BYTE identical
    # to E-16. Diagnosis (decisions.md / paper2_design.md §4.1, 2026-06-30):
    # E-16 globally squeezes the LLM's spatial representation; the regularizer
    # keeps the GLOBAL per-token visual hidden spread out (variance hinge) and
    # decorrelated (covariance). NOT a PENDING marker: the method (VICReg),
    # target (per-token visual hidden), and defaults are all fully specified.
    # -----------------------------------------------------------------------
    mope_lfp_reg_weight: float = field(
        default=0.0,
        metadata={"help": "E-16b anti-collapse regularization weight beta in "
                          "L = L_NTP + lambda*L_lfp + beta*(L_var + cov_scale*L_cov) "
                          "(VICReg on the per-token GLOBAL visual hidden). Default 0.0 "
                          "(regularizer OFF -> byte-for-byte identical to E-16; opt-in, "
                          "no baseline pollution). Set to a non-zero value (e.g. 1.0) to "
                          "enable. Tune per run by reading the rank0 '[LFP DIAG]' print "
                          "(first 3 steps): keep beta*L_reg ~0.05-0.2 of L_ntp so it "
                          "reshapes without swamping the LM objective. Only meaningful "
                          "when mope_lfp_enable=True."},
    )
    mope_lfp_reg_type: str = field(
        default="vicreg",
        metadata={"help": "E-16b regularization type. 'vicreg' (default, only type "
                          "implemented) = variance hinge + covariance decorrelation, no "
                          "invariance (single view). Only active when "
                          "mope_lfp_reg_weight > 0."},
    )
    mope_lfp_reg_var_hinge: float = field(
        default=1.0,
        metadata={"help": "E-16b VICReg variance hinge threshold tau in "
                          "L_var = mean_d max(0, tau - sqrt(Var(z_d)+eps)). Default 1.0 "
                          "(VICReg standard). Raise (e.g. 2.0) if per-dim std stays low; "
                          "lower (e.g. 0.5) if it overshoots. Only used when "
                          "mope_lfp_reg_type='vicreg' and mope_lfp_reg_weight > 0."},
    )
    mope_lfp_reg_cov_scale: float = field(
        default=1.0,
        metadata={"help": "E-16b VICReg covariance term scale in "
                          "L_reg = L_var + cov_scale*L_cov, where "
                          "L_cov = (sum_{i!=j} Cov_{ij}^2)/D. Default 1.0. Only used when "
                          "mope_lfp_reg_type='vicreg' and mope_lfp_reg_weight > 0."},
    )
    mope_lfp_reg_normalize: bool = field(
        default=True,
        metadata={"help": "E-16b WARN-1 fix: scale-normalize the VICReg target "
                          "before the variance/covariance terms. LLM raw hidden has "
                          "per-dim std >> 1, which saturates the variance hinge "
                          "relu(tau - std) (tau=1, tuned for O(1) features) so it only "
                          "catches absolute collapse, not relative collapse. True "
                          "(default) divides the target by a SINGLE global scalar std "
                          "(overall scale -> ~1, relative per-dim variance ratios kept) "
                          "so tau=1 fires. NOT per-dim standardization (that would make "
                          "L_var identically 0). Set False to use raw-hidden scale. "
                          "Only consulted when mope_lfp_reg_weight > 0 (when beta=0 the "
                          "reg branch is not entered and no normalization happens -> "
                          "byte-for-byte E-16)."},
    )
    mope_feed_features: bool = field(
        default=True,
        metadata={"help": "paper-2 Stage 1 feed-features bypass (orthogonal to mope_lfp_enable). "
                          "True (default) = E-03a behavior: the cross-attn residual fuses MoPE "
                          "features into each image-embed shard. False = skip the residual fusion "
                          "(return image_embeds untouched) — the LLM no longer SEES MoPE features "
                          "injected, but the frozen encoder still runs once per forward and its "
                          "latent is cached for the LFP target path. This flag + mope_lfp_enable "
                          "form the three orthogonal arms: E-03a (feed=T, lfp=F) / E-15 "
                          "prediction-only (feed=F, lfp=T) / E-16 both (feed=T, lfp=T). Default "
                          "True keeps E-03a/E-10/E-10b byte-for-byte unchanged."},
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
