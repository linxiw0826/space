#!/bin/bash
# =============================================================================
# E-16b: token-shift LFP (feed + predict) + VICReg anti-collapse regularizer.
#
# Identical to train_e16_lfp_both.sh (feed=True, lfp=True, lambda=0.05, same
# E-00b warm-start checkpoint, same VGGT geometry on, cards 0-3) PLUS the single
# E-16b anti-collapse flag --mope_lfp_reg_weight (beta). When beta=0 the loss is
# byte-for-byte identical to E-16; this script enables it (default 1.0).
#
# Diagnosis (paper2_design.md §4.1 / decisions.md 2026-06-30): E-16 dropped VSI
# on 7/8 subtasks — the auxiliary motion target GLOBALLY squeezed the LLM's
# spatial representation. E-16b adds a VICReg regularizer (variance hinge +
# covariance decorrelation) on the per-token GLOBAL visual hidden to keep it
# spread out / decorrelated. Stage A of the score-first roadmap (single variable:
# ONLY the regularizer is added vs E-16).
#
# The ONLY differences vs train_e16_lfp_both.sh:
#   --mope_lfp_reg_weight ${LFP_REG_WEIGHT}   (beta; 1.0 = this experiment value)
#   EXP_NAME / output_dir / run_name -> e16b_anticollapse_4b
# LFP hyperparameters use the E-16 values (feed=T, lfp=T, lambda=0.05 default).
# Before a full run, read the rank0 '[LFP DIAG]' print (first 3 steps): tune
# --mope_lfp_weight by ratio_L_lfp/L_ntp (~0.05-0.2) and --mope_lfp_reg_weight
# by beta*L_reg vs L_ntp (target ~0.05-0.2). Both overridable via env:
#   LFP_WEIGHT=0.05 LFP_REG_WEIGHT=1.0 bash train_e16b_anticollapse.sh
#
# Paper-2 Stage-1 score-first roadmap (paper2_design.md §4.1):
#   E-16   = feed + predict                       (the regressed baseline)
#   E-16b  = feed + predict + VICReg anti-collapse (THIS, Stage A)
#
# Usage:
#   MODEL_SIZE=4b bash train_e16b_anticollapse.sh   # default
#   MODEL_SIZE=8b bash train_e16b_anticollapse.sh
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../../_common/env/activate.sh"

# ---------------------------------------------------------------------------
# Model size switch (4b default)
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

# ---------------------------------------------------------------------------
# Distributed training configuration (locked to cards 0-3, per train_e03a)
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Single-node training only; multi-node not supported.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ---------------------------------------------------------------------------
# Path configuration (override via env vars as needed)
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}
GUIDE_ROOT="${SPACE_ROOT}/src"
MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

VGGT_PATH=${VGGT_PATH:-/home/nvme01/wlx/Space_sensing/models/VGGT-1B}

# MoPE checkpoint (ep199, vitb_1 full training run)
MOPE_CKPT_PATH=${MOPE_CKPT_PATH:-/home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_1/checkpoint-199.pth}

# Path to MoPE source code (for encoder loading)
MOPE_CODE_PATH=${MOPE_CODE_PATH:-${SPACE_ROOT}/src/vendor/mope}

# ---------------------------------------------------------------------------
# Per-size configuration
# ---------------------------------------------------------------------------
if [ "${MODEL_SIZE}" = "4b" ]; then
    batch_size=2
    grad_accum_steps=6
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e00b_mope_projector_only_4b}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e16b_anticollapse_4b}"
    run_name="space_e16b_anticollapse_4b_lr1e-5"
    # RESUME TRAP GUARD: resume ONLY from THIS experiment's own output_dir
    # checkpoint; never from e16's. The directory is unique per experiment.
    RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${SPACE_OUTPUT_ROOT}/train/e16b_anticollapse_4b/checkpoint-4000}
    [ ! -d "${RESUME_FROM_CHECKPOINT}" ] && RESUME_FROM_CHECKPOINT=""
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    grad_accum_steps=24
    # 8B requires ZeRO-3 to fit on 8x H20 GPUs.
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e00b_mope_projector_only_8b}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e16b_anticollapse_8b}"
    run_name="space_e16b_anticollapse_8b_lr1e-5"
    echo "WARNING: 8B E-16b is experimental — monitor VRAM usage carefully." >&2
else
    echo "ERROR: Unknown MODEL_SIZE='${MODEL_SIZE}'. Must be '4b' or '8b'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/train}
mkdir -p "${LOG_DIR}"
mkdir -p "${output_dir}"

# ---------------------------------------------------------------------------
# PYTHONPATH segments: see scripts/_common/setup_pythonpath.sh
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# VSI-590K dataset paths (read by src/qwenvl/data/__init__.py)
# ---------------------------------------------------------------------------
export VSI590K_SPAR_ANN=${VSI590K_SPAR_ANN:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_spar_590k.json}
export VSI590K_VIDEO_ANN=${VSI590K_VIDEO_ANN:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_video_590k.json}
export VSI590K_DATA_ROOT=${VSI590K_DATA_ROOT:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/}

# ---------------------------------------------------------------------------
# Weights & Biases (optional — leave WANDB_API_KEY unset to disable)
# ---------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
fi

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
lr=1e-5
# LFP auxiliary loss weight lambda — same value as E-16 (controlled variable).
# Override: LFP_WEIGHT=0.03 bash ...
LFP_WEIGHT=${LFP_WEIGHT:-0.05}
# E-16b anti-collapse (VICReg) weight beta. Default 1.0 = this experiment's
# value (the SINGLE new variable vs E-16). Set 0.0 to recover byte-equivalent
# E-16. Tune via [LFP DIAG] (target beta*L_reg ~0.05-0.2 of L_ntp).
# Override: LFP_REG_WEIGHT=0.5 bash ...
LFP_REG_WEIGHT=${LFP_REG_WEIGHT:-1.0}

# E-16b WARN-1 fix: scale-normalize the VICReg target to O(1) so the variance
# hinge (tau=1) fires on LLM raw hidden (per-dim std >> 1). Default True.
# Override: LFP_REG_NORMALIZE=False bash ...  (recover raw-hidden scale)
LFP_REG_NORMALIZE=${LFP_REG_NORMALIZE:-True}

# ---------------------------------------------------------------------------
# Entry point: our fork of the training framework
# ---------------------------------------------------------------------------
entry_file="${SPACE_ROOT}/src/train_framework/train_space.py"

# ---------------------------------------------------------------------------
# Training arguments
# (identical to E-16 except: output_dir + --mope_lfp_reg_weight)
# ---------------------------------------------------------------------------
args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${GUIDE_CKPT_PATH} \
    --dataset_use vsi590k_spar \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --optim adamw_torch \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size * 2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 268324 \
    --min_pixels 8192 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 12800 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to none \
    --use_geometry_inputs True \
    --use_geometry_encoder True \
    --use_feature_fusion_module True \
    --use_patch_size_alin True \
    --geometry_deepstack_indexes_pro 7:0,10:1,13:2,16:3,19:4,22:5 \
    --use_deepstack_importance_gate all \
    --use_deepstack_global_gate all \
    --geometry_encoder_type vggt \
    --geometry_encoder_path ${VGGT_PATH} \
    --use_mope True \
    --mope_fusion_mode crossattn \
    --mope_checkpoint_path ${MOPE_CKPT_PATH} \
    --mope_encoder_path ${MOPE_CODE_PATH} \
    --mope_all_frames 8 \
    --group_by_modality_length True \
    --mope_feed_features True \
    --mope_lfp_enable True \
    --mope_lfp_weight ${LFP_WEIGHT} \
    --mope_lfp_reg_weight ${LFP_REG_WEIGHT} \
    --mope_lfp_reg_normalize ${LFP_REG_NORMALIZE} \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/e16b_anticollapse_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-16b Training (MODEL_SIZE=${MODEL_SIZE}) — LFP feed+predict + VICReg anti-collapse ==="
echo "Output : ${output_dir}"
echo "Log    : ${LOG_FILE}"
echo "Fusion : crossattn, batch=${batch_size}, accum=${grad_accum_steps}"
echo "Feed   : ON  (mope_feed_features=True)"
echo "LFP    : ON  (mope_lfp_enable=True, weight=${LFP_WEIGHT}, token-shift predict-next-bin, src=per_frame_video)"
echo "VICReg : ON  (mope_lfp_reg_weight=${LFP_REG_WEIGHT}, normalize=${LFP_REG_NORMALIZE}, var+cov on per-token global visual hidden)"
echo "Trainable: LLM + crossattn projector + LFP head (warm-start joint, D-10)"

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
