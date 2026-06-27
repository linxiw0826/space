#!/bin/bash
# =============================================================================
# E-16: LFP feed + predict — both paths on (cross-attn feature injection AND
# the auxiliary future-latent prediction head).
#
# Two-stage warm-start training, STRUCTURALLY IDENTICAL to E-03a (same
# hyperparameters, same E-00b warm-start checkpoint, same VGGT geometry on).
# The ONLY differences vs train_e03a_mope_crossattn_two_stage.sh are the two
# paper-2 Stage-1 flags (spec: state/analyses/20260619_stage1_lfp_head_integration.md):
#   --mope_feed_features True    (keep injecting MoPE cross-attn residual; = E-03a)
#   --mope_lfp_enable   True     (ADD the 2-layer-MLP LFP head on top;
#                                  L_total = L_NTP + lambda * L_lfp)
# LFP hyperparameters use defaults (controlled variable): weight 0.1 /
# mse 1.0 / cos 1.0 / hidden 2048 / pred_source per_frame_video (FINAL
# token-shift: predict bin t+1 frozen-MoPE latent from bin t causal hidden,
# 3 pairs; decisions.md 2026-06-27) / align_strategy uniform.
# Before a full run, read the rank0 '[LFP DIAG]' print (first 3 steps) and
# tune --mope_lfp_weight by the ratio it reports (target ratio ~0.05-0.2).
#
# Paper-2 Stage-1 3-way ablation (see paper2_design.md §3):
#   E-03a  = feed only      (mope_feed_features=T, mope_lfp_enable=F)  baseline
#   E-15   = predict only   (mope_feed_features=F, mope_lfp_enable=T)
#   E-16   = feed + predict (mope_feed_features=T, mope_lfp_enable=T)  THIS
#
# Usage:
#   MODEL_SIZE=4b bash train_e16_lfp_both.sh   # default
#   MODEL_SIZE=8b bash train_e16_lfp_both.sh
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
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e16_lfp_both_4b}"
    run_name="space_e16_lfp_both_4b_lr1e-5"
    RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${SPACE_OUTPUT_ROOT}/train/e16_lfp_both_4b/checkpoint-4000}
    [ ! -d "${RESUME_FROM_CHECKPOINT}" ] && RESUME_FROM_CHECKPOINT=""
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    grad_accum_steps=24
    # 8B requires ZeRO-3 to fit on 8x H20 GPUs.
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e00b_mope_projector_only_8b}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e16_lfp_both_8b}"
    run_name="space_e16_lfp_both_8b_lr1e-5"
    echo "WARNING: 8B E-16 is experimental — monitor VRAM usage carefully." >&2
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

# ---------------------------------------------------------------------------
# Entry point: our fork of the training framework
# ---------------------------------------------------------------------------
entry_file="${SPACE_ROOT}/src/train_framework/train_space.py"

# ---------------------------------------------------------------------------
# Training arguments
# (identical to E-03a except: output_dir + the two paper-2 Stage-1 flags)
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
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/e16_lfp_both_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-16 Training (MODEL_SIZE=${MODEL_SIZE}) — LFP feed + predict ==="
echo "Output : ${output_dir}"
echo "Log    : ${LOG_FILE}"
echo "Fusion : crossattn, batch=${batch_size}, accum=${grad_accum_steps}"
echo "Feed   : ON  (mope_feed_features=True)"
echo "LFP    : ON  (mope_lfp_enable=True, weight=0.1, token-shift predict-next-bin, src=per_frame_video)"
echo "Trainable: LLM + crossattn projector + LFP head (warm-start joint, D-10)"

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
