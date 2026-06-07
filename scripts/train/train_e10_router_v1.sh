#!/bin/bash
# =============================================================================
# E-10: Router v1 — GUIDE + frozen VGGT static expert + frozen MoPE dynamic
#       expert + LEARNED content-driven scalar gate g, two-stage training
#       (unified 4B / 8B).
#
# Architecture (论文2 R1, D-16/D-17):
#   - Static expert  = frozen VGGT geometry stream (D-14 方案A, reused as-is).
#   - Dynamic expert = frozen MoPE (last layer, E-03a config; 取层不动).
#   - Gate g         = learned content-driven scalar gate (D-15 b): modulates
#                      the MoPE cross-attention residual: image_embeds + g*out.
#   - No auxiliary loss; MoPE layer selection unchanged.
#
# Two-stage training (same recipe as E-03a, D-09/D-10):
#   - Stage 1 (TRAIN_STAGE=stage1): freeze LLM, train projector + gate.
#       start = E-00b projector-only checkpoint (or GUIDE checkpoint).
#       --tune_mm_llm False  --freeze_mope_projector False  (projector+gate train)
#   - Stage 2 (TRAIN_STAGE=stage2, DEFAULT): warm-start joint training.
#       start = E-10 Stage-1 checkpoint.
#       --tune_mm_llm True   (projector + gate + LLM all trainable)
#
# The gate submodule lives inside MoPEProjectorCrossAttn, so it follows the
# projector's requires_grad (managed by --freeze_mope_projector) in BOTH stages
# — no separate parameter group needed, g is trainable & non-zero-grad in both.
#
# Usage:
#   # Stage 1 (projector + gate only, frozen LLM):
#   TRAIN_STAGE=stage1 MODEL_SIZE=4b bash train_e10_router_v1.sh
#   # Stage 2 (warm-start joint; start from Stage-1 ckpt):
#   TRAIN_STAGE=stage2 MODEL_SIZE=4b bash train_e10_router_v1.sh   # default
#
# Supported MODEL_SIZE values: 4b  8b
#
# Key differences from E-03a:
#   - --mope_use_gate True   --mope_gate_mode learned   (E-10 learned gate)
#   - --model qwen3_vl_mope_router at eval time (see eval_e10_*.sh)
#   - output_dir → e10_router_v1_{size} (stage2) / e10_router_v1_stage1_{size}
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env/activate.sh"

# ---------------------------------------------------------------------------
# Model size + stage switch
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}
TRAIN_STAGE=${TRAIN_STAGE:-stage2}   # stage1 | stage2 (default: stage2 warm-start)

# ---------------------------------------------------------------------------
# Distributed training configuration
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
#
# Global batch alignment with E-03a (the reference experiment): E-10 is the
# gated counterpart of E-03a, so the global batch MUST match E-03a exactly,
# otherwise differences cannot be attributed to the gate. We therefore pin a
# TARGET_GLOBAL_BATCH per model size and DERIVE grad_accum from the current
# GPU count, so the global batch stays constant regardless of NPROC.
#   global_batch = batch_size * NPROC_PER_NODE * grad_accum_steps
# ---------------------------------------------------------------------------
if [ "${MODEL_SIZE}" = "4b" ]; then
    batch_size=2
    # E-03a 6-GPU baseline: 2 (per_device) * 6 (NPROC) * 4 (accum) = 48
    TARGET_GLOBAL_BATCH=48
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
    DEEPSPEED_CONFIG_8B_FLAG=0
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    # E-03a 6-GPU baseline: 1 (per_device) * 6 (NPROC) * 16 (accum) = 96
    TARGET_GLOBAL_BATCH=96
    # 8B requires ZeRO-3 to fit on 8x H20 GPUs.
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    DEEPSPEED_CONFIG_8B_FLAG=1
    echo "WARNING: 8B E-10 is experimental — monitor VRAM usage carefully." >&2
else
    echo "ERROR: Unknown MODEL_SIZE='${MODEL_SIZE}'. Must be '4b' or '8b'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Derive gradient_accumulation_steps so the GLOBAL batch stays aligned with
# E-03a regardless of GPU count:
#   grad_accum_steps = TARGET_GLOBAL_BATCH / (batch_size * NPROC_PER_NODE)
# Examples (4B, target=48): 4 GPUs -> 6, 6 GPUs -> 4, 3 GPUs -> 8.
#
# GRAD_ACCUM_STEPS env explicitly overrides the auto-computation (and skips the
# divisibility check); the resulting global batch is still echoed for review.
# ---------------------------------------------------------------------------
per_step_batch=$((batch_size * NPROC_PER_NODE))
if [ -n "${GRAD_ACCUM_STEPS:-}" ]; then
    grad_accum_steps=${GRAD_ACCUM_STEPS}
    global_batch=$((per_step_batch * grad_accum_steps))
    echo "NOTE: GRAD_ACCUM_STEPS overridden via env -> accum=${grad_accum_steps}, global_batch=${global_batch} (target=${TARGET_GLOBAL_BATCH})." >&2
else
    if [ $((TARGET_GLOBAL_BATCH % per_step_batch)) -ne 0 ]; then
        echo "ERROR: TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by batch_size*NPROC_PER_NODE=${batch_size}*${NPROC_PER_NODE}=${per_step_batch}." >&2
        echo "       This GPU count cannot evenly hit the target global batch. Change NPROC_PER_NODE, or set GRAD_ACCUM_STEPS explicitly to override." >&2
        exit 1
    fi
    grad_accum_steps=$((TARGET_GLOBAL_BATCH / per_step_batch))
    global_batch=$((per_step_batch * grad_accum_steps))
fi

# ---------------------------------------------------------------------------
# Stage-specific configuration: start checkpoint, tune_mm_llm, output dir.
# Stage 1: freeze LLM, train projector+gate, start from E-00b (Stage-1 warm).
# Stage 2: warm-start joint, train LLM+projector+gate, start from E-10 Stage1.
# ---------------------------------------------------------------------------
if [ "${TRAIN_STAGE}" = "stage1" ]; then
    TUNE_MM_LLM=False
    # Stage 1 starts from the E-00b projector-only checkpoint (same as E-03a Stage 1 source).
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e00b_mope_projector_only_${MODEL_SIZE}}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e10_router_v1_stage1_${MODEL_SIZE}}"
    run_name="space_e10_router_v1_stage1_${MODEL_SIZE}_lr1e-5"
    exp_name="e10_router_v1_stage1_${MODEL_SIZE}"
elif [ "${TRAIN_STAGE}" = "stage2" ]; then
    TUNE_MM_LLM=True
    # Stage 2 (warm-start joint) starts from the E-10 Stage-1 checkpoint.
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e10_router_v1_stage1_${MODEL_SIZE}}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e10_router_v1_${MODEL_SIZE}}"
    run_name="space_e10_router_v1_${MODEL_SIZE}_lr1e-5"
    exp_name="e10_router_v1_${MODEL_SIZE}"
else
    echo "ERROR: Unknown TRAIN_STAGE='${TRAIN_STAGE}'. Must be 'stage1' or 'stage2'." >&2
    exit 1
fi

# Resume support (only auto-resume from this run's own output dir).
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${output_dir}/checkpoint-4000}
[ ! -d "${RESUME_FROM_CHECKPOINT}" ] && RESUME_FROM_CHECKPOINT=""

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/train}
mkdir -p "${LOG_DIR}"
mkdir -p "${output_dir}"

# ---------------------------------------------------------------------------
# PYTHONPATH: our data/__init__.py must shadow GUIDE's, so it comes first
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# VSI-590K dataset paths (read by src/train_framework/data/__init__.py)
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
# ---------------------------------------------------------------------------
args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${GUIDE_CKPT_PATH} \
    --dataset_use vsi590k_spar \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm ${TUNE_MM_LLM} \
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
    --mope_use_gate True \
    --mope_gate_mode learned \
    --mope_checkpoint_path ${MOPE_CKPT_PATH} \
    --mope_encoder_path ${MOPE_CODE_PATH} \
    --mope_all_frames 8 \
    --group_by_modality_length True \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-10 Router v1 Training (MODEL_SIZE=${MODEL_SIZE}, STAGE=${TRAIN_STAGE}) ==="
echo "Start ckpt: ${GUIDE_CKPT_PATH}"
echo "Output    : ${output_dir}"
echo "Log       : ${LOG_FILE}"
echo "Fusion    : crossattn + learned content-driven gate, batch=${batch_size}, accum=${grad_accum_steps}, NPROC=${NPROC_PER_NODE}, global_batch=${global_batch} (target=${TARGET_GLOBAL_BATCH})"
if [ "${TRAIN_STAGE}" = "stage1" ]; then
    echo "Trainable : MoPEProjectorCrossAttn + gate (LLM frozen, Stage 1)"
else
    echo "Trainable : LLM + MoPEProjectorCrossAttn + gate (warm-start joint, Stage 2)"
fi

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
