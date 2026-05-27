#!/bin/bash
# =============================================================================
# E-03a: GUIDE + MoPE cross-attention fusion, two-stage training (unified 4B / 8B)
#
# Stage 2 of two-stage training: warm-start joint training (projector + LLM both
# trainable from E-00b checkpoint). Starting checkpoint is E-00b (projector-only,
# Stage 1 trained).
#
# Usage:
#   MODEL_SIZE=4b bash train_e03a_mope_crossattn_two_stage.sh   # default
#   MODEL_SIZE=8b bash train_e03a_mope_crossattn_two_stage.sh
#
# Supported MODEL_SIZE values: 4b  8b
#
# Key differences from E-02c:
#   - GUIDE_CKPT_PATH → E-00b projector-only checkpoint (Stage 1)
#   - --tune_mm_llm True              (LLM trains in Stage 2)
#   - projector + LLM both trainable  (warm-start joint training, D-10)
#   - --mope_fusion_mode crossattn    (same architecture as E-00b/E-02c)
#   - output_dir → e03a_mope_crossattn_two_stage_{size}
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Model size switch (4b default)
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

# ---------------------------------------------------------------------------
# Distributed training configuration
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Single-node training only; multi-node not supported.
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

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
    grad_accum_steps=4
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-/home/nvme03/wlx/Space_sensing/output/train/e00b_mope_projector_only_4b}
    output_dir="${OUTPUT_DIR:-/home/nvme03/wlx/Space_sensing/output/train/e03a_mope_crossattn_two_stage_4b}"
    run_name="space_e03a_mope_crossattn_two_stage_4b_lr1e-5"
    RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-/home/nvme03/wlx/Space_sensing/output/train/e03a_mope_crossattn_two_stage_4b/checkpoint-4000}
    [ ! -d "${RESUME_FROM_CHECKPOINT}" ] && RESUME_FROM_CHECKPOINT=""
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    grad_accum_steps=16
    # 8B requires ZeRO-3 to fit on 8x H800 GPUs.
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-/home/nvme03/wlx/Space_sensing/output/train/e00b_mope_projector_only_8b}
    output_dir="${OUTPUT_DIR:-/home/nvme03/wlx/Space_sensing/output/train/e03a_mope_crossattn_two_stage_8b}"
    run_name="space_e03a_mope_crossattn_two_stage_8b_lr1e-5"
    echo "WARNING: 8B E-03a is experimental — monitor VRAM usage carefully." >&2
else
    echo "ERROR: Unknown MODEL_SIZE='${MODEL_SIZE}'. Must be '4b' or '8b'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR=${LOG_DIR:-/home/nvme03/wlx/Space_sensing/logs/train}
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
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/e03a_mope_crossattn_two_stage_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-03a Training (MODEL_SIZE=${MODEL_SIZE}) ==="
echo "Output : ${output_dir}"
echo "Log    : ${LOG_FILE}"
echo "Fusion : crossattn, batch=${batch_size}, accum=${grad_accum_steps}"
echo "Trainable: LLM + MoPEProjectorCrossAttn (warm-start joint, D-10)"

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
