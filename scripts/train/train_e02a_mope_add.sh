#!/bin/bash
# =============================================================================
# E-02a: GUIDE + MoPE input-level add (unified 4B / 8B)
#
# Usage:
#   MODEL_SIZE=4b bash train_e02a_mope_add.sh   # default
#   MODEL_SIZE=8b bash train_e02a_mope_add.sh
#
# Supported MODEL_SIZE values: 4b  8b
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
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# ---------------------------------------------------------------------------
# Path configuration (override via env vars as needed)
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}
GUIDE_ROOT="${SPACE_ROOT}/src"
MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

QWEN3_VL_4B_PATH=${QWEN3_VL_4B_PATH:-/home/nvme03/wlx/Space_sensing/models/Qwen3-VL-4B-Instruct}
QWEN3_VL_8B_PATH=${QWEN3_VL_8B_PATH:-/home/nvme03/wlx/Space_sensing/models/Qwen3-VL-8B-Instruct}

VGGT_PATH=${VGGT_PATH:-/home/nvme03/wlx/Space_sensing/models/VGGT-1B}
GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-/home/nvme03/wlx/Space_sensing/models/guide/}

# MoPE checkpoint (ep199, vitb_1 full training run)
MOPE_CKPT_PATH=${MOPE_CKPT_PATH:-/home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_1/checkpoint-199.pth}

# Path to MoPE source code (for encoder loading)
MOPE_CODE_PATH=${MOPE_CODE_PATH:-${SPACE_ROOT}/src/vendor/mope}

# ---------------------------------------------------------------------------
# Per-size configuration
# ---------------------------------------------------------------------------
if [ "${MODEL_SIZE}" = "4b" ]; then
    MODEL_PATH=${QWEN3_VL_4B_PATH}
    batch_size=1
    grad_accum_steps=8
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
    output_dir="${SPACE_ROOT}/outputs/e02a_mope_add_4b"
    run_name="space_e02a_mope_add_4b_lr1e-5"
elif [ "${MODEL_SIZE}" = "8b" ]; then
    MODEL_PATH=${QWEN3_VL_8B_PATH}
    batch_size=1
    grad_accum_steps=16
    # 8B requires ZeRO-3 to fit on 8x H800 GPUs
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    output_dir="${SPACE_ROOT}/outputs/e02a_mope_add_8b"
    run_name="space_e02a_mope_add_8b_lr1e-5"
else
    echo "ERROR: Unknown MODEL_SIZE='${MODEL_SIZE}'. Must be '4b' or '8b'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR=${LOG_DIR:-${SPACE_ROOT}/logs/train}
mkdir -p "${LOG_DIR}"
mkdir -p "${output_dir}"

# ---------------------------------------------------------------------------
# PYTHONPATH: our data/__init__.py must shadow GUIDE's, so it comes first
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# VSI-590K dataset paths (read by src/train_framework/data/__init__.py)
# ---------------------------------------------------------------------------
export VSI590K_SPAR_ANN=${VSI590K_SPAR_ANN:-${SPACE_ROOT}/data/vsi590k_spar.json}
export VSI590K_VIDEO_ANN=${VSI590K_VIDEO_ANN:-${SPACE_ROOT}/data/vsi590k_video.json}
export VSI590K_DATA_ROOT=${VSI590K_DATA_ROOT:-${SPACE_ROOT}/data/}

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
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to none \
    --use_geometry_inputs True \
    --use_geometry_encoder True \
    --use_feature_fusion_module True \
    --use_patch_size_alin True \
    --geometry_deepstack_indexes_pro \"7:0,10:1,13:2,16:3,19:4,22:5\" \
    --use_deepstack_importance_gate all \
    --use_deepstack_global_gate all \
    --geometry_encoder_type vggt \
    --geometry_encoder_path ${VGGT_PATH} \
    --use_mope True \
    --mope_checkpoint_path ${MOPE_CKPT_PATH} \
    --mope_encoder_path ${MOPE_CODE_PATH} \
    --mope_all_frames 8 \
    --group_by_modality_length True"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/e02a_mope_add_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-02a Training (MODEL_SIZE=${MODEL_SIZE}) ==="
echo "Output : ${output_dir}"
echo "Log    : ${LOG_FILE}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
