#!/bin/bash
# =============================================================================
# E-01: GUIDE fine-tune (no MoPE) — baseline
# Supports 4B and 8B via MODEL_SIZE env var (default: 4b)
#
# Usage:
#   bash train_e01_guide.sh                   # 4B (default)
#   MODEL_SIZE=8b bash train_e01_guide.sh     # 8B
#
# Log directory can be overridden via LOG_DIR env var.
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Distributed training env
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Single-node training only; multi-node not supported.
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

# ---------------------------------------------------------------------------
# Root paths (overridable via env vars)
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}
GUIDE_ROOT="${SPACE_ROOT}/src"
MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

# ---------------------------------------------------------------------------
# Model size selection (4b | 8b)
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

QWEN3_VL_4B_PATH=${QWEN3_VL_4B_PATH:-/home/nvme03/wlx/Space_sensing/models/Qwen3-VL-4B-Instruct}
QWEN3_VL_8B_PATH=${QWEN3_VL_8B_PATH:-/home/nvme03/wlx/Space_sensing/models/Qwen3-VL-8B-Instruct}
VGGT_PATH=${VGGT_PATH:-/home/nvme03/wlx/Space_sensing/models/VGGT-1B}
GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-/home/nvme03/wlx/Space_sensing/models/guide_reproduced/4b}

CONFIGS_DIR="${SPACE_ROOT}/configs"

if [ "${MODEL_SIZE}" = "8b" ]; then
    # 8B: larger model requires ZeRO-3 to fit on 8×H800
    batch_size=1
    grad_accum_steps=16
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${CONFIGS_DIR}/zero3.json}
    output_dir="${OUTPUT_DIR:-/home/nvme03/wlx/Space_sensing/models/e01_guide_8b}"
    run_name="space_e01_guide_8b_lr1e-5"
elif [ "${MODEL_SIZE}" = "4b" ]; then
    batch_size=1
    grad_accum_steps=8
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${CONFIGS_DIR}/zero2.json}
    output_dir="${OUTPUT_DIR:-/home/nvme03/wlx/Space_sensing/models/e01_guide_4b}"
    run_name="space_e01_guide_4b_lr1e-5"
else
    echo "ERROR: MODEL_SIZE must be '4b' or '8b', got '${MODEL_SIZE}'" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Python path
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
export VSI590K_SPAR_ANN=${VSI590K_SPAR_ANN:-${SPACE_ROOT}/data/vsi590k_spar.json}
export VSI590K_VIDEO_ANN=${VSI590K_VIDEO_ANN:-${SPACE_ROOT}/data/vsi590k_video.json}
export VSI590K_DATA_ROOT=${VSI590K_DATA_ROOT:-${SPACE_ROOT}/data/}

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
fi

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
lr=1e-5

# ---------------------------------------------------------------------------
# Output & log directories
# ---------------------------------------------------------------------------
mkdir -p "${output_dir}"

LOG_DIR=${LOG_DIR:-/home/nvme03/wlx/Space_sensing/logs/train}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e01_guide_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
entry_file="${SPACE_ROOT}/src/train_framework/train_space.py"

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
    --use_mope False \
    --group_by_modality_length True"

echo "MODEL_SIZE=${MODEL_SIZE}  batch_size=${batch_size}  grad_accum=${grad_accum_steps}  deepspeed=${DEEPSPEED_CONFIG}"
echo "output_dir=${output_dir}"
echo "log → ${LOG_FILE}"

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
