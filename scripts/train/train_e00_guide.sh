#!/bin/bash
# =============================================================================
# E-00: GUIDE 4B SR 复现训练
# 从 Qwen3-VL-4B-Instruct 开始，用 spar_234k + llava_hound_64k 训练。
# 产出：models/guide_reproduced/4b/  (~9h / 8×H800)
#
# Usage:
#   bash train_e00_guide.sh
#
# 关键路径可通过 env var 覆盖（见下方 PATH CONFIG 节）。
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Distributed training env
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

# ---------------------------------------------------------------------------
# PATH CONFIG — 全部可通过 env var 覆盖
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}
GUIDE_ROOT="${SPACE_ROOT}/src"
MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"
CONFIGS_DIR="${SPACE_ROOT}/configs"

# 模型路径 (nvme01)
QWEN3_VL_4B_PATH=${QWEN3_VL_4B_PATH:-/home/nvme01/wlx/Space_sensing/models/Qwen3-VL-4B-Instruct}
VGGT_PATH=${VGGT_PATH:-/home/nvme01/wlx/Space_sensing/models/VGGT-1B}

# 数据路径 (nvme01 guide_repro)
export SPAR_234K_ANN=${SPAR_234K_ANN:-/home/nvme01/wlx/Space_sensing/data/guide_repro/train/spar_234k.json}
export LLAVA_HOUND_64K_ANN=${LLAVA_HOUND_64K_ANN:-/home/nvme01/wlx/Space_sensing/data/guide_repro/train/llava_hound_64k.json}
export GUIDE_DATA_ROOT=${GUIDE_DATA_ROOT:-/home/nvme01/wlx/Space_sensing/data/guide_repro/media}

# 输出路径 (nvme03，空间更大)
output_dir=${OUTPUT_DIR:-/home/nvme03/wlx/Space_sensing/output/train/guide_reproduced/4b}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
lr=1e-5
batch_size=1
grad_accum_steps=11
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${CONFIGS_DIR}/zero2.json}
run_name="space_e00_guide_4b_sr_lr1e-5"

# ---------------------------------------------------------------------------
# Python path (src/train_framework 最优先，覆盖 refs 版本)
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
fi

# ---------------------------------------------------------------------------
# Output & log
# ---------------------------------------------------------------------------
mkdir -p "${output_dir}"

LOG_DIR=${LOG_DIR:-/home/nvme03/wlx/Space_sensing/logs/train}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e00_guide_4b_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
entry_file="${SPACE_ROOT}/src/train_framework/train_space.py"

args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${QWEN3_VL_4B_PATH} \
    --dataset_use llava_hound_64k,spar_234k \
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
    --geometry_deepstack_indexes_pro 7:0,10:1,13:2,16:3,19:4,22:5 \
    --use_deepstack_importance_gate all \
    --use_deepstack_global_gate all \
    --geometry_encoder_type vggt \
    --geometry_encoder_path ${VGGT_PATH} \
    --use_mope False \
    --group_by_modality_length True"

echo "=== E-00: GUIDE 4B SR ==="
echo "  base model : ${QWEN3_VL_4B_PATH}"
echo "  datasets   : llava_hound_64k, spar_234k"
echo "  output_dir : ${output_dir}"
echo "  deepspeed  : ${DEEPSPEED_CONFIG}"
echo "  log        : ${LOG_FILE}"
echo ""

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
