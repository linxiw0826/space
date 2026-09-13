#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="1,3,5,6"
NPROC_PER_NODE="4"
PER_DEVICE_TRAIN_BATCH_SIZE="2"
GRAD_ACCUM="6"
LEARNING_RATE="1e-5"
LR_SCHEDULER_TYPE="cosine"
SPACE_ROOT="${SPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export PYTHONPATH="${SPACE_ROOT}/src:${SPACE_ROOT}:${PYTHONPATH:-}"
export VSI590K_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json}"
export VSI590K_VIDEO_ANN="${VSI590K_VIDEO_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_video_590k.json}"
export VSI590K_DATA_ROOT="${VSI590K_DATA_ROOT:-/data2/wlx/data/vsi590k_processed}"

[[ -f "${VSI590K_SPAR_ANN}" ]] || { echo "Missing quarter manifest: ${VSI590K_SPAR_ANN}" >&2; exit 2; }
[[ -d "${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/guide_reproduced/4b}" ]] || {
  echo "Missing baseline initialization checkpoint" >&2
  exit 2
}

OUTPUT_DIR="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/baseline_quarter_4b}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/guide_reproduced/4b}"
LOG_FILE="${LOG_FILE:-${SPACE_LOG_ROOT:-/data2/wlx/logs}/train/baseline_quarter_4b_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

COMMAND=(python -m torch.distributed.run "--nproc_per_node=${NPROC_PER_NODE}" "--master_port=${MASTER_PORT:-29511}"
  "${SPACE_ROOT}/src/train_framework/train_space.py"
  --deepspeed "${SPACE_ROOT}/configs/zero2.json"
  --model_name_or_path "${GUIDE_CKPT_PATH}" --dataset_use vsi590k_spar --data_flatten False
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True --optim adamw_torch --bf16
  --output_dir "${OUTPUT_DIR}" --num_train_epochs 1
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" --per_device_eval_batch_size 4
  --gradient_accumulation_steps "${GRAD_ACCUM}" --max_pixels 268324 --min_pixels 8192
  --eval_strategy no --save_strategy steps --save_steps 500 --save_total_limit 2
  --learning_rate "${LEARNING_RATE}" --weight_decay 0.01 --warmup_ratio 0.03
  --max_grad_norm 1 --lr_scheduler_type "${LR_SCHEDULER_TYPE}" --logging_steps 1
  --model_max_length 12800 --gradient_checkpointing False --dataloader_num_workers 2 --report_to none
  --use_geometry_inputs True --use_geometry_encoder True --use_feature_fusion_module True
  --use_patch_size_alin True --geometry_deepstack_indexes_pro 7:0,10:1,13:2,16:3,19:4,22:5
  --use_deepstack_importance_gate all --use_deepstack_global_gate all
  --geometry_encoder_type vggt --geometry_encoder_path "${VGGT_PATH:-/data2/wlx/models/VGGT-1B}"
  --use_mope False --group_by_modality_length True --overwrite_output_dir)

echo "Experiment=baseline-quarter init=${GUIDE_CKPT_PATH} manifest=${VSI590K_SPAR_ANN} lr=${LEARNING_RATE} scheduler=${LR_SCHEDULER_TYPE} effective_batch=$((2*4*6))"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
exec "${COMMAND[@]}" >"${LOG_FILE}" 2>&1
