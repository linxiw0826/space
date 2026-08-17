#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOPE_ROOT="${MOPE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MOPE_ASSETS_ROOT="${MOPE_ASSETS_ROOT:-$(cd "${MOPE_ROOT}/.." && pwd)/mope-jepa-assets}"
MOPE_JEPA_OUTPUT_ROOT="${MOPE_JEPA_OUTPUT_ROOT:-${MOPE_ASSETS_ROOT}/jepa_checkpoints}"
MOPE_DATA_LIST="${MOPE_DATA_LIST:-${MOPE_ASSETS_ROOT}/manifests/videos_plus_wisa_dyn_ssd.txt}"

MODE="${1:-train}"
GPUS="${2:-0}"
NUM_GPUS=$(awk -F, '{print NF}' <<<"${GPUS}")
MASTER_PORT="${MASTER_PORT:-30451}"

OUTPUT_NAME="${OUTPUT_NAME:-native_mope8_dense4_moe4_top2_wisa_dyn_3dpos}"
OUTPUT_DIR="${MOPE_JEPA_OUTPUT_ROOT}/${OUTPUT_NAME}"

test -f "${MOPE_DATA_LIST}" || { echo "Missing data manifest: ${MOPE_DATA_LIST}"; exit 1; }
mkdir -p "${OUTPUT_DIR}/log"

COMMON_ARGS=(
  --model native_mope_jepa_base_patch16_224
  --datasets_root "${MOPE_DATA_LIST}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${OUTPUT_DIR}/log"
  --num_frames 16
  --sampling_rate 4
  --input_size 224
  --tubelet_size 2
  --encoder_depth 8
  --dense_layers 4
  --num_routed_experts 4
  --candidate_k 2
  --num_shared_experts 1
  --router_score_func sigmoid
  --router_bias_update_speed 0.001
  --future_num_anchors 2
  --future_anchor_candidates 0,1,2,3,4,5,6
  --future_anchor_weights 1.35,1.25,1.15,1.0,0.9,0.8,0.7
  --sigreg_weight 0.3
  --predictor_dim 384
  --predictor_depth 6
  --predictor_num_heads 6
  --pos_embed_type 3d_sincos
  --predictor_pos_embed_type 3d_sincos
  --lr 1.5e-4
  --min_lr 1e-5
  --weight_decay 0.05
)

echo "Native MoPE root: ${MOPE_ROOT}"
echo "Assets root: ${MOPE_ASSETS_ROOT}"
echo "Data manifest: ${MOPE_DATA_LIST}"
echo "Output: ${OUTPUT_DIR}"

if [[ "${MODE}" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" python "${MOPE_ROOT}/run_native_jepa_pretraining.py" \
    "${COMMON_ARGS[@]}" \
    --batch_size 2 --epochs 1 --warmup_epochs 0 --num_workers 2 \
    --max_train_steps_per_epoch 2 --save_ckpt_freq 1 \
    2>&1 | tee "${OUTPUT_DIR}/smoke.log"
  exit 0
fi

if (( NUM_GPUS > 1 )); then
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    "${MOPE_ROOT}/run_native_jepa_pretraining.py" \
    "${COMMON_ARGS[@]}" \
    --batch_size 16 --epochs 100 --warmup_epochs 5 \
    --num_workers 4 --save_ckpt_freq 10 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  CUDA_VISIBLE_DEVICES="${GPUS}" python "${MOPE_ROOT}/run_native_jepa_pretraining.py" \
    "${COMMON_ARGS[@]}" \
    --batch_size 16 --epochs 100 --warmup_epochs 5 \
    --num_workers 4 --save_ckpt_freq 10 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
fi
