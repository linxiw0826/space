#!/usr/bin/env bash
set -euo pipefail

: "${MOPE_NEW_EXPERIMENT:?wrapper must set MOPE_NEW_EXPERIMENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# Load the per-server output/log roots before deriving experiment paths.
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"
MODEL_SIZE="${MODEL_SIZE:-4b}"
[[ "${MODEL_SIZE}" == "4b" ]] || { echo "Only the verified 4b recipe is enabled" >&2; exit 2; }
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MISSING="${MOPE_NEW_ALLOW_MISSING_ASSETS:-0}"
OUTPUT_ROOT="${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${SPACE_LOG_ROOT:-${SPACE_ROOT}/logs}/train}"
MOPE_NEW_SOURCE_ROOT="${MOPE_NEW_SOURCE_ROOT:-${SPACE_ROOT}/refs/mope-jepa-native-final515k}"
MOPE_NEW_CKPT="${MOPE_NEW_CKPT:-/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-50.pth}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${OUTPUT_ROOT}/train/guide_reproduced/4b}"
VGGT_PATH="${VGGT_PATH:-/data2/wlx/models/VGGT-1B}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29517}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SPACE_ROOT}/src:${SPACE_ROOT}:${PYTHONPATH:-}"
# Keep the legacy GUIDE manifest untouched.  The final515k-specific copy adds
# full-video sidecars used only by the frozen MoPE encoder.
export VSI590K_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_final515k.json}"
export VSI590K_VIDEO_ANN="${VSI590K_VIDEO_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_video_590k.json}"
export VSI590K_DATA_ROOT="${VSI590K_DATA_ROOT:-/data2/wlx/data/vsi590k_processed}"

case "${MOPE_NEW_EXPERIMENT}" in
  e02c-new)
    OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/train/e02c_mope_new_crossattn_joint_4b}"
    TUNE_LLM=True
    GRAD_ACCUM="${GRAD_ACCUM:-6}"
    WARMSTART=""
    ;;
  *) echo "Unknown MoPE-new experiment: ${MOPE_NEW_EXPERIMENT}" >&2; exit 2 ;;
esac

EFFECTIVE_BATCH=$((2 * NPROC_PER_NODE * GRAD_ACCUM))
[[ "${EFFECTIVE_BATCH}" == "48" ]] || {
  echo "E-02c-new requires effective global batch 48; got 2 x ${NPROC_PER_NODE} x ${GRAD_ACCUM} = ${EFFECTIVE_BATCH}" >&2
  exit 2
}

LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "${OUTPUT_DIR}")_$(date +%Y%m%d_%H%M%S).log}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  case "$(realpath -m "${RESUME_FROM_CHECKPOINT}")" in
    "$(realpath -m "${OUTPUT_DIR}")"/*) ;;
    *) echo "Resume must be inside this experiment's output directory" >&2; exit 2 ;;
  esac
elif [[ -d "${OUTPUT_DIR}" ]] && find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 \
    \( -type d -name 'checkpoint-*' -o -type f -name 'model*.safetensors' \
       -o -type f -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
  echo "Existing training artifacts found in ${OUTPUT_DIR}; refusing implicit resume." >&2
  echo "Move/archive the old output directory for a fresh final515k run, or set RESUME_FROM_CHECKPOINT explicitly." >&2
  exit 2
fi

if [[ "${ALLOW_MISSING}" != "1" ]]; then
  [[ -f "${MOPE_NEW_CKPT}" ]] || { echo "Missing MoPE-new checkpoint: ${MOPE_NEW_CKPT}" >&2; exit 2; }
  [[ -d "${GUIDE_CKPT_PATH}" ]] || { echo "Missing GUIDE checkpoint: ${GUIDE_CKPT_PATH}" >&2; exit 2; }
  [[ -d "${VGGT_PATH}" ]] || { echo "Missing VGGT: ${VGGT_PATH}" >&2; exit 2; }
  [[ -f "${MOPE_NEW_SOURCE_ROOT}/models/native_mope.py" ]] || { echo "Incomplete MoPE-new source" >&2; exit 2; }
  [[ -f "${VSI590K_SPAR_ANN}" ]] || { echo "Missing final515k SPAR manifest: ${VSI590K_SPAR_ANN}" >&2; exit 2; }
fi

COMMAND=(python -m torch.distributed.run "--nproc_per_node=${NPROC_PER_NODE}" "--master_port=${MASTER_PORT}"
  "${SPACE_ROOT}/src/train_framework/train_space_mope_new.py"
  --deepspeed "${SPACE_ROOT}/configs/zero2.json"
  --model_name_or_path "${GUIDE_CKPT_PATH}"
  --dataset_use vsi590k_spar --data_flatten False
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm "${TUNE_LLM}"
  --optim adamw_torch --bf16 --output_dir "${OUTPUT_DIR}"
  --num_train_epochs 1 --per_device_train_batch_size 2
  --per_device_eval_batch_size 4 --gradient_accumulation_steps "${GRAD_ACCUM}"
  --max_pixels 268324 --min_pixels 8192 --eval_strategy no
  --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --learning_rate 1e-5 --weight_decay 0.01 --warmup_ratio 0.03
  --max_grad_norm 1 --lr_scheduler_type cosine --logging_steps 1
  --model_max_length 12800 --gradient_checkpointing False
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" --report_to none
  --use_geometry_inputs True --use_geometry_encoder True
  --use_feature_fusion_module True --use_patch_size_alin True
  --geometry_deepstack_indexes_pro 7:0,10:1,13:2,16:3,19:4,22:5
  --use_deepstack_importance_gate all --use_deepstack_global_gate all
  --geometry_encoder_type vggt --geometry_encoder_path "${VGGT_PATH}"
  --use_mope True --mope_fusion_mode crossattn
  --mope_checkpoint_path "${MOPE_NEW_CKPT}" --mope_all_frames 16
  --mope_new_experiment "${MOPE_NEW_EXPERIMENT}"
  --mope_new_source_root "${MOPE_NEW_SOURCE_ROOT}"
  --mope_new_groups 4 --mope_new_frames_per_group 4
  --mope_new_input_size 224 --mope_new_pool_mode temporal
  --group_by_modality_length True)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  COMMAND+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
else
  # Transformers otherwise auto-discovers checkpoint-* in output_dir.  That is
  # unsafe when an experiment name is intentionally reused for a new MoPE
  # architecture/contract.
  COMMAND+=(--overwrite_output_dir)
fi

echo "Experiment=${MOPE_NEW_EXPERIMENT} train_llm=${TUNE_LLM} train_projector=True train_mope=False grad_accum=${GRAD_ACCUM} effective_batch=${EFFECTIVE_BATCH}"
echo "MoPE=${MOPE_NEW_CKPT} architecture=dense8+moe8/top1/shared1 pos=3d_sincos frames=16 sampling=4x4 input=224 pool=temporal expected=[B,8,768]"
echo "Output=${OUTPUT_DIR} warmstart=${WARMSTART:-none} resume=${RESUME_FROM_CHECKPOINT:-none}"
echo "DataLoader workers/rank=${DATALOADER_NUM_WORKERS} total_workers=$((DATALOADER_NUM_WORKERS * NPROC_PER_NODE)) save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT}"
echo "Log=${LOG_FILE}"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec "${COMMAND[@]}" > "${LOG_FILE}" 2>&1
