#!/usr/bin/env bash
set -euo pipefail

: "${MOPE_NEW_EXPERIMENT:?wrapper must set MOPE_NEW_EXPERIMENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"

MODEL_SIZE="${MODEL_SIZE:-4b}"
[[ "${MODEL_SIZE}" == "4b" ]] || { echo "Only the verified 4b recipe is enabled" >&2; exit 2; }
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MISSING="${MOPE_NEW_ALLOW_MISSING_ASSETS:-0}"
OUTPUT_ROOT="${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${SPACE_LOG_ROOT:-${SPACE_ROOT}/logs}/eval}"
MOPE_NEW_SOURCE_ROOT="${MOPE_NEW_SOURCE_ROOT:-${SPACE_ROOT}/refs/mope-jepa-native-final515k}"
MOPE_NEW_CKPT="${MOPE_NEW_CKPT:-/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-50.pth}"
GUIDE_LMMS_EVAL="${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}"
VLM4D_VIDEO_ROOT="${VLM4D_VIDEO_ROOT:-/data2/wlx/data/VLM4D}"
VLM4D_JSONL="${VLM4D_JSONL:-${VLM4D_VIDEO_ROOT}/QA/real_mc.json}"
export VLM4D_VIDEO_ROOT

case "${MOPE_NEW_EXPERIMENT}" in
  e02c-new) NAME=e02c_mope_new_crossattn_joint_4b ;;
  *) echo "final515k VLM4D eval only supports e02c-new; old wrappers are historical" >&2; exit 2 ;;
esac

CKPT_PATH="${CKPT_PATH:-${OUTPUT_ROOT}/train/${NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/eval/vlm4d/${NAME}}"
TASK_DIR="${RESULTS_DIR}/task_config"
TASK_NAME="vlm4d_real_mc_mope_new"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_vlm4d_$(date +%Y%m%d_%H%M%S).log}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29529}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ "${ALLOW_MISSING}" != "1" ]]; then
  [[ -d "${CKPT_PATH}" ]] || { echo "Missing experiment checkpoint: ${CKPT_PATH}" >&2; exit 2; }
  [[ -f "${MOPE_NEW_CKPT}" ]] || { echo "Missing MoPE-new checkpoint: ${MOPE_NEW_CKPT}" >&2; exit 2; }
  [[ -d "${MOPE_NEW_SOURCE_ROOT}" ]] || { echo "Missing MoPE-new source: ${MOPE_NEW_SOURCE_ROOT}" >&2; exit 2; }
  [[ -f "${VLM4D_JSONL}" ]] || { echo "Missing VLM4D real_mc annotations: ${VLM4D_JSONL}" >&2; exit 2; }
fi

MODEL_ARGS="pretrained=${CKPT_PATH},mope_checkpoint_path=${MOPE_NEW_CKPT},mope_source_root=${MOPE_NEW_SOURCE_ROOT},mope_all_frames=16,mope_groups=4,mope_frames_per_group=4,mope_input_size=224,mope_pool_mode=temporal,max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"
COMMAND=(accelerate launch "--num_processes=${NUM_PROCESSES}" "--main_process_port=${MAIN_PORT}"
  -m lmms_eval --model qwen3_vl_mope_new_crossattn --model_args "${MODEL_ARGS}"
  --include_path "${TASK_DIR}" --tasks "${TASK_NAME}"
  --batch_size 1 --log_samples --log_samples_suffix "${NAME}_vlm4d"
  --output_path "${RESULTS_DIR}" --force_simple)

echo "Experiment=${MOPE_NEW_EXPERIMENT} checkpoint=${CKPT_PATH}"
echo "Model=qwen3_vl_mope_new_crossattn MoPE=${MOPE_NEW_CKPT}"
echo "frames=16 sampling=4x4 pos=3d_sincos input=224 pool=temporal expected=[B,8,768]"
echo "VLM4D real_mc=${VLM4D_JSONL} video_root=${VLM4D_VIDEO_ROOT} output=${RESULTS_DIR}"
echo "Log=${LOG_FILE}"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0

export LMMS_EVAL_PLUGINS=src.mope_new_eval_plugin
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${SPACE_ROOT}/src:${SPACE_ROOT}:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
mkdir -p "${RESULTS_DIR}" "${TASK_DIR}" "${LOG_DIR}"
cp "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vlm4d/utils.py" "${TASK_DIR}/utils.py"
sed -e "s#^    test: .*#    test: ${VLM4D_JSONL}#" \
    -e "s/^task: vlm4d$/task: ${TASK_NAME}/" \
    "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vlm4d/vlm4d.yaml" \
    > "${TASK_DIR}/vlm4d_real_mc.yaml"
cd "${GUIDE_LMMS_EVAL}"
exec > >(tee "${LOG_FILE}") 2>&1
exec "${COMMAND[@]}"
