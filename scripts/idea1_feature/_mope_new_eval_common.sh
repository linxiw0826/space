#!/usr/bin/env bash
set -euo pipefail

: "${MOPE_NEW_EXPERIMENT:?wrapper must set MOPE_NEW_EXPERIMENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# Load per-server roots before deriving checkpoint, result, and log paths.
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"
MODEL_SIZE="${MODEL_SIZE:-4b}"
[[ "${MODEL_SIZE}" == "4b" ]] || { echo "Only the verified 4b recipe is enabled" >&2; exit 2; }
DRY_RUN="${DRY_RUN:-0}"
ALLOW_MISSING="${MOPE_NEW_ALLOW_MISSING_ASSETS:-0}"
OUTPUT_ROOT="${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${SPACE_LOG_ROOT:-${SPACE_ROOT}/logs}/eval}"
MOPE_NEW_SOURCE_ROOT="${MOPE_NEW_SOURCE_ROOT:-${SPACE_ROOT}/refs/mope_new}"
MOPE_NEW_CKPT="${MOPE_NEW_CKPT:-/data2/mope-jepa-assets/jepa_checkpoints/native_mope8_dense4_moe4_top2_wisa_dyn_3dpos/checkpoint-50.pth}"
GUIDE_LMMS_EVAL="${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}"
VSIBENCH_VIDEO_ROOT="${VSIBENCH_VIDEO_ROOT:-/data2/wlx/data/VSIBench}"
VSIBENCH_JSONL="${VSIBENCH_JSONL:-${VSIBENCH_VIDEO_ROOT}/test.jsonl}"
export VSIBENCH_VIDEO_ROOT VSIBENCH_JSONL

case "${MOPE_NEW_EXPERIMENT}" in
  e00b-new) NAME=e00b_mope_new_projector_only_4b ;;
  e02c-new) NAME=e02c_mope_new_crossattn_joint_4b ;;
  e03a-new) NAME=e03a_mope_new_crossattn_two_stage_4b ;;
  *) echo "Unknown MoPE-new experiment: ${MOPE_NEW_EXPERIMENT}" >&2; exit 2 ;;
esac
CKPT_PATH="${CKPT_PATH:-${OUTPUT_ROOT}/train/${NAME}}"
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/eval/vsibench/${NAME}}"
TASK_DIR="${RESULTS_DIR}/task_config"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_vsibench_$(date +%Y%m%d_%H%M%S).log}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29527}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ "${ALLOW_MISSING}" != "1" ]]; then
  [[ -d "${CKPT_PATH}" ]] || { echo "Missing experiment checkpoint: ${CKPT_PATH}" >&2; exit 2; }
  [[ -f "${MOPE_NEW_CKPT}" ]] || { echo "Missing MoPE-new checkpoint: ${MOPE_NEW_CKPT}" >&2; exit 2; }
  [[ -f "${VSIBENCH_JSONL}" ]] || { echo "Missing VSI-Bench annotations: ${VSIBENCH_JSONL}" >&2; exit 2; }
fi

MODEL_ARGS="pretrained=${CKPT_PATH},mope_checkpoint_path=${MOPE_NEW_CKPT},mope_source_root=${MOPE_NEW_SOURCE_ROOT},mope_all_frames=16,mope_sampling_rate=4,mope_input_size=224,mope_pool_mode=none,max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"
COMMAND=(accelerate launch "--num_processes=${NUM_PROCESSES}" "--main_process_port=${MAIN_PORT}"
  -m lmms_eval --model qwen3_vl_mope_new_crossattn --model_args "${MODEL_ARGS}"
  --include_path "${TASK_DIR}" --tasks vsibench_mope_new
  --batch_size 1 --log_samples --log_samples_suffix "${NAME}"
  --output_path "${RESULTS_DIR}")

echo "Experiment=${MOPE_NEW_EXPERIMENT} checkpoint=${CKPT_PATH}"
echo "Model=qwen3_vl_mope_new_crossattn MoPE=${MOPE_NEW_CKPT}"
echo "frames=16 sampling_rate=4 input=224 pool=none expected=[B,1568,768]"
echo "VSI-Bench=${VSIBENCH_JSONL} video_root=${VSIBENCH_VIDEO_ROOT} output=${RESULTS_DIR}"
echo "Log=${LOG_FILE}"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0
export LMMS_EVAL_PLUGINS=src.mope_new_eval_plugin
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${SPACE_ROOT}/src:${SPACE_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULTS_DIR}" "${TASK_DIR}" "${LOG_DIR}"
cp "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/utils.py" "${TASK_DIR}/utils.py"
sed -e "s#^    test: .*#    test: ${VSIBENCH_JSONL}#" \
    -e 's/^task: vsibench$/task: vsibench_mope_new/' \
    "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/vsibench.yaml" \
    > "${TASK_DIR}/vsibench.yaml"
cd "${GUIDE_LMMS_EVAL}"
exec > >(tee "${LOG_FILE}") 2>&1
exec "${COMMAND[@]}"
