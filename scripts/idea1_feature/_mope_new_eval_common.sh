#!/usr/bin/env bash
set -euo pipefail

: "${MOPE_NEW_EXPERIMENT:?wrapper must set MOPE_NEW_EXPERIMENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
# Load per-server roots before deriving checkpoint, result, and log paths.
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"
source "${SPACE_ROOT}/scripts/idea1_feature/_mope_new_eval_lib.sh"
MODEL_SIZE="${MODEL_SIZE:-4b}"
[[ "${MODEL_SIZE}" == "4b" ]] || { echo "Only the verified 4b recipe is enabled" >&2; exit 2; }
DRY_RUN="${DRY_RUN:-0}"
SMOKE_MODE="${SMOKE_MODE:-0}"
SMOKE_DECODE_LIMIT="${SMOKE_DECODE_LIMIT:-10}"
ALLOW_MISSING="${MOPE_NEW_ALLOW_MISSING_ASSETS:-0}"
OUTPUT_ROOT="${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}"
LOG_DIR="${LOG_DIR:-${SPACE_LOG_ROOT:-${SPACE_ROOT}/logs}/eval}"
MOPE_NEW_SOURCE_ROOT="${MOPE_NEW_SOURCE_ROOT:-${SPACE_ROOT}/refs/mope-jepa-native-final515k}"
MOPE_NEW_CKPT="${MOPE_NEW_CKPT:-/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-50.pth}"
GUIDE_LMMS_EVAL="${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}"
VSIBENCH_VIDEO_ROOT="${VSIBENCH_VIDEO_ROOT:-/data2/wlx/data/VSIBench}"
VSIBENCH_JSONL="${VSIBENCH_JSONL:-${VSIBENCH_VIDEO_ROOT}/test.jsonl}"
export VSIBENCH_VIDEO_ROOT VSIBENCH_JSONL

case "${MOPE_NEW_EXPERIMENT}" in
  e02c-new) NAME=e02c_mope_new_crossattn_joint_4b ;;
  *) echo "final515k eval only supports e02c-new; old E-00b/E-03a wrappers are historical" >&2; exit 2 ;;
esac
REQUESTED_CKPT_PATH="${CKPT_PATH:-${OUTPUT_ROOT}/train/${NAME}}"
CKPT_PATH="${REQUESTED_CKPT_PATH}"
FORMAL_RESULTS_DIR="${OUTPUT_ROOT}/eval/vsibench/${NAME}"
if [[ "${SMOKE_MODE}" == "1" ]]; then
  [[ "${SMOKE_DECODE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    echo "SMOKE_DECODE_LIMIT must be a positive integer" >&2
    exit 2
  }
  RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/eval/smoke/vsibench/${NAME}}"
  mope_assert_smoke_output_isolated \
    "${RESULTS_DIR}" "${FORMAL_RESULTS_DIR}" "${OUTPUT_ROOT}/eval/smoke" || {
    echo "Invalid VSI-Bench smoke output: ${RESULTS_DIR}" >&2
    exit 2
  }
else
  [[ "${SMOKE_MODE}" == "0" ]] || { echo "SMOKE_MODE must be 0 or 1" >&2; exit 2; }
  RESULTS_DIR="${RESULTS_DIR:-${FORMAL_RESULTS_DIR}}"
fi
RUN_ROOT="${RESULTS_DIR}.work.$$"
TASK_DIR="${RUN_ROOT}/task_config"
RUN_OUTPUT_DIR="${RUN_ROOT}/lmms_output"
if [[ "${SMOKE_MODE}" == "1" ]]; then
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/smoke/${NAME}_vsibench_smoke_$(date +%Y%m%d_%H%M%S).log}"
  EVAL_JSONL="${RUN_ROOT}/vsibench_smoke.jsonl"
  EXPECTED_SAMPLE_COUNT=10
  DATA_PREFLIGHT_REPORT="${OUTPUT_ROOT}/audit/mope_final515k_eval/vsibench_data_preflight.json"
else
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/${NAME}_vsibench_$(date +%Y%m%d_%H%M%S).log}"
  EVAL_JSONL="${VSIBENCH_JSONL}"
  EXPECTED_SAMPLE_COUNT=5130
fi
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29527}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ "${ALLOW_MISSING}" != "1" ]]; then
  CKPT_PATH="$(mope_resolve_complete_hf_checkpoint "${REQUESTED_CKPT_PATH}")" || {
    echo "Invalid experiment checkpoint: ${REQUESTED_CKPT_PATH}" >&2
    exit 2
  }
  [[ -f "${MOPE_NEW_CKPT}" ]] || { echo "Missing MoPE-new checkpoint: ${MOPE_NEW_CKPT}" >&2; exit 2; }
  [[ -d "${MOPE_NEW_SOURCE_ROOT}" ]] || { echo "Missing MoPE-new source: ${MOPE_NEW_SOURCE_ROOT}" >&2; exit 2; }
  [[ -f "${VSIBENCH_JSONL}" ]] || { echo "Missing VSI-Bench annotations: ${VSIBENCH_JSONL}" >&2; exit 2; }
fi

MODEL_ARGS="pretrained=${CKPT_PATH},mope_checkpoint_path=${MOPE_NEW_CKPT},mope_source_root=${MOPE_NEW_SOURCE_ROOT},mope_all_frames=16,mope_groups=4,mope_frames_per_group=4,mope_input_size=224,mope_pool_mode=temporal,max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"
COMMAND=(accelerate launch "--num_processes=${NUM_PROCESSES}" "--main_process_port=${MAIN_PORT}"
  -m lmms_eval --model qwen3_vl_mope_new_crossattn --model_args "${MODEL_ARGS}"
  --include_path "${TASK_DIR}" --tasks vsibench_mope_new
  --batch_size 1 --log_samples --log_samples_suffix "${NAME}"
  --output_path "${RUN_OUTPUT_DIR}")

echo "Mode=$([[ "${SMOKE_MODE}" == "1" ]] && echo smoke || echo full)"
echo "Experiment=${MOPE_NEW_EXPERIMENT} requested_checkpoint=${REQUESTED_CKPT_PATH}"
echo "Effective_checkpoint=${CKPT_PATH}"
echo "Model=qwen3_vl_mope_new_crossattn MoPE=${MOPE_NEW_CKPT}"
echo "frames=16 sampling=4x4 pos=3d_sincos input=224 pool=temporal expected=[B,8,768]"
echo "VSI-Bench=${VSIBENCH_JSONL} video_root=${VSIBENCH_VIDEO_ROOT} output=${RESULTS_DIR}"
if [[ "${SMOKE_MODE}" == "1" ]]; then
  echo "Smoke coverage=all 10 question types total=10 decode_limit=${SMOKE_DECODE_LIMIT}"
  echo "Data_preflight_report=${DATA_PREFLIGHT_REPORT}"
fi
echo "Log=${LOG_FILE}"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ "${DRY_RUN}" == "1" ]] && exit 0
export LMMS_EVAL_PLUGINS=src.mope_new_eval_plugin
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${SPACE_ROOT}/src:${SPACE_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${RESULTS_DIR}")" "${TASK_DIR}" "${RUN_OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"
cleanup_run_root() {
  if [[ -n "${BACKUP_RESULTS:-}" && -e "${BACKUP_RESULTS}" && ! -e "${RESULTS_DIR}" ]]; then
    mv -- "${BACKUP_RESULTS}" "${RESULTS_DIR}"
  fi
  rm -rf -- "${RUN_ROOT}"
  [[ -z "${STAGED_RESULTS:-}" ]] || rm -rf -- "${STAGED_RESULTS}"
}
trap cleanup_run_root EXIT
if [[ "${SMOKE_MODE}" == "1" ]]; then
  mkdir -p "$(dirname "${DATA_PREFLIGHT_REPORT}")"
  python "${SPACE_ROOT}/scripts/preprocess/preflight_mope_final515k_eval_data.py" \
    --dataset vsibench \
    --annotation "${VSIBENCH_JSONL}" \
    --video-root "${VSIBENCH_VIDEO_ROOT}" \
    --expected-rows 5130 \
    --expected-videos 288 \
    --decode sample \
    --decode-limit "${SMOKE_DECODE_LIMIT}" \
    --smoke-output "${EVAL_JSONL}" \
    --report "${DATA_PREFLIGHT_REPORT}"
fi
cp "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/utils.py" "${TASK_DIR}/utils.py"
sed -e "s#^    test: .*#    test: ${EVAL_JSONL}#" \
    -e 's/^task: vsibench$/task: vsibench_mope_new/' \
    "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/vsibench.yaml" \
    > "${TASK_DIR}/vsibench.yaml"
cd "${GUIDE_LMMS_EVAL}"
exec > >(tee "${LOG_FILE}") 2>&1
"${COMMAND[@]}"

mapfile -t RESULT_FILES < <(find "${RUN_OUTPUT_DIR}" -type f -name '*_results.json' -print)
mapfile -t SAMPLE_FILES < <(find "${RUN_OUTPUT_DIR}" -type f -name '*_samples_vsibench_mope_new.jsonl' -print)
[[ "${#RESULT_FILES[@]}" -eq 1 ]] || {
  echo "Expected exactly one aggregated result JSON, found ${#RESULT_FILES[@]}" >&2
  exit 3
}
[[ "${#SAMPLE_FILES[@]}" -eq 1 ]] || {
  echo "Expected exactly one samples JSONL, found ${#SAMPLE_FILES[@]}" >&2
  exit 3
}
[[ -s "${RESULT_FILES[0]}" && -s "${SAMPLE_FILES[0]}" ]] || {
  echo "Evaluation produced an empty result artifact" >&2
  exit 3
}
python -m json.tool "${RESULT_FILES[0]}" >/dev/null
mope_validate_jsonl_count "${SAMPLE_FILES[0]}" "${EXPECTED_SAMPLE_COUNT}"

# Build the complete public result directory before touching an older valid run.
# The rename-based promotion prevents a failed eval from destroying prior results.
STAGED_RESULTS="${RESULTS_DIR}.staged.$$"
BACKUP_RESULTS="${RESULTS_DIR}.backup.$$"
mkdir -p "${STAGED_RESULTS}"
cp -- "${RESULT_FILES[0]}" "${STAGED_RESULTS}/${NAME}_results.json"
cp -- "${SAMPLE_FILES[0]}" "${STAGED_RESULTS}/${NAME}_samples.jsonl"

if [[ -e "${RESULTS_DIR}" ]]; then
  mv -- "${RESULTS_DIR}" "${BACKUP_RESULTS}"
fi
if mv -- "${STAGED_RESULTS}" "${RESULTS_DIR}"; then
  rm -rf -- "${BACKUP_RESULTS}"
else
  [[ ! -e "${RESULTS_DIR}" && -e "${BACKUP_RESULTS}" ]] && \
    mv -- "${BACKUP_RESULTS}" "${RESULTS_DIR}"
  exit 3
fi

echo "Results=${RESULTS_DIR}/${NAME}_results.json"
echo "Samples=${RESULTS_DIR}/${NAME}_samples.jsonl"
