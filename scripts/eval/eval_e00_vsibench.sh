#!/bin/bash
# =============================================================================
# Evaluate E-00 (GUIDE pre-trained weights, zero-shot, no fine-tune) on VSIBench
#
# Usage:
#   bash eval_e00_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Arguments:
#   CKPT_PATH  — path to the GUIDE pre-trained checkpoint directory
#                (defaults based on MODEL_SIZE: 4b or 8b)
#   EXP_NAME   — short name for this evaluation run, used for output dir
#                (defaults based on MODEL_SIZE: e00_zeroshot_4b or e00_zeroshot_8b)
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — model size to evaluate: 4b (default) or 8b
#   SPACE_ROOT           — project root on execution server
#   VSIBENCH_VIDEO_ROOT  — directory containing VSIBench video files
#   VSIBENCH_JSONL       — path to vsibench test.jsonl annotation file
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for log files
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3,4,5)
#   NUM_PROCESSES        — number of accelerate processes (default: 6)
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Model size selection
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}

if [ "${MODEL_SIZE}" = "4b" ]; then
    CKPT_PATH=${1:-/home/nvme03/wlx/Space_sensing/output/train/guide_reproduced/4b}
    EXP_NAME=${2:-e00_zeroshot_4b}
elif [ "${MODEL_SIZE}" = "8b" ]; then
    CKPT_PATH=${1:-/home/nvme03/wlx/Space_sensing/output/train/guide_reproduced/8b}
    EXP_NAME=${2:-e00_zeroshot_8b}
else
    echo "ERROR: MODEL_SIZE must be '4b' or '8b', got '${MODEL_SIZE}'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
GUIDE_LMMS_EVAL=${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}
GUIDE_TRAIN_ROOT=${SPACE_ROOT}/src

# VSIBench data paths
# VSIBENCH_VIDEO_ROOT: directory that contains <dataset>/<scene_name>.mp4
# The vsibench utils.py resolves: ${VSIBENCH_VIDEO_ROOT}/<dataset>/<scene_name>.mp4
VSIBENCH_VIDEO_ROOT=${VSIBENCH_VIDEO_ROOT:-/home/nvme01/wlx/Space_sensing/data/VSIBench}
VSIBENCH_JSONL=${VSIBENCH_JSONL:-/home/nvme01/wlx/Space_sensing/data/VSIBench/test.jsonl}

# Results output
RESULTS_DIR=${RESULTS_DIR:-/home/nvme03/wlx/Space_sensing/output/eval/vsibench}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# Log directory (separate from results, as required)
LOG_DIR=${LOG_DIR:-/home/nvme03/wlx/Space_sensing/logs/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e00_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NUM_PROCESSES=${NUM_PROCESSES:-6}
MAIN_PORT=${MAIN_PORT:-$(shuf -i 20001-29999 -n 1)}

# ---------------------------------------------------------------------------
# PYTHONPATH: lmms-eval needs GUIDE's qwen-vl-finetune for model loading
# (qwen3_vl_my adds qwen-vl-finetune to sys.path itself, but be explicit)
# ---------------------------------------------------------------------------
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${GUIDE_TRAIN_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# Disable NCCL NVLS (known to cause hangs with multi-GPU lmms-eval)
# ---------------------------------------------------------------------------
export NCCL_NVLS_ENABLE=0

# ---------------------------------------------------------------------------
# Model args
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"

# ---------------------------------------------------------------------------
# Status output before launch
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation — E-00 Zero-Shot (${MODEL_SIZE}) ==="
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "JSONL      : ${VSIBENCH_JSONL}"
echo "Model size : ${MODEL_SIZE}"
echo "Processes  : ${NUM_PROCESSES}"
echo "GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "======================================================="

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
cd "${GUIDE_LMMS_EVAL}"

accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PORT} \
    -m lmms_eval \
    --model qwen3_vl_my \
    --model_args "${MODEL_ARGS}" \
    --tasks vsibench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${EXP_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --force_simple \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Evaluation complete. Results at: ${OUTPUT_PATH} ==="
