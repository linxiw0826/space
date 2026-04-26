#!/bin/bash
# =============================================================================
# Evaluate E-02a (MoPE add fusion) checkpoint on VSIBench
#
# Usage:
#   bash eval_e02a_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Arguments (optional — derived from MODEL_SIZE when omitted):
#   CKPT_PATH  — path to the E-02a checkpoint directory to evaluate
#   EXP_NAME   — short name for this evaluation run, used for output dir
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — model size to evaluate: "4b" or "8b" (default: 4b)
#   SPACE_ROOT           — project root on the execution server
#   VSIBENCH_VIDEO_ROOT  — directory containing VSIBench video files
#   VSIBENCH_JSONL       — path to vsibench test.jsonl annotation file
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for log files (default: SPACE_ROOT/logs/eval)
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3,4,5)
#   NUM_PROCESSES        — number of accelerate processes (default: 6)
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Project root — can be overridden via env var
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}

# ---------------------------------------------------------------------------
# MODEL_SIZE switch: 4b / 8b
# Sets default CKPT_PATH and EXP_NAME based on model size.
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

case "${MODEL_SIZE}" in
    4b)
        DEFAULT_CKPT_PATH="/home/nvme03/wlx/Space_sensing/output/train/e02a_mope_add_4b"
        DEFAULT_EXP_NAME="e02a_mope_4b"
        ;;
    8b)
        DEFAULT_CKPT_PATH="/home/nvme03/wlx/Space_sensing/output/train/e02a_mope_add_8b"
        DEFAULT_EXP_NAME="e02a_mope_8b"
        ;;
    *)
        echo "ERROR: MODEL_SIZE must be '4b' or '8b', got '${MODEL_SIZE}'"
        exit 1
        ;;
esac

# Positional args override the model-size defaults
CKPT_PATH=${1:-${DEFAULT_CKPT_PATH}}
EXP_NAME=${2:-${DEFAULT_EXP_NAME}}

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
GUIDE_LMMS_EVAL=${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}
GUIDE_TRAIN_ROOT=${SPACE_ROOT}/src

# VSIBench data paths
VSIBENCH_VIDEO_ROOT=${VSIBENCH_VIDEO_ROOT:-/home/nvme01/wlx/Space_sensing/data/VSIBench}
VSIBENCH_JSONL=${VSIBENCH_JSONL:-/home/nvme01/wlx/Space_sensing/data/VSIBench/test.jsonl}

# Results output
RESULTS_DIR=${RESULTS_DIR:-/home/nvme03/wlx/Space_sensing/output/eval/vsibench}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# Log file — independent log directory, tee'd to stdout
LOG_DIR=${LOG_DIR:-/home/nvme03/wlx/Space_sensing/logs/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e02a_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NUM_PROCESSES=${NUM_PROCESSES:-6}
MAIN_PORT=${MAIN_PORT:-$(shuf -i 20001-29999 -n 1)}

# ---------------------------------------------------------------------------
# LMMS_EVAL_PLUGINS: register qwen3_vl_mope model type.
# SPACE_ROOT is added to PYTHONPATH so that "src.eval" is importable as a
# Python package (src/eval/__init__.py + src/eval/models/__init__.py).
# ---------------------------------------------------------------------------
export LMMS_EVAL_PLUGINS="src.eval"
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${GUIDE_TRAIN_ROOT}:${SPACE_ROOT}:${PYTHONPATH}"
# Note: mope (src/vendor/mope) is added to sys.path by mope_encoder.py at import time.

# ---------------------------------------------------------------------------
# Disable NCCL NVLS (known to cause hangs with multi-GPU lmms-eval)
# ---------------------------------------------------------------------------
export NCCL_NVLS_ENABLE=0

# ---------------------------------------------------------------------------
# Model args — MoPE model (qwen3_vl_mope), mope_all_frames=8
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2,mope_all_frames=8"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation (E-02a MoPE add fusion) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "JSONL      : ${VSIBENCH_JSONL}"
echo "Plugin     : ${LMMS_EVAL_PLUGINS}"
echo "Processes  : ${NUM_PROCESSES}  Port: ${MAIN_PORT}"
echo "==================================================="

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "Starting accelerate launch..."

cd "${GUIDE_LMMS_EVAL}"

accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PORT} \
    -m lmms_eval \
    --model qwen3_vl_mope \
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
echo "=== Log saved to: ${LOG_FILE} ==="
