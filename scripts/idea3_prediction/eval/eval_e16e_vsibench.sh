#!/bin/bash
# =============================================================================
# Evaluate E-16e (LFP feed + predict + feed-causal mask) on VSIBench.
# Model type: qwen3_vl_mope_crossattn (MoPE-ON inference).
# (forked from eval_e16b_vsibench.sh — EXP_NAME / ckpt / output differ, PLUS
#  the CRITICAL feed-causal-mask env toggle below.)
#
# =============================================================================
# CRITICAL — feed causal-mask MUST be ON at eval (train/test consistency)
# =============================================================================
# E-16e is TRAINED with --mope_feed_causal_mask True: every image token of frame
# f attends ONLY to MoPE K/V of time-bins <= b_f (future-bin motion is masked
# with -inf). If we evaluate WITHOUT the mask, the feed cross-attn runs
# BIDIRECTIONAL — a train/test mismatch that leaks future-bin motion backward
# and makes the E-16 vs E-16e comparison INVALID. The eval plugin
# (src/eval/models/qwen3_vl_mope.py :: _read_feed_causal_mask) reads the env var
# MOPE_FEED_CAUSAL_MASK (truthy = 1/true) and, when set, patches the crossattn
# with feed_causal_mask=True. This script DEFAULTS it to 1 so E-16e can never be
# accidentally evaluated bidirectionally. DO NOT set it to 0 for E-16e.
# =============================================================================
export MOPE_FEED_CAUSAL_MASK=${MOPE_FEED_CAUSAL_MASK:-1}
#
# ATTRIBUTION RATIONALE (paper-2 Stage-1 eval design):
# E-16e is trained with --mope_feed_features=True (cross-attn residual injection
# ON), exactly like E-16, PLUS the feed-causal mask (training-only argument that
# shapes the trained cross-attn); the LFP predict head is TRAINING-ONLY (no
# head/loss at inference). So the MoPE-ON inference path
# (qwen3_vl_mope_crossattn) with the feed-causal mask ON is the correct
# evaluation mode — the projector weights load (strict=False keeps the matched
# keys) and the residual is injected exactly as in E-03a/E-16, but with the
# causal mask matching training.
#
# Paper-2 Stage-1 attribution table:
# | experiment | train feed | train LFP | train reg | causal-mask | eval model               | eval MoPE |
# |------------|:---------:|:---------:|:---------:|:-----------:|--------------------------|:---------:|
# | E-16       | ON        | ON        | OFF       | OFF         | qwen3_vl_mope_crossattn  | ON        |
# | E-16e      | ON        | ON        | OFF       | ON          | qwen3_vl_mope_crossattn  | ON        |  <-- THIS
#
# Usage:
#   bash eval_e16e_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — model size to evaluate: "4b" or "8b" (default: 4b)
#   MOPE_FEED_CAUSAL_MASK— feed causal mask at eval (default: 1 = ON; keep ON!)
#   SPACE_ROOT           — project root on the execution server
#   VSIBENCH_VIDEO_ROOT  — directory containing VSIBench video files
#   VSIBENCH_JSONL       — path to vsibench test.jsonl annotation file
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for log files (default: SPACE_ROOT/logs/eval)
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3)
#   NUM_PROCESSES        — number of accelerate processes (default: 4)
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../../_common/env/activate.sh"

# --- New-server runtime workarounds ---------------------------------------
# 保护性处理:剔除其它 conda 环境(videorepa)注入 LD_LIBRARY_PATH 的路径,
# 避免加载到错误的 libcudnn,确保使用 space 环境自带的正确 cuDNN(快路径)。
if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v 'videorepa' | paste -sd: -)"
fi

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
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e16e_feedcausal_4b"
        DEFAULT_EXP_NAME="e16e_feedcausal_4b"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e16e_feedcausal_8b"
        DEFAULT_EXP_NAME="e16e_feedcausal_8b"
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

# Inject the annotation path into the task yaml's dataset_kwargs.data_files.test
# line so that VSIBENCH_JSONL takes effect (the yaml hardcodes an old default).
# The data path key is nested under dataset_kwargs.data_files and is indented
# 4 spaces ("    test: "); the trailing space in the anchor avoids matching the
# unindented "test_split:" line.
VSIBENCH_TASK_YAML="${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/vsibench.yaml"
if [ -f "${VSIBENCH_TASK_YAML}" ]; then
    sed -i -E "s#^(    test: ).*#\1${VSIBENCH_JSONL}#" "${VSIBENCH_TASK_YAML}"
fi

# Results output
RESULTS_DIR=${RESULTS_DIR:-${SPACE_OUTPUT_ROOT}/eval/vsibench}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# Log file — independent log directory, tee'd to stdout
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e16e_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PORT=${MAIN_PORT:-$(shuf -i 20001-29999 -n 1)}

# ---------------------------------------------------------------------------
# LMMS_EVAL_PLUGINS: register qwen3_vl_mope_crossattn model type.
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
# Model args — MoPE cross-attention model (qwen3_vl_mope_crossattn), mope_all_frames=8
# E-16e's trained cross-attn projector loads (strict=False keeps matched keys);
# the LFP head keys are dropped (training-only). MoPE residual is injected, and
# the feed cross-attn runs with the causal mask (MOPE_FEED_CAUSAL_MASK=1).
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2,mope_all_frames=8"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation (E-16e LFP feed+predict+causal-mask, MoPE-ON inference) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "JSONL      : ${VSIBENCH_JSONL}"
echo "Plugin     : ${LMMS_EVAL_PLUGINS}"
echo "FeedCausal : MOPE_FEED_CAUSAL_MASK=${MOPE_FEED_CAUSAL_MASK}  (MUST be 1 for E-16e train/test consistency)"
echo "Processes  : ${NUM_PROCESSES}  Port: ${MAIN_PORT}"
echo "============================================================="

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "Starting accelerate launch..."

cd "${GUIDE_LMMS_EVAL}"

accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port ${MAIN_PORT} \
    -m lmms_eval \
    --model qwen3_vl_mope_crossattn \
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
