#!/bin/bash
# =============================================================================
# Evaluate E-15 (LFP prediction-only) on VSIBench.
# Model type: qwen3_vl_my (BASE GUIDE inference — VGGT ON, MoPE OFF).
#
# ATTRIBUTION RATIONALE (paper-2 Stage-1 2x2 eval design):
# E-15 is trained with --mope_feed_features=False (cross-attn residual
# injection OFF). The cross-attn projector's k/v/out_proj weights therefore
# never receive a feature-injection signal — they are NOT trained to inject
# anything useful. Loading such a checkpoint with the MoPE-on inference path
# (qwen3_vl_mope_crossattn) would inject the projector's untrained garbage
# residual into image_embeds and corrupt the LLM. Hence E-15 MUST be evaluated
# MoPE-OFF, i.e. as a plain base GUIDE model (--model qwen3_vl_my).
#
# Checkpoint loading (strict=False): the E-15 checkpoint contains extra keys
# for the MoPE encoder/projector and the LFP head (model._mope_projector.*,
# model._mope_lfp_head.*). When loaded as qwen3_vl_my these extra keys are
# silently dropped (HF from_pretrained ignores them), and ONLY the LLM +
# VGGT (geometry encoder) weights are loaded. The LLM weights are the ones
# actually reshaped by the LFP auxiliary objective during E-15 training, so
# this is exactly the comparison we want: "did the prediction-only training
# signal produce better LLM representations than E-01 (no MoPE at all)?"
#
# Paper-2 Stage-1 2x2 attribution table:
# | experiment | train feed | train LFP | eval model               | eval MoPE |
# |------------|:---------:|:---------:|--------------------------|:---------:|
# | E-01       | n/a       | n/a       | qwen3_vl_my              | OFF (base)|
# | E-03a      | ON        | OFF       | qwen3_vl_mope_crossattn  | ON        |
# | E-15       | OFF       | ON        | qwen3_vl_my              | OFF (base)|  <-- THIS
# | E-16       | ON        | ON        | qwen3_vl_mope_crossattn  | ON        |
#
# Usage:
#   bash eval_e15_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — model size to evaluate: "4b" or "8b" (default: 4b)
#   SPACE_ROOT           — project root on the execution server
#   VSIBENCH_VIDEO_ROOT  — directory containing VSIBench video files
#   VSIBENCH_JSONL       — path to vsibench test.jsonl annotation file
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for eval log files
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
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e15_lfp_predonly_4b"
        DEFAULT_EXP_NAME="e15_lfp_predonly_4b"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e15_lfp_predonly_8b"
        DEFAULT_EXP_NAME="e15_lfp_predonly_8b"
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
LOG_FILE="${LOG_DIR}/e15_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PORT=${MAIN_PORT:-$(shuf -i 20001-29999 -n 1)}

# ---------------------------------------------------------------------------
# PYTHONPATH: lmms-eval needs GUIDE's qwen-vl-finetune for model loading
# ---------------------------------------------------------------------------
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${GUIDE_TRAIN_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# Disable NCCL NVLS (known to cause hangs with multi-GPU lmms-eval)
# ---------------------------------------------------------------------------
export NCCL_NVLS_ENABLE=0

# ---------------------------------------------------------------------------
# Model args — standard GUIDE model (qwen3_vl_my), no MoPE at inference.
# E-15 was trained feed=False, so the cross-attn projector is untrained and
# MUST NOT be invoked at eval (it would inject garbage). strict=False on
# from_pretrained drops the MoPE/projector/LFP keys; only LLM+VGGT load.
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation (E-15 LFP prediction-only, MoPE-OFF inference) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "JSONL      : ${VSIBENCH_JSONL}"
echo "Processes  : ${NUM_PROCESSES}  Port: ${MAIN_PORT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
echo "Starting accelerate launch..."

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
echo "=== Log saved to: ${LOG_FILE} ==="
