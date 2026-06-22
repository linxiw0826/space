#!/bin/bash
# =============================================================================
# Evaluate E-10 (Router v1: CrossAttn + learned content-driven gate) on VLM4D
# Model type: qwen3_vl_mope_router
# (forked from eval_e03a_vlm4d.sh — only the model type + ckpt/exp names differ).
#
# VLM4D: 4-way multiple-choice motion benchmark, pure letter matching, no judge.
#
# PENDING[D-11]: 论文2 评测范围基本定（VLM4D 动态主场），D-11 形式上仍 OPEN。
#
# Usage:
#   bash eval_e10_vlm4d.sh [CKPT_PATH] [EXP_NAME]
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — model size to evaluate: "4b" or "8b" (default: 4b)
#   SPACE_ROOT           — project root on the execution server
#   VLM4D_VIDEO_ROOT     — directory containing VLM4D video files (read by utils.py)
#   VLM4D_JSONL          — path to VLM4D annotation jsonl (injected into task yaml)
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for log files (default: SPACE_ROOT/logs/eval)
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3); single-card override OK
#   NUM_PROCESSES        — number of accelerate processes (default: 4); =1 for single-card
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
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

case "${MODEL_SIZE}" in
    4b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e10_router_v1_4b"
        DEFAULT_EXP_NAME="e10_router_v1_4b_vlm4d"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e10_router_v1_8b"
        DEFAULT_EXP_NAME="e10_router_v1_8b_vlm4d"
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
# E-10 gate diagnostics (pure inference, no retrain):
#   VLM4D does ① per-question gate-value logging ONLY — there is no clean
#   static/dynamic task axis on VLM4D (all questions are motion questions,
#   split by video content), so the ② oracle hard-gate is NOT applied here.
#   GATE_MODE is therefore pinned to "learned" (the learned content-driven gate
#   is exactly what we want to read out per source = ego4d/davis/youtube-vos).
#   E10_GATE_LOG : if set, ① per-question gate values + source are written to
#                  this path (one <base>.rank{r}.jsonl per rank; merge with cat).
# ---------------------------------------------------------------------------
export GATE_MODE="learned"
# E10_GATE_LOG default is set below, once OUTPUT_PATH is known.

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
GUIDE_LMMS_EVAL=${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}
GUIDE_TRAIN_ROOT=${SPACE_ROOT}/src

# VLM4D data paths (defaults point at new server /data2/wlx/data/VLM4D)
# VERIFIED 2026-06-06: 标注是 QA/real_mc.json（JSON 数组）。VLM4D_JSONL 默认指向它，
#   下面的 sed 注入到 yaml 的 4 空格缩进 "    test:" 行（.json 路径无 sed 特殊字符，安全）。
VLM4D_VIDEO_ROOT=${VLM4D_VIDEO_ROOT:-/data2/wlx/data/VLM4D}
VLM4D_JSONL=${VLM4D_JSONL:-/data2/wlx/data/VLM4D/QA/real_mc.json}
export VLM4D_VIDEO_ROOT  # utils.py 直接从 env 读 media_dir

# Inject the annotation path into the task yaml's data_files.test line so that
# VLM4D_JSONL takes effect (the yaml hardcodes a default like vsibench does).
VLM4D_TASK_YAML="${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vlm4d/vlm4d.yaml"
if [ -f "${VLM4D_TASK_YAML}" ]; then
    sed -i -E "s#^(    test: ).*#\1${VLM4D_JSONL}#" "${VLM4D_TASK_YAML}"
fi

# Results output
RESULTS_DIR=${RESULTS_DIR:-${SPACE_OUTPUT_ROOT}/eval/vlm4d}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# E-10 ① g-log default path (into the run's output dir). Override by exporting
# E10_GATE_LOG before calling; set E10_GATE_LOG="" to disable.
export E10_GATE_LOG="${E10_GATE_LOG-${OUTPUT_PATH}/gate_log_learned.jsonl}"

# Log file
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e10_vlm4d_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_PROCESSES=${NUM_PROCESSES:-4}
MAIN_PORT=${MAIN_PORT:-$(shuf -i 20001-29999 -n 1)}

# ---------------------------------------------------------------------------
# LMMS_EVAL_PLUGINS: register qwen3_vl_mope_router model type.
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
# Model args — MoPE router model (qwen3_vl_mope_router), mope_all_frames=8
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2,mope_all_frames=8"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VLM4D Evaluation (E-10 Router v1: CrossAttn + learned gate) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VLM4D_VIDEO_ROOT}"
echo "JSONL      : ${VLM4D_JSONL}"
echo "Plugin     : ${LMMS_EVAL_PLUGINS}"
echo "Gate mode  : ${GATE_MODE} (① g-log only; ② oracle N/A on VLM4D)"
echo "Gate log   : ${E10_GATE_LOG:-<disabled>}"
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
    --model qwen3_vl_mope_router \
    --model_args "${MODEL_ARGS}" \
    --tasks vlm4d \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${EXP_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --force_simple \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Evaluation complete. Results at: ${OUTPUT_PATH} ==="
echo "=== Log saved to: ${LOG_FILE} ==="
