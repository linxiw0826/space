#!/bin/bash
# =============================================================================
# Evaluate E-10b (Router v1.1 gatefix: CrossAttn + learned content-driven gate +
# anti-collapse) on VSIBench. Model type: qwen3_vl_mope_router (UNCHANGED — the
# gate fix is training-only; the eval model class is the same as E-10).
# (forked from eval_e10_vsibench.sh — only the ckpt/exp/output names differ. The
#  E10_GATE_LOG g-log is default-ON and now ALSO records per-question
#  ‖mope_out‖ / ‖image_embeds‖ / residual ratio — see change C.)
#
# Usage:
#   bash eval_e10b_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Arguments (optional — derived from MODEL_SIZE when omitted):
#   CKPT_PATH  — path to the E-10b checkpoint directory to evaluate
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
# Sets default CKPT_PATH and EXP_NAME based on model size.
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

case "${MODEL_SIZE}" in
    4b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e10b_router_v1_4b"
        DEFAULT_EXP_NAME="e10b_router_v1_4b"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e10b_router_v1_8b"
        DEFAULT_EXP_NAME="e10b_router_v1_8b"
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
#   GATE_MODE = learned (default) | oracle
#     learned : gate g from the trained gate_mlp (byte-equivalent to before).
#     oracle  : VSI static/dynamic task-label hard gate (PENDING[D-15]) — dynamic
#               subtasks (rel_direction/rel_distance/route_planning/appearance
#               order) -> g=1 (MoPE full open); static subtasks (abs_distance/
#               counting/size/room_size) -> g=0 (MoPE off). Routing upper bound.
#   E10_GATE_LOG : if set, ① per-question gate values are written to this path
#               (one <base>.rank{r}.jsonl per accelerate rank; merge with cat).
# When GATE_MODE is unset/=learned AND E10_GATE_LOG is unset, behaviour is
# byte-for-byte identical to the original learned eval.
# ---------------------------------------------------------------------------
GATE_MODE=${GATE_MODE:-learned}
export GATE_MODE
# E10_GATE_LOG default is set below, once OUTPUT_PATH is known.

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
VSIBENCH_TASK_YAML="${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/vsibench.yaml"
if [ -f "${VSIBENCH_TASK_YAML}" ]; then
    sed -i -E "s#^(    test: ).*#\1${VSIBENCH_JSONL}#" "${VSIBENCH_TASK_YAML}"
fi

# Results output
RESULTS_DIR=${RESULTS_DIR:-${SPACE_OUTPUT_ROOT}/eval/vsibench}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# E-10 ① g-log default path (into the run's output dir, tagged by gate mode).
# Override by exporting E10_GATE_LOG before calling; set E10_GATE_LOG="" to disable.
export E10_GATE_LOG="${E10_GATE_LOG-${OUTPUT_PATH}/gate_log_${GATE_MODE}.jsonl}"

# Log file — independent log directory, tee'd to stdout
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e10b_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

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
# Disable NCCL NVLS (known to cause hangs with multi-GPU lmms-eval).
# Also disable NCCL cuMem API (new server huirui: driver 570 / CUDA 12.8 /
# NCCL 2.28.9 falsely reports "CUDA driver version is insufficient" during
# multi-GPU NCCL init — see train_e10_router_v1.sh for full root-cause note).
# eval currently dodges NCCL via NUM_PROCESSES=1 single-card; these are kept
# as no-side-effect defaults for any future multi-card eval. ${VAR:-0} so an
# explicit env override still wins.
# ---------------------------------------------------------------------------
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}

# ---------------------------------------------------------------------------
# Force the pip-installed NCCL (nvidia-nccl-cu12 == 2.21.5) — prevents a rogue
# system / other-conda libnccl.so.2 (2.28.9) from shadowing pip's 2.21.5 and
# causing `NCCL version 2.28.9` + `Cuda failure 'driver insufficient'` in
# multi-GPU NCCL init. Same class of "library shadowing" as the cuDNN/videorepa
# issue above; the fix mirrors the cuDNN LD_PRELOAD approach. eval currently
# dodges NCCL via NUM_PROCESSES=1 single-card (harmless there); kept for any
# future multi-card eval. Path is resolved dynamically via `python -c` — no
# hardcoding, machine/env portable; only the nccl so is preloaded.
# See train_e10_router_v1.sh for the full root-cause note.
# ---------------------------------------------------------------------------
NCCL_LIB_DIR=$(python -c "import os,nvidia.nccl;print(os.path.join(os.path.dirname(nvidia.nccl.__file__),'lib'))" 2>/dev/null)
if [ -n "${NCCL_LIB_DIR}" ] && [ -f "${NCCL_LIB_DIR}/libnccl.so.2" ]; then
    export LD_PRELOAD="${NCCL_LIB_DIR}/libnccl.so.2:${LD_PRELOAD}"
    export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "[nccl] forcing pip NCCL via LD_PRELOAD: ${NCCL_LIB_DIR}/libnccl.so.2"
else
    echo "[nccl] WARNING: pip nvidia-nccl-cu12 lib not found; cannot force-preload NCCL (got NCCL_LIB_DIR='${NCCL_LIB_DIR}')." >&2
fi

# ---------------------------------------------------------------------------
# Model args — MoPE router model (qwen3_vl_mope_router), mope_all_frames=8
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2,mope_all_frames=8"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation (E-10b Router v1.1 gatefix: CrossAttn + learned gate + anti-collapse) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "JSONL      : ${VSIBENCH_JSONL}"
echo "Plugin     : ${LMMS_EVAL_PLUGINS}"
echo "Gate mode  : ${GATE_MODE}"
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
