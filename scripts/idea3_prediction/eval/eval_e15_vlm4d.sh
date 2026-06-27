#!/bin/bash
# =============================================================================
# Evaluate E-15 (LFP prediction-only) on VLM4D.
# Model type: qwen3_vl_my (BASE GUIDE inference — VGGT ON, MoPE OFF).
# (forked from eval_e15_vsibench.sh — only the task + data paths differ.)
#
# VLM4D: 4-way multiple-choice motion benchmark, pure letter matching, no judge.
#
# ATTRIBUTION RATIONALE (paper-2 Stage-1 2x2 eval design):
# See eval_e15_vsibench.sh header — E-15 is trained with feed=False, so the
# cross-attn projector is untrained and MUST be evaluated MoPE-OFF
# (--model qwen3_vl_my). strict=False drops MoPE/projector/LFP keys; only the
# LLM+VGGT weights (reshaped by the LFP auxiliary objective) are loaded.
#
# Paper-2 Stage-1 2x2 attribution table:
# | experiment | train feed | train LFP | eval model               | eval MoPE |
# |------------|:---------:|:---------:|--------------------------|:---------:|
# | E-01       | n/a       | n/a       | qwen3_vl_my              | OFF (base)|
# | E-03a      | ON        | OFF       | qwen3_vl_mope_crossattn  | ON        |
# | E-15       | OFF       | ON        | qwen3_vl_my              | OFF (base)|  <-- THIS
# | E-16       | ON        | ON        | qwen3_vl_mope_crossattn  | ON        |
#
# PENDING[D-11]: 论文2 评测范围基本定（VLM4D 动态主场），D-11 形式上仍 OPEN。
#
# Usage:
#   bash eval_e15_vlm4d.sh [CKPT_PATH] [EXP_NAME]
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — "4b" or "8b" (default: 4b)
#   SPACE_ROOT           — project root on the execution server
#   VLM4D_VIDEO_ROOT     — directory containing VLM4D video files (read by utils.py)
#   VLM4D_JSONL          — path to VLM4D annotation jsonl (injected into task yaml)
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
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

case "${MODEL_SIZE}" in
    4b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e15_lfp_predonly_4b"
        DEFAULT_EXP_NAME="e15_lfp_predonly_4b_vlm4d"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_OUTPUT_ROOT}/train/e15_lfp_predonly_8b"
        DEFAULT_EXP_NAME="e15_lfp_predonly_8b_vlm4d"
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

# Log file
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e15_vlm4d_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

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
echo "=== VLM4D Evaluation (E-15 LFP prediction-only, MoPE-OFF inference) ==="
echo "Model size : ${MODEL_SIZE}"
echo "Checkpoint : ${CKPT_PATH}"
echo "Experiment : ${EXP_NAME}"
echo "Output     : ${OUTPUT_PATH}"
echo "Log        : ${LOG_FILE}"
echo "Video root : ${VLM4D_VIDEO_ROOT}"
echo "JSONL      : ${VLM4D_JSONL}"
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
