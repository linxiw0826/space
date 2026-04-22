#!/bin/bash
# =============================================================================
# Evaluate E-01 (GUIDE fine-tuned on VSI-590K, no MoPE) checkpoint on VSIBench
#
# Usage:
#   bash eval_e01_vsibench.sh [CKPT_PATH] [EXP_NAME]
#
# Arguments (optional — derived from MODEL_SIZE when omitted):
#   CKPT_PATH  — path to the E-01 fine-tuned checkpoint directory
#   EXP_NAME   — short name for this evaluation run, used for output dir
#
# Env vars (all optional, have defaults):
#   MODEL_SIZE           — "4b" or "8b" (default: 4b)
#   SPACE_ROOT           — project root on the execution server
#   VSIBENCH_VIDEO_ROOT  — directory containing VSIBench video files
#   VSIBENCH_JSONL       — path to vsibench test.jsonl annotation file
#   GUIDE_LMMS_EVAL      — path to GUIDE's lmms-eval repo root
#   RESULTS_DIR          — base directory for evaluation results
#   LOG_DIR              — directory for eval log files
#   CUDA_VISIBLE_DEVICES — GPUs to use (default: 0,1,2,3,4,5,6,7)
#   NUM_PROCESSES        — number of accelerate processes (default: 8)
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
        DEFAULT_CKPT_PATH="${SPACE_ROOT}/outputs/e01_guide_4b"
        DEFAULT_EXP_NAME="e01_guide_4b"
        ;;
    8b)
        DEFAULT_CKPT_PATH="${SPACE_ROOT}/outputs/e01_guide_8b"
        DEFAULT_EXP_NAME="e01_guide_8b"
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
VSIBENCH_VIDEO_ROOT=${VSIBENCH_VIDEO_ROOT:-/home/nvme03/wlx/Space_sensing/data/vsibench/}
VSIBENCH_JSONL=${VSIBENCH_JSONL:-/home/nvme03/wlx/Space_sensing/data/vsibench/test.jsonl}

# Results output
RESULTS_DIR=${RESULTS_DIR:-${SPACE_ROOT}/results/vsibench}
OUTPUT_PATH="${RESULTS_DIR}/${EXP_NAME}"
mkdir -p "${OUTPUT_PATH}"

# Log file — independent log directory, tee'd to stdout
LOG_DIR=${LOG_DIR:-${SPACE_ROOT}/logs/eval}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/e01_vsibench_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-8}
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
# VSIBench task override
#
# The built-in vsibench.yaml hardcodes paths to /storage/chongyu/...
# We write a local override yaml that redirects to our data paths and
# pass it via --include_path so lmms-eval finds it first.
# ---------------------------------------------------------------------------
TASK_OVERRIDE_DIR="${OUTPUT_PATH}/task_override"
mkdir -p "${TASK_OVERRIDE_DIR}/vsibench"

cat > "${TASK_OVERRIDE_DIR}/vsibench/vsibench.yaml" << YAML_EOF
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${VSIBENCH_JSONL}
task: vsibench
test_split: test
output_type: generate_until
process_docs: !function utils.process_docs
doc_to_visual: !function utils.vsibench_doc_to_visual
doc_to_text: !function utils.vsibench_doc_to_text
doc_to_target: "ground_truth"
generation_kwargs:
  max_new_tokens: 16
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
process_results: !function utils.vsibench_process_results
metric_list:
  - metric: vsibench_score
    aggregation: !function utils.vsibench_aggregate_results
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    mca_post_prompt: "Answer with the option's letter from the given choices directly."
    na_post_prompt: "Please answer the question using a single word or phrase."
metadata:
  media_dir: ${VSIBENCH_VIDEO_ROOT}
  version: 0.0
YAML_EOF

# Copy utils.py from the GUIDE reference so the override task dir is complete
cp "${GUIDE_LMMS_EVAL}/lmms_eval/tasks/vsibench/utils.py" \
   "${TASK_OVERRIDE_DIR}/vsibench/utils.py"

# ---------------------------------------------------------------------------
# Model args — standard GUIDE model (qwen3_vl_my), no MoPE
# ---------------------------------------------------------------------------
MODEL_ARGS="pretrained=${CKPT_PATH},max_pixels=268324,min_pixels=8192,attn_implementation=flash_attention_2"

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== VSIBench Evaluation (E-01 GUIDE fine-tune, no MoPE) ==="
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
    --include_path "${TASK_OVERRIDE_DIR}" \
    --force_simple \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Evaluation complete. Results at: ${OUTPUT_PATH} ==="
echo "=== Log saved to: ${LOG_FILE} ==="
