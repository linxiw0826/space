#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"; NPROC_PER_NODE="${NPROC_PER_NODE:-8}"; PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"; GRAD_ACCUM="${GRAD_ACCUM:-3}"; LEARNING_RATE="${LEARNING_RATE:-3e-6}"; LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MOPE_NEW_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json}"

# Warm-start from the completed epoch-4 full-LLM run (e05_quarter_epoch4),
# continuing the same warm-start chain with a fresh optimizer/LR schedule,
# equivalent in effect to an 5th epoch on the same quarter data.
EPOCH4_OUTPUT_DIR="${EPOCH4_OUTPUT_DIR:-/data2/wlx/output/train/e05_quarter_epoch4}"
if [[ -z "${GUIDE_CKPT_PATH:-}" ]]; then
  if [[ -f "${EPOCH4_OUTPUT_DIR}/preprocessor_config.json" && -f "${EPOCH4_OUTPUT_DIR}/config.json" ]]; then
    # Training finished; the final HF-format save (with processor/tokenizer files)
    # lives at the output_dir root, not in a checkpoint-N subdir.
    GUIDE_CKPT_PATH="${EPOCH4_OUTPUT_DIR}"
  else
    LATEST_EPOCH4_CKPT="$(ls -d "${EPOCH4_OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
    [[ -n "${LATEST_EPOCH4_CKPT}" ]] || { echo "No completed run or checkpoint-* found under ${EPOCH4_OUTPUT_DIR}; set GUIDE_CKPT_PATH explicitly" >&2; exit 2; }
    GUIDE_CKPT_PATH="${LATEST_EPOCH4_CKPT}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-/data2/wlx/output/train/e05_quarter_epoch5}"
MOPE_NEW_EXPERIMENT=e05a-quarter
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM LEARNING_RATE LR_SCHEDULER_TYPE MOPE_NEW_CKPT MOPE_NEW_SPAR_ANN OUTPUT_DIR MOPE_NEW_EXPERIMENT GUIDE_CKPT_PATH
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
echo "GUIDE_CKPT_PATH(epoch4 checkpoint)=${GUIDE_CKPT_PATH}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
