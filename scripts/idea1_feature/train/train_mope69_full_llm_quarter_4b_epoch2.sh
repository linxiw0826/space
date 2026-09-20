#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,5,6}"; NPROC_PER_NODE="${NPROC_PER_NODE:-4}"; PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"; GRAD_ACCUM="${GRAD_ACCUM:-6}"; LEARNING_RATE="${LEARNING_RATE:-1e-6}"; LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MOPE_NEW_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json}"

# Warm-start from the completed epoch-1 full-LLM run (mope69_full_llm_quarter_4b),
# not the projector-only run — this makes epoch2 a continuation on the same data
# with a fresh optimizer/LR schedule, equivalent in effect to a 2nd epoch.
EPOCH1_OUTPUT_DIR="${EPOCH1_OUTPUT_DIR:-/data2/wlx/output/train/mope69_full_llm_quarter_4b}"
if [[ -z "${GUIDE_CKPT_PATH:-}" ]]; then
  if [[ -f "${EPOCH1_OUTPUT_DIR}/preprocessor_config.json" && -f "${EPOCH1_OUTPUT_DIR}/config.json" ]]; then
    # Training finished; the final HF-format save (with processor/tokenizer files)
    # lives at the output_dir root, not in a checkpoint-N subdir.
    GUIDE_CKPT_PATH="${EPOCH1_OUTPUT_DIR}"
  else
    LATEST_EPOCH1_CKPT="$(ls -d "${EPOCH1_OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
    [[ -n "${LATEST_EPOCH1_CKPT}" ]] || { echo "No completed run or checkpoint-* found under ${EPOCH1_OUTPUT_DIR}; set GUIDE_CKPT_PATH explicitly" >&2; exit 2; }
    GUIDE_CKPT_PATH="${LATEST_EPOCH1_CKPT}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-/data2/wlx/output/train/mope69_full_llm_quarter_4b_epoch2}"
MOPE_NEW_EXPERIMENT=e05a-quarter
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM LEARNING_RATE LR_SCHEDULER_TYPE MOPE_NEW_CKPT MOPE_NEW_SPAR_ANN OUTPUT_DIR MOPE_NEW_EXPERIMENT GUIDE_CKPT_PATH
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
echo "GUIDE_CKPT_PATH(epoch1 checkpoint)=${GUIDE_CKPT_PATH}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
