#!/usr/bin/env bash
set -euo pipefail

# E-04a scheduler-only control: restart from the complete E-01 checkpoint and
# change only cosine -> constant_with_warmup. Never resume the cosine run.
# 2 samples/GPU x 4 GPUs x 6 accumulation steps = 48 samples/update.
CUDA_VISIBLE_DEVICES="1,3,5,6"
NPROC_PER_NODE="4"
PER_DEVICE_TRAIN_BATCH_SIZE="2"
GRAD_ACCUM="6"
LEARNING_RATE="1e-4"
LR_SCHEDULER_TYPE="constant_with_warmup"
MOPE_PROJECTOR_DIAG="1"
MOPE_PROJECTOR_DIAG_EVERY_FORWARDS="1920"
CANONICAL_OUTPUT_DIR="${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/e04a_mope_new_e01_projector_only_lr1e4_constant_bs2_diag_4b"
if [[ -n "${OUTPUT_DIR:-}" ]] && \
   [[ "$(realpath -m "${OUTPUT_DIR}")" != "$(realpath -m "${CANONICAL_OUTPUT_DIR}")" ]]; then
  echo "E-04a constant control OUTPUT_DIR is locked to ${CANONICAL_OUTPUT_DIR}; refusing override: ${OUTPUT_DIR}" >&2
  exit 2
fi
OUTPUT_DIR="${CANONICAL_OUTPUT_DIR}"
MOPE_NEW_EXPERIMENT=e04a-new
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM
export LEARNING_RATE LR_SCHEDULER_TYPE
export MOPE_PROJECTOR_DIAG MOPE_PROJECTOR_DIAG_EVERY_FORWARDS
export OUTPUT_DIR MOPE_NEW_EXPERIMENT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
