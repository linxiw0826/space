#!/usr/bin/env bash
set -euo pipefail

# E-04a LR ablation: preserve the original effective global batch of 48 while
# increasing the fresh projector peak LR from 1e-5 to 1e-3.
# 2 samples/GPU x 4 GPUs x 6 accumulation steps = 48 samples/update.
# Use a separate output directory so the completed E-04a baseline is immutable.
CUDA_VISIBLE_DEVICES="1,3,5,6"
NPROC_PER_NODE="4"
PER_DEVICE_TRAIN_BATCH_SIZE="2"
GRAD_ACCUM="6"
LEARNING_RATE="1e-3"
OUTPUT_DIR="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/e04a_mope_new_e01_projector_only_lr1e3_bs2_4b}"
MOPE_NEW_EXPERIMENT=e04a-new
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM
export LEARNING_RATE OUTPUT_DIR MOPE_NEW_EXPERIMENT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
