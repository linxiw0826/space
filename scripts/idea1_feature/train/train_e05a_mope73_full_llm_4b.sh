#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="1,3,5,6"; NPROC_PER_NODE="4"; PER_DEVICE_TRAIN_BATCH_SIZE="2"; GRAD_ACCUM="6"; LEARNING_RATE="1e-6"; LR_SCHEDULER_TYPE="cosine"
MOPE_NEW_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_final515k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/wlx/output/train/e05a_mope73_full_llm_4b}"
MOPE_NEW_EXPERIMENT=e05a-full
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM LEARNING_RATE LR_SCHEDULER_TYPE MOPE_NEW_SPAR_ANN OUTPUT_DIR MOPE_NEW_EXPERIMENT
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
