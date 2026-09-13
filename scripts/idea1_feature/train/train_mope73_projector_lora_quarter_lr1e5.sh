#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="1,3,5,6"; NPROC_PER_NODE="4"; PER_DEVICE_TRAIN_BATCH_SIZE="2"; GRAD_ACCUM="6"; LEARNING_RATE="1e-5"; LR_SCHEDULER_TYPE="cosine"
MOPE_NEW_CKPT="${MOPE_NEW_CKPT:-/data2/mope-jepa-assets/jepa_checkpoints/native_mope_b_dense8_moe8_top1_shared1_anchor1_final515k_3dpos_ep100_warm3_cos_lr75e6_min25e6/checkpoint-73.pth}"
MOPE_NEW_SPAR_ANN="${MOPE_NEW_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT:-/data2/wlx/output}/train/mope73_projector_lora_quarter_lr1e5_4b}"
MOPE_NEW_EXPERIMENT=e04b-quarter; LORA_ENABLE=True; LORA_R=8; LORA_ALPHA=16; LORA_DROPOUT=0.05; MOPE_LORA_LR=5e-6; MOPE_LORA_LAST_N=8
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM LEARNING_RATE LR_SCHEDULER_TYPE MOPE_NEW_CKPT MOPE_NEW_SPAR_ANN OUTPUT_DIR MOPE_NEW_EXPERIMENT LORA_ENABLE LORA_R LORA_ALPHA LORA_DROPOUT MOPE_LORA_LR MOPE_LORA_LAST_N
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH:-}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
