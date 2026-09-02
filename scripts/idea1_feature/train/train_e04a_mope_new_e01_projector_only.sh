#!/usr/bin/env bash
set -euo pipefail

# E-04a: initialize from the completed E-01 SFT checkpoint, attach a fresh
# zero-init final515k CrossAttn branch, freeze the full E-01/MoPE backbones,
# and train only the CrossAttn projector. No gate and no projector warm-start.
# The verified default launch uses physical GPUs 1,2,4,6. Keep the effective
# global batch at 48 while reducing activation memory after batch=2 OOM:
# 1 sample/GPU x 4 GPUs x 12 accumulation steps.
# Pin the verified launch contract instead of inheriting stale values from a
# long-lived shell/tmux session.
CUDA_VISIBLE_DEVICES="1,2,4,6"
NPROC_PER_NODE="4"
PER_DEVICE_TRAIN_BATCH_SIZE="1"
GRAD_ACCUM="12"
MOPE_NEW_EXPERIMENT=e04a-new
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PER_DEVICE_TRAIN_BATCH_SIZE GRAD_ACCUM MOPE_NEW_EXPERIMENT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
