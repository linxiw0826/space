#!/usr/bin/env bash
set -euo pipefail

# E-04a: initialize from the completed E-01 SFT checkpoint, attach a fresh
# zero-init final515k CrossAttn branch, freeze the full E-01/MoPE backbones,
# and train only the CrossAttn projector. No gate and no projector warm-start.
# The verified default launch uses physical GPUs 1,2,4,6. Keep the effective
# global batch at 48: 2 samples/GPU x 4 GPUs x 6 accumulation steps.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,4,6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
MOPE_NEW_EXPERIMENT=e04a-new
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE GRAD_ACCUM MOPE_NEW_EXPERIMENT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_train_common.sh"
