#!/usr/bin/env bash
set -euo pipefail

MOPE_NEW_EXPERIMENT=e04a-new
MOPE_NEW_EVAL_NAME=e04a_mope_new_e01_projector_only_lr1e4_constant_bs2_diag_4b
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3,5,6}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PORT="${MAIN_PORT:-29527}"

export MOPE_NEW_EXPERIMENT MOPE_NEW_EVAL_NAME CUDA_VISIBLE_DEVICES NUM_PROCESSES MAIN_PORT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_eval_common.sh"
