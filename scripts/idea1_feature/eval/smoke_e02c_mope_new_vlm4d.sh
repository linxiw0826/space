#!/usr/bin/env bash
set -euo pipefail

# Uses four real samples covering all three VLM4D sources on four GPUs.
# NUM_PROCESSES, CUDA_VISIBLE_DEVICES, and MAIN_PORT remain configurable.
MOPE_NEW_EXPERIMENT=e02c-new
SMOKE_MODE=1
export MOPE_NEW_EXPERIMENT SMOKE_MODE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_vlm4d_eval_common.sh"
