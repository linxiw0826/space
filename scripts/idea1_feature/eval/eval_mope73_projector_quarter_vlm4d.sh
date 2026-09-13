#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e04a-quarter
MOPE_NEW_EVAL_NAME=mope73_projector_quarter_lr1e5_4b
export MOPE_NEW_EXPERIMENT MOPE_NEW_EVAL_NAME
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_vlm4d_eval_common.sh"
