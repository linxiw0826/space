#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e04a-new
SMOKE_MODE=1
export MOPE_NEW_EXPERIMENT SMOKE_MODE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_vlm4d_eval_common.sh"
