#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e00b-new
export MOPE_NEW_EXPERIMENT
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_vlm4d_eval_common.sh"
