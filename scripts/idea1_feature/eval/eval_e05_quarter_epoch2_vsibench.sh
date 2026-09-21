#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e05a-quarter
MOPE_NEW_EVAL_NAME=e05_quarter_epoch2
CKPT_PATH="${CKPT_PATH:-/data2/wlx/output/train/e05_quarter_epoch2}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_eval_common.sh"
