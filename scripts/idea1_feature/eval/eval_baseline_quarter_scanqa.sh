#!/usr/bin/env bash
set -euo pipefail
SPACE_ROOT="${SPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"
CKPT_PATH="${CKPT_PATH:-/data2/wlx/output/train/baseline_quarter_4b}"
export SPACE_ROOT LOG_PREFIX=baseline_quarter_4b
exec bash "${SPACE_ROOT}/scripts/idea1_feature/eval/eval_e01_scanqa.sh" "${CKPT_PATH}" baseline_quarter_4b_scanqa
