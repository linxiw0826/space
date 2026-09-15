#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e05a-full
MOPE_NEW_EVAL_NAME=e05a_mope73_full_llm_4b
# Override this single path to evaluate any complete intermediate checkpoint.
# Example: .../checkpoint-1000 (or another HF checkpoint directory).
CKPT_PATH="${CKPT_PATH:-/data2/wlx/output/train/e05a_mope73_full_llm_4b}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_eval_common.sh"
