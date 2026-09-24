#!/usr/bin/env bash
set -euo pipefail
MOPE_NEW_EXPERIMENT=e05a-quarter
MOPE_NEW_EVAL_NAME=mope69_full_llm_quarter_4b
CKPT_PATH="${CKPT_PATH:-/data2/wlx/output/train/mope69_full_llm_quarter_4b}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_scanqa_eval_common.sh"
