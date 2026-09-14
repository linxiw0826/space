#!/usr/bin/env bash
set -euo pipefail
SPACE_ROOT="${SPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
QUARTER_MANIFEST="${VSI590K_SPAR_ANN:-/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k_quarter_stratified.json}"
[[ -f "${QUARTER_MANIFEST}" && "${QUARTER_MANIFEST}" == *quarter_stratified.json ]] || { echo "Baseline quarter requires ${QUARTER_MANIFEST}" >&2; exit 2; }
CKPT_PATH="${CKPT_PATH:-${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}/train/baseline_quarter_4b}"
export SPACE_ROOT VSI590K_SPAR_ANN
exec bash "${SPACE_ROOT}/scripts/idea1_feature/eval/eval_e01_vlm4d.sh" "${CKPT_PATH}" baseline_quarter_4b_vlm4d
