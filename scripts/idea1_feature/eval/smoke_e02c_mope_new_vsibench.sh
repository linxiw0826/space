#!/usr/bin/env bash
set -euo pipefail

# VSI-Bench aggregation requires all ten question types, so smoke mode builds
# a one-example-per-type subset instead of using the unsafe generic --limit 1.
MOPE_NEW_EXPERIMENT=e02c-new
SMOKE_MODE=1
export MOPE_NEW_EXPERIMENT SMOKE_MODE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_mope_new_eval_common.sh"
