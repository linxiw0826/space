#!/bin/bash
# Source this file to set up PYTHONPATH for Space Sensing project.
# Convenience wrapper: train/eval scripts source env/activate.sh directly
# (which handles PYTHONPATH inside the per-server profile), so this file is
# intended for manual / interactive use (e.g. `source scripts/_common/setup_pythonpath.sh`).
#
# Usage: source scripts/_common/setup_pythonpath.sh
#
# SPACE_ROOT and the VSI-590K dataset paths are populated by the per-server
# profile sourced via env/activate.sh (profile_old.sh or profile_new.sh).

source "$(dirname "${BASH_SOURCE[0]}")/env/activate.sh"

export GUIDE_ROOT="${SPACE_ROOT}/src"
export MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

# PYTHONPATH segments (left-to-right):
#   1. ${SPACE_ROOT}/src/train_framework  — historically intended to shadow
#      GUIDE's `data` package via a project data/__init__.py, but that shadow
#      never existed (src/train_framework/data/ has no __init__.py; data_dict
#      is registered in src/qwenvl/data/__init__.py). Kept for backward
#      compatibility; effectively redundant since code imports
#      `src.train_framework.*` (resolved by segment 2). See audit
#      state/analyses/20260622_src_structure_audit.md §B.3 / §0.1.
#   2. ${SPACE_ROOT}            — exposes `src.*` as importable (CRITICAL for
#                                  train_space.py:98 `from src.train_framework.argument import ...`
#                                  and eval LMMS_EVAL_PLUGINS="src.eval").
#   3. ${GUIDE_ROOT}=${SPACE_ROOT}/src — exposes top-level packages
#                                  qwenvl.*, model.*, eval.*, preprocess.*,
#                                  analysis.* (CRITICAL for `from qwenvl.model...`
#                                  and `from model.mope_patch` imports).
#   4. ${MOPE_ROOT}=${SPACE_ROOT}/src/vendor/mope — exposes `models.*` (MoPE
#                                  timm registration). Redundant in practice:
#                                  mope_encoder.py:16-21 inserts this path via
#                                  sys.path at import time.
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

echo "[setup_pythonpath] PYTHONPATH set."
echo "  SPACE_ROOT       = ${SPACE_ROOT}"
echo "  GUIDE_ROOT       = ${GUIDE_ROOT}"
echo "  MOPE_ROOT        = ${MOPE_ROOT}"
echo "  VSI590K_DATA_ROOT= ${VSI590K_DATA_ROOT}"
