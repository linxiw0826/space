#!/bin/bash
# Source this file to set up PYTHONPATH for Space Sensing project
# Usage: source scripts/setup_pythonpath.sh
#
# SPACE_ROOT and the VSI-590K dataset paths are populated by the per-server
# profile sourced via env/activate.sh (profile_old.sh or profile_new.sh).

source "$(dirname "${BASH_SOURCE[0]}")/env/activate.sh"

export GUIDE_ROOT="${SPACE_ROOT}/src"
export MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

# src/train_framework must come first so our data/__init__.py shadows GUIDE's
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

echo "[setup_pythonpath] PYTHONPATH set."
echo "  SPACE_ROOT       = ${SPACE_ROOT}"
echo "  GUIDE_ROOT       = ${GUIDE_ROOT}"
echo "  MOPE_ROOT        = ${MOPE_ROOT}"
echo "  VSI590K_DATA_ROOT= ${VSI590K_DATA_ROOT}"
