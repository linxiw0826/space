#!/bin/bash
# Source this file to set up PYTHONPATH for Space Sensing project
# Usage: source scripts/setup_pythonpath.sh

export SPACE_ROOT="/home/nvme03/wlx/Space_sensing/projects/space"
export GUIDE_ROOT="${SPACE_ROOT}/src"
export MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

# src/train_framework must come first so our data/__init__.py shadows GUIDE's
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# VSI-590K dataset paths (must be set before running training)
export VSI590K_SPAR_ANN="${SPACE_ROOT}/data/vsi590k_spar.json"
export VSI590K_VIDEO_ANN="${SPACE_ROOT}/data/vsi590k_video.json"
export VSI590K_DATA_ROOT="${SPACE_ROOT}/data/"

echo "[setup_pythonpath] PYTHONPATH set."
echo "  SPACE_ROOT  = ${SPACE_ROOT}"
echo "  GUIDE_ROOT  = ${GUIDE_ROOT}"
echo "  MOPE_ROOT   = ${MOPE_ROOT}"
