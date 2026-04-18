#!/usr/bin/env bash
# =============================================================================
# run_preprocess_vsi590k.sh
# Plain bash script — wraps preprocess_vsi590k.py
# Runs on ubuntu@27.190.15.135 (no SLURM). (D-06)
#
# Usage:
#   bash run_preprocess_vsi590k.sh
#
# Variables to edit before running:
#   VIDEO_DIR   — local root of VSI video files
#   OUTPUT_DIR  — where JSON + extracted frames are written
#   NUM_SAMPLES — 0 = full dataset; set >0 to subsample
# =============================================================================

# =============================================================================
# Log setup: stdout + stderr → terminal AND log file simultaneously
# =============================================================================

LOG_DIR="/home/nvme03/wlx/Space_sensing/projects/space/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"

# Redirect all subsequent stdout and stderr through tee into the log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# =============================================================================
# User-configurable variables — edit before running
# =============================================================================

VIDEO_DIR="/home/nvme03/wlx/Space_sensing/data/VSI-590K"

OUTPUT_DIR="/home/nvme03/wlx/Space_sensing/data/vsi590k_processed"

# Number of samples to use; 0 = full dataset
NUM_SAMPLES=0

# Number of frames to extract per video for SPAR-style format
NUM_FRAMES=8

# HuggingFace dataset identifier
HF_DATASET="nyu-visionx/VSI-590K"

# HuggingFace cache directory
HF_CACHE_DIR="/home/nvme03/wlx/Space_sensing/data/hf_cache"

# Absolute path to the Python preprocessing script
PYTHON_SCRIPT="/home/nvme03/wlx/Space_sensing/projects/space/src/preprocess/preprocess_vsi590k.py"

# =============================================================================
# Validate required variables
# =============================================================================

if [[ -z "${VIDEO_DIR}" ]]; then
    echo "ERROR: VIDEO_DIR is not set. Edit this script and set VIDEO_DIR before running." >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: OUTPUT_DIR is not set. Edit this script and set OUTPUT_DIR before running." >&2
    exit 1
fi

# =============================================================================
# Environment setup
# =============================================================================

# TODO: fill in conda env name (run `conda env list` to find it)
# conda activate YOUR_ENV_NAME

# Force HuggingFace to use the large-disk cache directory
export HF_HOME="${HF_CACHE_DIR}"

# =============================================================================
# Main
# =============================================================================

echo "=========================================="
echo "Job start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "CPUs:       $(nproc)"
echo "Log file:   ${LOG_FILE}"
echo "------------------------------------------"
echo "VIDEO_DIR:  ${VIDEO_DIR}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NUM_SAMPLES:${NUM_SAMPLES}"
echo "NUM_FRAMES: ${NUM_FRAMES}"
echo "HF_DATASET: ${HF_DATASET}"
echo "HF_CACHE:   ${HF_CACHE_DIR}"
echo "=========================================="

# Build optional --num_samples argument (omit when 0 to use full dataset)
SAMPLES_ARG=""
if [[ "${NUM_SAMPLES}" -gt 0 ]]; then
    SAMPLES_ARG="--num_samples ${NUM_SAMPLES}"
fi

python "${PYTHON_SCRIPT}" \
    --video_dir    "${VIDEO_DIR}" \
    --output_dir   "${OUTPUT_DIR}" \
    --num_frames   "${NUM_FRAMES}" \
    --hf_dataset   "${HF_DATASET}" \
    --hf_cache_dir "${HF_CACHE_DIR}" \
    ${SAMPLES_ARG}

EXIT_CODE=$?

echo "=========================================="
echo "Job end:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code:  ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
