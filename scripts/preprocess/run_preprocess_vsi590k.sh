#!/usr/bin/env bash
# =============================================================================
# run_preprocess_vsi590k.sh
# SLURM batch script — wraps preprocess_vsi590k.py
#
# Usage:
#   sbatch run_preprocess_vsi590k.sh
#
# Variables to edit before submitting:
#   VIDEO_DIR   — local root of VSI video files
#   OUTPUT_DIR  — where JSON + extracted frames are written
#   NUM_SAMPLES — 0 = full dataset; set >0 to subsample
# =============================================================================

#SBATCH --account=bgkq-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=space_preprocess
#SBATCH --output=/u/lwu9/Space_sensing/projects/space/logs/preprocess_%j.out
#SBATCH --error=/u/lwu9/Space_sensing/projects/space/logs/preprocess_%j.err

# =============================================================================
# User-configurable variables — edit before submitting
# =============================================================================

# TODO: set VIDEO_DIR to the local root directory that contains
#       {dataset}/{scene_name}.mp4 files for VSI-590K
VIDEO_DIR=""                      # e.g. /work/hdd/bgkq/lwu9/data/vsi_videos

# TODO: set OUTPUT_DIR to where you want the JSON files and extracted frames
OUTPUT_DIR=""                     # e.g. /work/hdd/bgkq/lwu9/data/vsi590k_processed

# Number of samples to use; 0 = full dataset
NUM_SAMPLES=0

# Number of frames to extract per video for SPAR-style format
NUM_FRAMES=8

# HuggingFace dataset identifier
HF_DATASET="nyu-visionx/VSI-590K"

# HuggingFace cache directory (large files must live under /work/ per cluster rules)
HF_CACHE_DIR="/work/hdd/bgkq/lwu9/hf_cache"

# Absolute path to the Python preprocessing script
PYTHON_SCRIPT="/u/lwu9/Space_sensing/projects/space/src/preprocess/preprocess_vsi590k.py"

# =============================================================================
# Validate required variables
# =============================================================================

if [[ -z "${VIDEO_DIR}" ]]; then
    echo "ERROR: VIDEO_DIR is not set. Edit this script and set VIDEO_DIR before submitting." >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: OUTPUT_DIR is not set. Edit this script and set OUTPUT_DIR before submitting." >&2
    exit 1
fi

# =============================================================================
# Environment setup
# =============================================================================

# TODO: activate your conda environment — replace YOUR_ENV_NAME with the actual name
# conda activate YOUR_ENV_NAME

# =============================================================================
# Main
# =============================================================================

echo "=========================================="
echo "Job start:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
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
