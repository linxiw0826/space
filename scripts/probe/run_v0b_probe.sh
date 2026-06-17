#!/bin/bash
# =============================================================================
# run_v0b_probe.sh  —  V0-B Frozen Probing: MoPE layer separability
#
# Runs v0b_layer_probe.py to extract frozen MoPE features at specified
# transformer block layers and train linear probes (dynamic vs. static).
#
# Usage:
#   bash run_v0b_probe.sh [MANIFEST] [MOPE_CKPT] [EXP_NAME]
#
# Positional args (all optional — override env vars below):
#   MANIFEST   — path to the manifest CSV  (video_path,label)
#   MOPE_CKPT  — path to the MoPE .pth checkpoint
#   EXP_NAME   — short tag for this run, used in output/log filenames
#
# Env vars (override defaults by setting before running):
#   SPACE_ROOT     — project root (default: auto-detected from script location)
#   MANIFEST       — manifest CSV path
#   MOPE_CKPT      — MoPE checkpoint path
#   EXP_NAME       — experiment name tag
#   LAYERS         — comma-separated block indices to probe (default: "7,11")
#   NUM_FRAMES     — frames to read per video (default: 16)
#   SAMPLING_RATE  — temporal sampling rate (default: 4)
#   ALL_FRAMES     — all_frames param for MoPEEncoder (default: 8)
#   N_FOLDS        — number of CV folds (default: 5)
#   DEVICE         — pytorch device string, e.g. "cuda:0" or "cpu"
#   OUTPUT_DIR     — directory for JSON reports (default: SPACE_ROOT/logs/probe)
#   LOG_DIR        — directory for log files   (default: SPACE_ROOT/logs/probe)
#
# Manifest CSV format:
#   video_path,label
#   /remote/server/videos/explosion_001.mp4,1
#   /remote/server/videos/still_room_042.mp4,0
#   ...
#   label: 0=static, 1=dynamic
#   Fill in paths that are accessible on the remote server.
#
# Example manifest placeholder (remote server — fill in actual paths):
#   MANIFEST=/data2/wlx/Space_sensing/data/probe_manifests/dynamic_vs_static.csv
#
# Quickstart on remote server:
#   export MANIFEST=/path/to/manifest.csv
#   export MOPE_CKPT=/path/to/checkpoint.pth
#   bash run_v0b_probe.sh
# =============================================================================
set -e

# ---------------------------------------------------------------------------
# Locate project root (two levels up from this script's directory)
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}"

# ---------------------------------------------------------------------------
# Source per-server environment profile (sets SPACE_PROFILE / conda env etc.)
# Non-fatal: the warning from activate.sh is acceptable on unknown hosts.
# ---------------------------------------------------------------------------
source "${_SCRIPT_DIR}/../env/activate.sh" || true

# ---------------------------------------------------------------------------
# Configuration — all overridable via env or positional args
# ---------------------------------------------------------------------------
MANIFEST="${1:-${MANIFEST:-/path/to/remote/manifest.csv}}"        # FILL IN on remote server
MOPE_CKPT="${2:-${MOPE_CKPT:-/path/to/remote/checkpoint.pth}}"   # FILL IN on remote server
EXP_NAME="${3:-${EXP_NAME:-v0b_probe}}"

LAYERS="${LAYERS:-7,11}"
NUM_FRAMES="${NUM_FRAMES:-16}"
SAMPLING_RATE="${SAMPLING_RATE:-4}"
ALL_FRAMES="${ALL_FRAMES:-8}"
N_FOLDS="${N_FOLDS:-5}"
DEVICE="${DEVICE:-cuda}"

OUTPUT_DIR="${OUTPUT_DIR:-${SPACE_ROOT}/logs/probe}"
LOG_DIR="${LOG_DIR:-${SPACE_ROOT}/logs/probe}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_JSON="${OUTPUT_DIR}/${EXP_NAME}_${TIMESTAMP}.json"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# PYTHONPATH: project root + src + vendor/mope  (mirrors setup_pythonpath.sh)
# Note: mope_encoder.py also adds vendor/mope to sys.path at import time,
# but setting it here ensures dataset.loader resolves correctly as well.
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}:${SPACE_ROOT}/src:${SPACE_ROOT}/src/vendor/mope:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Validate required paths before launching
# ---------------------------------------------------------------------------
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    echo "       Set MANIFEST= or pass it as the first positional argument." >&2
    exit 1
fi

if [ ! -f "${MOPE_CKPT}" ]; then
    echo "ERROR: MoPE checkpoint not found: ${MOPE_CKPT}" >&2
    echo "       Set MOPE_CKPT= or pass it as the second positional argument." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Status output
# ---------------------------------------------------------------------------
echo "=== V0-B Layer Probe (frozen MoPE encoder) ==="
echo "SPACE_ROOT    : ${SPACE_ROOT}"
echo "Manifest      : ${MANIFEST}"
echo "MoPE ckpt     : ${MOPE_CKPT}"
echo "Experiment    : ${EXP_NAME}"
echo "Layers        : ${LAYERS}  + postnorm"
echo "num_frames    : ${NUM_FRAMES}   sampling_rate: ${SAMPLING_RATE}"
echo "all_frames    : ${ALL_FRAMES}   n_folds: ${N_FOLDS}"
echo "Device        : ${DEVICE}"
echo "Output JSON   : ${OUTPUT_JSON}"
echo "Log file      : ${LOG_FILE}"
echo "PYTHONPATH    : ${PYTHONPATH}"
echo "================================================"

# ---------------------------------------------------------------------------
# Run probe script — tee stdout+stderr to log file
# ---------------------------------------------------------------------------
python "${_SCRIPT_DIR}/v0b_layer_probe.py" \
    --manifest       "${MANIFEST}"     \
    --mope_ckpt      "${MOPE_CKPT}"    \
    --layers         "${LAYERS}"       \
    --output         "${OUTPUT_JSON}"  \
    --num_frames     "${NUM_FRAMES}"   \
    --sampling_rate  "${SAMPLING_RATE}" \
    --all_frames     "${ALL_FRAMES}"   \
    --n_folds        "${N_FOLDS}"      \
    --device         "${DEVICE}"       \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=== Probe complete ==="
echo "    JSON report : ${OUTPUT_JSON}"
echo "    Log         : ${LOG_FILE}"
