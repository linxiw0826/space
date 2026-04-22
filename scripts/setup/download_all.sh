#!/usr/bin/env bash
# download_all.sh — one-shot download for all Space project models and datasets.
# Safe to re-run: each section writes a .done marker and is skipped on subsequent runs.
# Usage: bash download_all.sh
# No set -e: errors are printed as WARNINGs and execution continues.

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT=/home/nvme01/wlx/Space_sensing/data
MODELS_ROOT=/home/nvme01/wlx/Space_sensing/models
NVME03_DATA=/home/nvme03/wlx/Space_sensing/data
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DONE_DIR="${DATA_ROOT}/.done"

export HF_ENDPOINT=https://hf-mirror.com

# ============================================================
# SECTION 0: Create directories
# ============================================================
echo ""
echo "=== SECTION 0: Create directories ==="

mkdir -p "${MODELS_ROOT}/Qwen3-VL-4B-Instruct"
mkdir -p "${MODELS_ROOT}/VGGT-1B"
mkdir -p "${DONE_DIR}"
mkdir -p "${DATA_ROOT}/guide_repro/train"
mkdir -p "${DATA_ROOT}/guide_repro/media/llava_hound/frames"
mkdir -p "${DATA_ROOT}/guide_repro/media/spar/scannet/images"
mkdir -p "${DATA_ROOT}/VSI-590K"
mkdir -p "${DATA_ROOT}/vsi590k_processed"
mkdir -p "${DATA_ROOT}/VSIBench"

echo "✓ done"

# ============================================================
# SECTION 1: Model downloads
# ============================================================
echo ""
echo "=== SECTION 1: Model downloads ==="

if [[ -f "${DONE_DIR}/models" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/models to re-run)"
else
    echo "  Downloading Qwen/Qwen3-VL-4B-Instruct ..."
    huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
        --local-dir "${MODELS_ROOT}/Qwen3-VL-4B-Instruct" \
        --resume-download \
        || echo "WARNING: Qwen3-VL-4B-Instruct download failed or incomplete"

    echo "  Downloading facebook/vggt-1b ..."
    # VGGT uses PyTorchModelHubMixin — downloaded directly from HF hub
    huggingface-cli download facebook/vggt-1b \
        --local-dir "${MODELS_ROOT}/VGGT-1B" \
        --resume-download \
        || echo "WARNING: vggt-1b download failed or incomplete"

    touch "${DONE_DIR}/models"
    echo "✓ done"
fi

# ============================================================
# SECTION 2: GUIDE training annotation JSONs
# ============================================================
echo ""
echo "=== SECTION 2: GUIDE training annotation JSONs ==="

if [[ -f "${DONE_DIR}/guide_annotations" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/guide_annotations to re-run)"
else
    echo "  Downloading llava_hound_64k.json and spar_234k.json from zd11024/VG-LLM-Data ..."
    huggingface-cli download zd11024/VG-LLM-Data \
        --repo-type dataset \
        --include "train/llava_hound_64k.json" "train/spar_234k.json" \
        --local-dir "${DATA_ROOT}/guide_repro" \
        --resume-download \
        || echo "WARNING: VG-LLM-Data annotation download failed or incomplete"

    # Verify
    if [[ ! -f "${DATA_ROOT}/guide_repro/train/llava_hound_64k.json" ]]; then
        echo "WARNING: ${DATA_ROOT}/guide_repro/train/llava_hound_64k.json not found after download"
    fi
    if [[ ! -f "${DATA_ROOT}/guide_repro/train/spar_234k.json" ]]; then
        echo "WARNING: ${DATA_ROOT}/guide_repro/train/spar_234k.json not found after download"
    fi

    touch "${DONE_DIR}/guide_annotations"
    echo "✓ done"
fi

# ============================================================
# SECTION 3: LLaVA-Hound video frames (selective download)
# ============================================================
echo ""
echo "=== SECTION 3: LLaVA-Hound video frames ==="

if [[ -f "${DONE_DIR}/llava_hound_frames" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/llava_hound_frames to re-run)"
else
    ANN_JSON="${DATA_ROOT}/guide_repro/train/llava_hound_64k.json"
    if [[ ! -f "${ANN_JSON}" ]]; then
        echo "WARNING: ${ANN_JSON} not found (Section 2 may have failed). Skipping LLaVA-Hound frame download."
    else
        python3 "${SCRIPT_DIR}/_fetch_llava_hound_frames.py" \
            --ann_json "${ANN_JSON}" \
            --output_dir "${DATA_ROOT}/guide_repro/media/llava_hound/frames" \
            --progress_file "${DONE_DIR}/llava_hound_frames_progress.json" \
            || echo "WARNING: _fetch_llava_hound_frames.py exited with errors — partial progress saved to progress_file"

        touch "${DONE_DIR}/llava_hound_frames"
        echo "✓ done"
    fi
fi

# ============================================================
# SECTION 4: ScanNet frames (MANUAL — requires license)
# ============================================================
echo ""
echo "=== SECTION 4: ScanNet (MANUAL - requires license) ==="
echo "  ScanNet frames for spar_234k require a non-commercial license."
echo "  Steps:"
echo "    1. Apply at https://kaldir.vc.in.tum.de/scannet/registration.php"
echo "    2. After approval (~1-3 days), download ScanNet v2 RGB frames"
echo "    3. Run: python prepare_scannet_layout.py"
echo "    4. Target: ${DATA_ROOT}/guide_repro/media/spar/scannet/images/"
echo "  Skipping automatic download."
# NOTE: No done-marker written here — reminder appears on every run intentionally.

# ============================================================
# SECTION 5: VSI-590K (rsync from nvme03, fallback to HF)
# ============================================================
echo ""
echo "=== SECTION 5: VSI-590K ==="

if [[ -f "${DONE_DIR}/vsi590k" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/vsi590k to re-run)"
else
    if [[ -d "${NVME03_DATA}/VSI-590K" ]]; then
        echo "  Syncing from nvme03 ..."
        rsync -av --progress --append-verify \
            "${NVME03_DATA}/VSI-590K/" \
            "${DATA_ROOT}/VSI-590K/" \
            || echo "WARNING: rsync of VSI-590K from nvme03 failed or incomplete"
    else
        echo "  nvme03 source not found (${NVME03_DATA}/VSI-590K). Falling back to huggingface-cli ..."
        huggingface-cli download nyu-visionx/VSI-590K \
            --repo-type dataset \
            --local-dir "${DATA_ROOT}/VSI-590K" \
            --resume-download \
            || echo "WARNING: VSI-590K HuggingFace download failed or incomplete"
    fi

    touch "${DONE_DIR}/vsi590k"
    echo "✓ done"
fi

# ============================================================
# SECTION 6: vsi590k_processed (rsync from nvme03)
# ============================================================
echo ""
echo "=== SECTION 6: vsi590k_processed ==="

if [[ -f "${DONE_DIR}/vsi590k_processed" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/vsi590k_processed to re-run)"
else
    if [[ -d "${NVME03_DATA}/vsi590k_processed" ]]; then
        echo "  Syncing from nvme03 ..."
        rsync -av --progress --append-verify \
            "${NVME03_DATA}/vsi590k_processed/" \
            "${DATA_ROOT}/vsi590k_processed/" \
            || echo "WARNING: rsync of vsi590k_processed from nvme03 failed or incomplete"

        touch "${DONE_DIR}/vsi590k_processed"
        echo "✓ done"
    else
        echo "WARNING: nvme03 source not found (${NVME03_DATA}/vsi590k_processed)."
        echo "  vsi590k_processed is a preprocessed artifact — generate it locally with the preprocess pipeline."
    fi
fi

# ============================================================
# SECTION 7: VSIBench
# ============================================================
echo ""
echo "=== SECTION 7: VSIBench ==="

if [[ -f "${DONE_DIR}/vsibench" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/vsibench to re-run)"
else
    echo "  Downloading nyu-visionx/VSIBench ..."
    huggingface-cli download nyu-visionx/VSIBench \
        --repo-type dataset \
        --local-dir "${DATA_ROOT}/VSIBench" \
        --resume-download \
        || echo "WARNING: VSIBench download failed or incomplete"

    touch "${DONE_DIR}/vsibench"
    echo "✓ done"
fi

# ============================================================
# SECTION 8: Validation summary
# ============================================================
echo ""
echo "=== SECTION 8: Validation summary ==="

_check_file() {
    local label="$1"
    local path="$2"
    if [[ -f "${path}" ]]; then
        echo "  ✓ ${label}"
    else
        echo "  ✗ ${label}  (missing: ${path})"
    fi
}

_check_dir_nonempty() {
    local label="$1"
    local dir="$2"
    if [[ -d "${dir}" ]] && [[ -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
        echo "  ✓ ${label}"
    else
        echo "  ✗ ${label}  (missing or empty: ${dir})"
    fi
}

_check_file  "Qwen3-VL-4B-Instruct config.json" \
    "${MODELS_ROOT}/Qwen3-VL-4B-Instruct/config.json"

_check_dir_nonempty "VGGT-1B/ (non-empty)" \
    "${MODELS_ROOT}/VGGT-1B"

_check_file  "guide_repro/train/llava_hound_64k.json" \
    "${DATA_ROOT}/guide_repro/train/llava_hound_64k.json"

_check_file  "guide_repro/train/spar_234k.json" \
    "${DATA_ROOT}/guide_repro/train/spar_234k.json"

_check_dir_nonempty "VSI-590K/ (non-empty)" \
    "${DATA_ROOT}/VSI-590K"

_check_dir_nonempty "VSIBench/ (non-empty)" \
    "${DATA_ROOT}/VSIBench"

echo "  ⚠️  ScanNet: manual download required (see Section 4 above)"

echo ""
echo "=== DOWNLOAD SUMMARY ==="
echo "MoPE checkpoints: already at /home/nvme04/mope-jepa/ (no download needed)"
echo "ScanNet: manual download required - see Section 4 output above"
echo "Re-run this script to resume any incomplete downloads."
