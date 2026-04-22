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
# LOGGING
# ============================================================
LOG_DIR="/home/nvme03/wlx/Space_sensing/logs/setup"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/download_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== Log: ${LOG_FILE} ==="

# ============================================================
# HELPERS
# ============================================================
_hf_download() {
    local attempt=1
    while [[ $attempt -le 5 ]]; do
        hf download "$@" && return 0
        echo "  [retry] Attempt ${attempt}/5 failed — retrying in 30s ..."
        sleep 30
        ((attempt++))
    done
    return 1
}

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
mkdir -p "${DATA_ROOT}/guide_repro/media/spar/structured3d/images"
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
    _ok=true
    echo "  Downloading Qwen/Qwen3-VL-4B-Instruct ..."
    _hf_download Qwen/Qwen3-VL-4B-Instruct \
        --local-dir "${MODELS_ROOT}/Qwen3-VL-4B-Instruct" \
        || { echo "WARNING: Qwen3-VL-4B-Instruct download failed or incomplete"; _ok=false; }

    echo "  Downloading facebook/vggt-1b ..."
    # VGGT uses PyTorchModelHubMixin — downloaded directly from HF hub
    _hf_download facebook/vggt-1b \
        --local-dir "${MODELS_ROOT}/VGGT-1B" \
        || { echo "WARNING: vggt-1b download failed or incomplete"; _ok=false; }

    if [[ "${_ok}" == "true" ]]; then
        touch "${DONE_DIR}/models"
        echo "✓ done"
    else
        echo "✗ models section incomplete — re-run to retry"
    fi
fi

# ============================================================
# SECTION 2: GUIDE training annotation JSONs
# ============================================================
echo ""
echo "=== SECTION 2: GUIDE training annotation JSONs ==="

if [[ -f "${DONE_DIR}/guide_annotations" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/guide_annotations to re-run)"
else
    _ok=true
    echo "  Downloading llava_hound_64k.json and spar_234k.json from zd11024/VG-LLM-Data ..."
    _hf_download zd11024/VG-LLM-Data \
        "train/llava_hound_64k.json" "train/spar_234k.json" \
        --repo-type dataset \
        --local-dir "${DATA_ROOT}/guide_repro" \
        || { echo "WARNING: VG-LLM-Data annotation download failed or incomplete"; _ok=false; }

    # Verify
    if [[ ! -f "${DATA_ROOT}/guide_repro/train/llava_hound_64k.json" ]]; then
        echo "WARNING: ${DATA_ROOT}/guide_repro/train/llava_hound_64k.json not found after download"
        _ok=false
    fi
    if [[ ! -f "${DATA_ROOT}/guide_repro/train/spar_234k.json" ]]; then
        echo "WARNING: ${DATA_ROOT}/guide_repro/train/spar_234k.json not found after download"
        _ok=false
    fi

    if [[ "${_ok}" == "true" ]]; then
        touch "${DONE_DIR}/guide_annotations"
        echo "✓ done"
    else
        echo "✗ guide_annotations section incomplete — re-run to retry"
    fi
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
        _ok=true
        python3 "${SCRIPT_DIR}/_fetch_llava_hound_frames.py" \
            --ann_json "${ANN_JSON}" \
            --output_dir "${DATA_ROOT}/guide_repro/media/llava_hound/frames" \
            --progress_file "${DONE_DIR}/llava_hound_frames_progress.json" \
            || { echo "WARNING: _fetch_llava_hound_frames.py exited with errors — partial progress saved to progress_file"; _ok=false; }

        if [[ "${_ok}" == "true" ]]; then
            touch "${DONE_DIR}/llava_hound_frames"
            echo "✓ done"
        else
            echo "✗ llava_hound_frames section incomplete — re-run to retry"
        fi
    fi
fi

# ============================================================
# SECTION 4: SPAR image data (jasonzhango/SPAR-7M-RGBD on HuggingFace)
# Contains RGB images + depth + pose for ScanNet, ScanNet++, Structured3D (~184 GB).
# Split into 18 shards (spar-rgbd-00..17.tar.gz); must be concatenated before extraction.
# Extracts to: ${DATA_ROOT}/guide_repro/media/spar/{scannet,scannetpp,structured3d}/
# ============================================================
echo ""
echo "=== SECTION 4: SPAR image data (jasonzhango/SPAR-7M-RGBD) ==="

if [[ -f "${DONE_DIR}/spar_images" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/spar_images to re-run)"
else
    SPAR_MEDIA="${DATA_ROOT}/guide_repro/media"
    SPAR_STAGE="${DATA_ROOT}/.stage_spar7m"
    mkdir -p "${SPAR_STAGE}" "${SPAR_MEDIA}"

    _ok=true
    echo "  Downloading 18 shards from jasonzhango/SPAR-7M-RGBD (~184 GB total) ..."
    _hf_download jasonzhango/SPAR-7M-RGBD \
        --repo-type dataset \
        --include "spar-rgbd-*.tar.gz" \
        --local-dir "${SPAR_STAGE}" \
        || { echo "WARNING: SPAR-7M-RGBD download failed or incomplete"; _ok=false; }

    shard_count=$(ls "${SPAR_STAGE}"/spar-rgbd-*.tar.gz 2>/dev/null | wc -l)
    echo "  Shards present: ${shard_count}/18"

    if [[ "${shard_count}" -eq 18 ]]; then
        echo "  Extracting all shards → ${SPAR_MEDIA}/ ..."
        # Split archives must be concatenated; extracts to spar/{scannet,scannetpp,structured3d}/
        cat "${SPAR_STAGE}"/spar-rgbd-*.tar.gz \
            | tar -xzf - -C "${SPAR_MEDIA}" \
            && rm -f "${SPAR_STAGE}"/spar-rgbd-*.tar.gz \
            || { echo "WARNING: SPAR-7M-RGBD extraction failed"; _ok=false; }
    else
        echo "WARNING: Only ${shard_count}/18 shards present — skipping extraction. Re-run to retry."
        _ok=false
    fi

    if [[ "${_ok}" == "true" ]]; then
        touch "${DONE_DIR}/spar_images"
        echo "✓ done"
    else
        echo "✗ spar_images section incomplete — re-run to retry"
    fi
fi

# ============================================================
# SECTION 5: VSI-590K (rsync from nvme03, fallback to HF)
# ============================================================
echo ""
echo "=== SECTION 5: VSI-590K ==="

if [[ -f "${DONE_DIR}/vsi590k" ]]; then
    echo "  Already done — skipping (delete ${DONE_DIR}/vsi590k to re-run)"
else
    _ok=true
    if [[ -d "${NVME03_DATA}/VSI-590K" ]]; then
        echo "  Syncing from nvme03 ..."
        rsync -av --progress --append-verify \
            "${NVME03_DATA}/VSI-590K/" \
            "${DATA_ROOT}/VSI-590K/" \
            || { echo "WARNING: rsync of VSI-590K from nvme03 failed or incomplete"; _ok=false; }
    else
        echo "  nvme03 source not found (${NVME03_DATA}/VSI-590K). Falling back to hf ..."
        _hf_download nyu-visionx/VSI-590K \
            --repo-type dataset \
            --local-dir "${DATA_ROOT}/VSI-590K" \
            || { echo "WARNING: VSI-590K HuggingFace download failed or incomplete"; _ok=false; }
    fi

    if [[ "${_ok}" == "true" ]]; then
        touch "${DONE_DIR}/vsi590k"
        echo "✓ done"
    else
        echo "✗ vsi590k section incomplete — re-run to retry"
    fi
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
        _ok=true
        echo "  Syncing from nvme03 ..."
        rsync -av --progress --append-verify \
            "${NVME03_DATA}/vsi590k_processed/" \
            "${DATA_ROOT}/vsi590k_processed/" \
            || { echo "WARNING: rsync of vsi590k_processed from nvme03 failed or incomplete"; _ok=false; }

        if [[ "${_ok}" == "true" ]]; then
            touch "${DONE_DIR}/vsi590k_processed"
            echo "✓ done"
        else
            echo "✗ vsi590k_processed section incomplete — re-run to retry"
        fi
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
    _ok=true
    echo "  Downloading nyu-visionx/VSIBench ..."
    _hf_download nyu-visionx/VSI-Bench \
        --repo-type dataset \
        --local-dir "${DATA_ROOT}/VSIBench" \
        || { echo "WARNING: VSIBench download failed or incomplete"; _ok=false; }

    if [[ "${_ok}" == "true" ]]; then
        touch "${DONE_DIR}/vsibench"
        echo "✓ done"
    else
        echo "✗ VSIBench section incomplete — re-run to retry"
    fi
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

_check_dir_nonempty "SPAR scannet images (non-empty)" \
    "${DATA_ROOT}/guide_repro/media/spar/scannet/images"

_check_dir_nonempty "SPAR scannetpp images (non-empty)" \
    "${DATA_ROOT}/guide_repro/media/spar/scannetpp/images"

_check_dir_nonempty "SPAR structured3d images (non-empty)" \
    "${DATA_ROOT}/guide_repro/media/spar/structured3d/images"

echo ""
echo "=== DOWNLOAD SUMMARY ==="
echo "MoPE checkpoints: already at /home/nvme04/mope-jepa/ (no download needed)"
echo "Re-run this script to resume any incomplete downloads."
