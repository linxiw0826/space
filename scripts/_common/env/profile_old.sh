#!/bin/bash
# =============================================================================
# Old server profile (ubuntu@27.190.15.135, multi-NVMe layout).
#
# Sourced by activate.sh. Exports all paths used by train/eval/preprocess
# scripts. Layout:
#   nvme01  -> models + raw data
#   nvme03  -> code + output + logs
#   nvme04  -> mope-jepa pretrained checkpoint
# =============================================================================

# --- Project roots -----------------------------------------------------------
export SPACE_ROOT="/home/nvme03/wlx/Space_sensing/projects/space"
export SPACE_OUTPUT_ROOT="/home/nvme03/wlx/Space_sensing/output"
export SPACE_LOG_ROOT="/home/nvme03/wlx/Space_sensing/logs"

# --- Models ------------------------------------------------------------------
export QWEN3_VL_4B_PATH="/home/nvme01/wlx/Space_sensing/models/Qwen3-VL-4B-Instruct"
export VGGT_PATH="/home/nvme01/wlx/Space_sensing/models/VGGT-1B"
export MOPE_CKPT_PATH="/home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_1/checkpoint-199.pth"

# --- Training data (guide_repro) --------------------------------------------
export SPAR_234K_ANN="/home/nvme01/wlx/Space_sensing/data/guide_repro/train/spar_234k.json"
export LLAVA_HOUND_64K_ANN="/home/nvme01/wlx/Space_sensing/data/guide_repro/train/llava_hound_64k.json"
export GUIDE_DATA_ROOT="/home/nvme01/wlx/Space_sensing/data/guide_repro/media"

# --- Training data (VSI-590K preprocessed) ----------------------------------
export VSI590K_SPAR_ANN="/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_spar_590k.json"
export VSI590K_VIDEO_ANN="/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_video_590k.json"
export VSI590K_DATA_ROOT="/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/"

# --- Eval data (VSIBench) ----------------------------------------------------
export VSIBENCH_VIDEO_ROOT="/home/nvme01/wlx/Space_sensing/data/VSIBench"
export VSIBENCH_JSONL="/home/nvme01/wlx/Space_sensing/data/VSIBench/test.jsonl"

# --- Eval data (VLM4D) -------------------------------------------------------
# PENDING[D-11]: 论文2 评测范围基本定（VLM4D 动态主场），D-11 形式上仍 OPEN。
# VERIFIED 2026-06-06: 数据已下载。标注是 QA/real_mc.json（JSON 数组, 1371 条）。
#   视频在 VLM4D_VIDEO_ROOT/videos_real/{davis,ego4d,youtube-vos}/ 和 videos_synthetic/。
export VLM4D_VIDEO_ROOT="/home/nvme01/wlx/Space_sensing/data/VLM4D"
export VLM4D_JSONL="/home/nvme01/wlx/Space_sensing/data/VLM4D/QA/real_mc.json"
