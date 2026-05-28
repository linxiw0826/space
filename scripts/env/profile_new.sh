#!/bin/bash
# =============================================================================
# New server profile (user@112.91.161.190:12013, single /data2/wlx root).
#
# Sourced by activate.sh. Mirrors old server's semantic layout:
#   /data2/wlx/
#     projects/space/       <- git clone
#     models/               <- Qwen3-VL, VGGT, mope/
#     data/                 <- VSIBench, guide_repro
#     output/{train,eval}/
#     logs/{train,eval}/
# =============================================================================

# --- Project roots -----------------------------------------------------------
export SPACE_ROOT="/data2/wlx/projects/space"
export SPACE_OUTPUT_ROOT="/data2/wlx/output"
export SPACE_LOG_ROOT="/data2/wlx/logs"

# --- Models ------------------------------------------------------------------
export QWEN3_VL_4B_PATH="/data2/wlx/models/Qwen3-VL-4B-Instruct"
export VGGT_PATH="/data2/wlx/models/VGGT-1B"
export MOPE_CKPT_PATH="/data2/wlx/models/mope/checkpoint-199.pth"

# --- Training data (guide_repro) --------------------------------------------
export SPAR_234K_ANN="/data2/wlx/data/guide_repro/train/spar_234k.json"
export LLAVA_HOUND_64K_ANN="/data2/wlx/data/guide_repro/train/llava_hound_64k.json"
export GUIDE_DATA_ROOT="/data2/wlx/data/guide_repro/media"

# --- Training data (VSI-590K preprocessed) ----------------------------------
export VSI590K_SPAR_ANN="/data2/wlx/data/vsi590k_processed/vsi590k_spar_590k.json"
export VSI590K_VIDEO_ANN="/data2/wlx/data/vsi590k_processed/vsi590k_video_590k.json"
export VSI590K_DATA_ROOT="/data2/wlx/data/vsi590k_processed/"

# --- Eval data (VSIBench) ----------------------------------------------------
export VSIBENCH_VIDEO_ROOT="/data2/wlx/data/VSIBench"
export VSIBENCH_JSONL="/data2/wlx/data/VSIBench/test.jsonl"
