#!/bin/bash
# =============================================================================
# E-10b WARM-GATE PROBE — "doubly-dead gate" falsification probe (~47 min run).
#
# WHAT THIS IS (and is NOT)
#   This is NOT a new experiment. It is a CHEAP FALSIFICATION PROBE for the
#   2026-06-17 "doubly-dead gate (双重死门)" root-cause decision (decisions.md):
#     The E-10b gate g·mope_out can't learn because mope_out itself is ≈0
#     (out_proj zero-init) -> d(loss)/d(g) = <upstream_grad, mope_out> ≈ 0, so the
#     gate has NO learning signal no matter how it is seeded. Fix B = move gate
#     learning into stage2 (train the gate only once the branch is warm/non-zero).
#
#   This probe tests the "branch-non-zero -> gate can move" half of that claim
#   the cheap way: start the gated crossattn projector from the ALREADY-WARM
#   E-03a checkpoint (out_proj trained to ~1.147, dynamic branch non-zero), add
#   the gate, and run a stage2-style JOINT short train. If the gate now has a live
#   gradient, g_std / MI / gate_mlp weights should START MOVING off their init.
#
# HOW TO READ IT (probe = watch the trend, then kill by hand)
#   Monitor the [gate-diag] lines (printed every 10 steps). Look for, over the
#   first ~100–300 steps:
#     g_std rising off ~0 ; MI rising ; logit_absmean leaving 0 ;
#     gate_grad_norm non-zero ; wlast_norm / w0_norm moving off init.
#   There is intentionally NO max_steps cap here (do not hard-stop): run it for
#   roughly 47 min / a few hundred steps, eyeball the [gate-diag] trend, then
#   KILL IT BY HAND. save_steps is small (200) + save_total_limit 1 only so a
#   short run still leaves one checkpoint; we do not need the run to finish.
#   - Probe SUCCEEDS  -> gate moves once branch is non-zero -> invest in the real
#                        two-stage (stage1 warms branch, stage2 learns gate).
#   - Probe FAILS     -> gate inert even with a warm branch -> doubly-dead-gate
#                        hypothesis falsified, look elsewhere.
#
# RELATION TO train_e10b_router_v1_gatefix.sh
#   Body = that script's STAGE 2 (TUNE_MM_LLM=True, warm-start JOINT training)
#   config, with the v2.2 gate hyperparameters byte-for-byte identical:
#     init_bias=0.0  lastw_std=8.0  zloss=0.0  entropy_coef=1e-2  warmup=0
#     anticollapse=True  diag_every=10
#   THE ONLY meaningful differences vs gatefix stage2 are:
#     1. START CKPT = E-03a checkpoint-4000 (GUIDE_CKPT_PATH below), NOT the
#        E-10b stage1 dir. This is the warm out_proj≈1.147 source.
#     2. --load_mope_projector_from_ckpt True : because the projector is attached
#        AFTER from_pretrained, HF silently drops the checkpoint's
#        model._mope_projector.* keys; this flag re-loads k/v/out_proj from the
#        E-03a ckpt (strict=False) WHILE keeping the projector TRAINABLE and
#        letting the (absent-in-E-03a) gate_mlp keep its fresh MI-seed init.
#        Without it the probe would start with a dead (zero) branch — defeating
#        the whole point.
#     3. INDEPENDENT output dir e10b_probe_warmgate_4b — NEVER reuses E-03a or
#        E-10b dirs (must not pollute existing results).
#     4. RESUME only from THIS probe's own output dir (never from E-03a/E-10b).
#     5. save_steps 200 (small) — short probe run only.
#
# Architecture (same as E-10b gatefix stage2):
#   - Static expert  = frozen VGGT geometry stream (D-14 方案A).
#   - Dynamic expert = frozen MoPE (last layer, E-03a config).
#   - Gate g         = learned content-driven scalar gate (D-15 b): image_embeds + g*out.
#   - No auxiliary task loss; MoPE layer selection unchanged.
#   - tune_mm_llm True (warm-start JOINT, projector + gate + LLM all trainable).
#
# Usage:
#   MODEL_SIZE=4b bash train_e10b_probe_warmgate.sh   # default
#
# Supported MODEL_SIZE values: 4b  8b
# Output dir: e10b_probe_warmgate_{size}
#
# RUN-BEFORE CHECK (server huirui): the E-03a start checkpoint must exist:
#   ${SPACE_OUTPUT_ROOT}/train/e03a_mope_crossattn_two_stage_4b/checkpoint-4000
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env/activate.sh"

# ---------------------------------------------------------------------------
# Model size (no TRAIN_STAGE switch: the probe is intrinsically stage2-style
# JOINT training from a warm E-03a ckpt).
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}

# ---------------------------------------------------------------------------
# Distributed training configuration (huirui 8-GPU shared; lock to 0-3, NPROC=4).
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Single-node training only; multi-node not supported.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ---------------------------------------------------------------------------
# NCCL runtime workarounds (new server huirui: driver 570 / CUDA 12.8,
# torch 2.6.0+cu124, NCCL 2.28.9). Copied verbatim from
# train_e10b_router_v1_gatefix.sh — required for multi-card DeepSpeed on huirui.
# ---------------------------------------------------------------------------
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# Force the pip-installed NCCL (torch 2.6 配套版本); prevents a rogue system /
# other-conda libnccl.so.2 (2.28.9) from shadowing pip's and causing
# `Cuda failure 'driver insufficient'` in multi-GPU NCCL init.
NCCL_SO="${NCCL_SO:-/data1/miniconda3/envs/space/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2}"
if [ ! -f "${NCCL_SO}" ]; then
    NCCL_SO=$(python - <<'PY' 2>/dev/null
import os, sys, glob
c = glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "nccl", "lib", "libnccl.so.2"))
print(c[0] if c else "")
PY
)
fi
# 自清洗：剔除会盖掉 pip nccl/cudnn 的已知污染目录(cuda-12.8 / videorepa)
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE 'cuda-12\.8|videorepa' | paste -sd: -)
if [ -n "${NCCL_SO}" ] && [ -f "${NCCL_SO}" ]; then
    NCCL_LIB_DIR=$(dirname "${NCCL_SO}")
    export LD_LIBRARY_PATH="${NCCL_LIB_DIR}:${LD_LIBRARY_PATH}"   # 主机制：torchrun 会把它传给 worker
    export LD_PRELOAD="${NCCL_SO}:${LD_PRELOAD}"                  # 父进程双保险
    echo "[nccl] forcing pip NCCL: ${NCCL_SO}"
    echo "[nccl] torch.cuda.nccl.version() -> $(python -c 'import torch;print(torch.cuda.nccl.version())' 2>/dev/null)"
else
    echo "[nccl] WARNING: pip nvidia-nccl-cu12 libnccl.so.2 not found via sys.prefix glob (got NCCL_SO='${NCCL_SO}'); NCCL may load a shadowed system 2.28.9." >&2
fi

# ---------------------------------------------------------------------------
# Path configuration (override via env vars as needed)
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/home/nvme03/wlx/Space_sensing/projects/space"}
GUIDE_ROOT="${SPACE_ROOT}/src"
MOPE_ROOT="${SPACE_ROOT}/src/vendor/mope"

VGGT_PATH=${VGGT_PATH:-/home/nvme01/wlx/Space_sensing/models/VGGT-1B}

# MoPE checkpoint (ep199, vitb_1 full training run)
MOPE_CKPT_PATH=${MOPE_CKPT_PATH:-/home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_1/checkpoint-199.pth}

# Path to MoPE source code (for encoder loading)
MOPE_CODE_PATH=${MOPE_CODE_PATH:-${SPACE_ROOT}/src/vendor/mope}

# ---------------------------------------------------------------------------
# E-10b gate anti-collapse hyperparameters — BYTE-FOR-BYTE identical to
# train_e10b_router_v1_gatefix.sh (v2.2). Overridable via env for ablations.
# ---------------------------------------------------------------------------
GATE_INIT_BIAS=${GATE_INIT_BIAS:-0.0}             # A1: g starts ≈0.5 (non-saturated)
GATE_LASTW_STD=${GATE_LASTW_STD:-8.0}             # v2.2: final-layer weight init std = content seed
GATE_ZLOSS_COEF=${GATE_ZLOSS_COEF:-0.0}           # A2: z-loss OFF by default
GATE_ENTROPY_COEF=${GATE_ENTROPY_COEF:-1e-2}      # A3: lambda_max for MI objective
GATE_ENTROPY_WARMUP=${GATE_ENTROPY_WARMUP:-0}     # v2.2: NO warmup (MI at full strength from step 0)
GATE_DIAG_EVERY=${GATE_DIAG_EVERY:-10}            # [gate-diag] interval (every 10 steps)

# ---------------------------------------------------------------------------
# Per-size configuration. Global batch is pinned to E-03a/E-10 (target=48 for 4B,
# 96 for 8B) and grad_accum is DERIVED from the GPU count, so global batch is
# constant regardless of NPROC (the warm-gate probe stays the only variable).
# ---------------------------------------------------------------------------
if [ "${MODEL_SIZE}" = "4b" ]; then
    batch_size=2
    TARGET_GLOBAL_BATCH=48
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    TARGET_GLOBAL_BATCH=96
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    echo "WARNING: 8B E-10b probe is experimental — monitor VRAM usage carefully." >&2
else
    echo "ERROR: Unknown MODEL_SIZE='${MODEL_SIZE}'. Must be '4b' or '8b'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Derive gradient_accumulation_steps so the GLOBAL batch stays aligned with
# E-03a/E-10 regardless of GPU count.
# ---------------------------------------------------------------------------
per_step_batch=$((batch_size * NPROC_PER_NODE))
if [ -n "${GRAD_ACCUM_STEPS:-}" ]; then
    grad_accum_steps=${GRAD_ACCUM_STEPS}
    global_batch=$((per_step_batch * grad_accum_steps))
    echo "NOTE: GRAD_ACCUM_STEPS overridden via env -> accum=${grad_accum_steps}, global_batch=${global_batch} (target=${TARGET_GLOBAL_BATCH})." >&2
else
    if [ $((TARGET_GLOBAL_BATCH % per_step_batch)) -ne 0 ]; then
        echo "ERROR: TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH} is not divisible by batch_size*NPROC_PER_NODE=${batch_size}*${NPROC_PER_NODE}=${per_step_batch}." >&2
        echo "       This GPU count cannot evenly hit the target global batch. Change NPROC_PER_NODE, or set GRAD_ACCUM_STEPS explicitly to override." >&2
        exit 1
    fi
    grad_accum_steps=$((TARGET_GLOBAL_BATCH / per_step_batch))
    global_batch=$((per_step_batch * grad_accum_steps))
fi

# ---------------------------------------------------------------------------
# Probe configuration: warm START checkpoint, tune_mm_llm, INDEPENDENT output dir.
#   - START = E-03a checkpoint-4000 (out_proj warm ~1.147). Overridable via env
#     GUIDE_CKPT_PATH; defaults to the E-03a ckpt, NOT the E-10b stage1 dir.
#   - tune_mm_llm True (stage2-style warm-start JOINT training).
#   - --load_mope_projector_from_ckpt True re-loads the projector from the E-03a
#     ckpt (strict=False) and keeps it trainable (see header).
# ---------------------------------------------------------------------------
TUNE_MM_LLM=True
GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e03a_mope_crossattn_two_stage_${MODEL_SIZE}/checkpoint-4000}
output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e10b_probe_warmgate_${MODEL_SIZE}}"
run_name="space_e10b_probe_warmgate_${MODEL_SIZE}_lr1e-5"
exp_name="e10b_probe_warmgate_${MODEL_SIZE}"

# Resume support: ONLY auto-resume from THIS probe's own output dir (never from
# E-03a/E-10b). If absent, degrade to a fresh run (no resume).
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${output_dir}/checkpoint-200}
[ ! -d "${RESUME_FROM_CHECKPOINT}" ] && RESUME_FROM_CHECKPOINT=""

# ---------------------------------------------------------------------------
# Log directory
# ---------------------------------------------------------------------------
LOG_DIR=${LOG_DIR:-${SPACE_LOG_ROOT}/train}
mkdir -p "${LOG_DIR}"
mkdir -p "${output_dir}"

# ---------------------------------------------------------------------------
# PYTHONPATH: our data/__init__.py must shadow GUIDE's, so it comes first
# ---------------------------------------------------------------------------
export PYTHONPATH="${SPACE_ROOT}/src/train_framework:${SPACE_ROOT}:${GUIDE_ROOT}:${MOPE_ROOT}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# VSI-590K dataset paths (read by src/train_framework/data/__init__.py)
# ---------------------------------------------------------------------------
export VSI590K_SPAR_ANN=${VSI590K_SPAR_ANN:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_spar_590k.json}
export VSI590K_VIDEO_ANN=${VSI590K_VIDEO_ANN:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/vsi590k_video_590k.json}
export VSI590K_DATA_ROOT=${VSI590K_DATA_ROOT:-/home/nvme01/wlx/Space_sensing/data/vsi590k_processed/}

# ---------------------------------------------------------------------------
# Weights & Biases (optional — leave WANDB_API_KEY unset to disable)
# ---------------------------------------------------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
fi

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
lr=1e-5

# ---------------------------------------------------------------------------
# Entry point: our fork of the training framework
# ---------------------------------------------------------------------------
entry_file="${SPACE_ROOT}/src/train_framework/train_space.py"

# ---------------------------------------------------------------------------
# Training arguments (= E-10b gatefix stage2 + the warm-gate probe deltas:
# load_mope_projector_from_ckpt True, save_steps 200, warm E-03a start ckpt).
# ---------------------------------------------------------------------------
args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${GUIDE_CKPT_PATH} \
    --dataset_use vsi590k_spar \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm ${TUNE_MM_LLM} \
    --optim adamw_torch \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size * 2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 268324 \
    --min_pixels 8192 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 12800 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to none \
    --use_geometry_inputs True \
    --use_geometry_encoder True \
    --use_feature_fusion_module True \
    --use_patch_size_alin True \
    --geometry_deepstack_indexes_pro 7:0,10:1,13:2,16:3,19:4,22:5 \
    --use_deepstack_importance_gate all \
    --use_deepstack_global_gate all \
    --geometry_encoder_type vggt \
    --geometry_encoder_path ${VGGT_PATH} \
    --use_mope True \
    --mope_fusion_mode crossattn \
    --load_mope_projector_from_ckpt True \
    --mope_use_gate True \
    --mope_gate_mode learned \
    --mope_gate_anticollapse True \
    --mope_gate_init_bias ${GATE_INIT_BIAS} \
    --mope_gate_lastw_std ${GATE_LASTW_STD} \
    --mope_gate_zloss_coef ${GATE_ZLOSS_COEF} \
    --mope_gate_entropy_coef ${GATE_ENTROPY_COEF} \
    --mope_gate_entropy_warmup_steps ${GATE_ENTROPY_WARMUP} \
    --mope_gate_diag_every ${GATE_DIAG_EVERY} \
    --gate_log_interval 25 \
    --mope_checkpoint_path ${MOPE_CKPT_PATH} \
    --mope_encoder_path ${MOPE_CODE_PATH} \
    --mope_all_frames 8 \
    --group_by_modality_length True \
    ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}}"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

echo "=== E-10b WARM-GATE PROBE (MODEL_SIZE=${MODEL_SIZE}) ==="
echo "Probe        : doubly-dead-gate falsification — warm branch (E-03a) + gate, stage2 joint"
echo "Start ckpt   : ${GUIDE_CKPT_PATH}  (out_proj warm ~1.147; load via --load_mope_projector_from_ckpt)"
echo "Output       : ${output_dir}  (INDEPENDENT — never E-03a/E-10b dirs)"
echo "Log          : ${LOG_FILE}"
echo "Fusion       : crossattn + learned content-driven gate, batch=${batch_size}, accum=${grad_accum_steps}, NPROC=${NPROC_PER_NODE}, global_batch=${global_batch} (target=${TARGET_GLOBAL_BATCH})"
echo "Gate fix     : v2.2 (== gatefix) | anticollapse=True init_bias=${GATE_INIT_BIAS} lastw_std=${GATE_LASTW_STD}(seed) zloss=${GATE_ZLOSS_COEF}(OFF) entropy=MI(coef=${GATE_ENTROPY_COEF}) warmup=${GATE_ENTROPY_WARMUP} diag_every=${GATE_DIAG_EVERY}"
echo "Trainable    : LLM + MoPEProjectorCrossAttn + gate (warm-start JOINT, stage2-style)"
echo "HOW TO STOP  : no max_steps cap by design — watch [gate-diag] g_std/MI/gate weights for ~47min/few-hundred steps, then KILL BY HAND."

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
