#!/bin/bash
# =============================================================================
# E-10b: Router v1.1 — gate anti-collapse ("gatefix") version of E-10.
#
# ARCHITECTURE / TWO-STAGE WARM-START RECIPE = IDENTICAL TO E-10.
# The ONLY difference vs E-10 is the gate fix bundle, so "E-10b vs E-10 differs
# only by the gate fix" is the single variable (D-17 naming):
#   v2.2 = SEED-INIT FIX: the gate MLP final-layer weight init std was near-zero
#   (1e-3) so the init logit≈0 with only ~1e-4 per-sample divergence -> MI had no
#   seed to amplify and the grad to the content-read layer W1 was choked to ~0
#   (W1 frozen). v2.2 enlarges that std into a real content seed AND drops the MI
#   warmup so MI runs at full strength from step 0.
#   A1 non-saturated gate init : --mope_gate_init_bias 0.0  (g starts ≈0.5)
#                                 (E-10 used +4.0 → g≈0.98 saturated → collapse)
#   v2.2 last-layer seed       : --mope_gate_lastw_std 0.5  (lastw_std=0.5 seed)
#                                 (E-10 used 1e-3 ≈ near-zero -> logit≈0, no seed)
#                                 init g spreads to ~0.3–0.7 -> MI has content to
#                                 amplify and the gradient reaches W1.
#   A2 gate logit z-loss       : OFF (coef=0) by default. v2's always-on z-loss
#                                 pulled the gate to the constant g=0.5; the MI
#                                 objective is self-saturating so z-loss is now a
#                                 disabled optional guard (--mope_gate_zloss_coef 0.0)
#   A3 mutual-information +warmup: --mope_gate_entropy_coef 1e-2
#                                 --mope_gate_entropy_warmup_steps 0  (v2.2: NO
#                                 warmup, MI at full strength from step 0)
#                                 MI = marginal entropy − mean per-sample entropy
#                                 (RIM/IMSAT form); replaces v2's Bernoulli
#                                 batch-mean entropy, whose stable max at g=0.5
#                                 could not pry the gate open
#   anti-collapse master switch: --mope_gate_anticollapse True
#   rich [gate-diag] training log: --mope_gate_diag_every 10  (print every 10 steps)
#
# All new args default to E-10's status quo in argument.py, so E-03a/E-02c/E-10
# scripts are unchanged; this script turns the fix ON explicitly.
#
# Architecture (论文2 R1, D-16/D-17), same as E-10:
#   - Static expert  = frozen VGGT geometry stream (D-14 方案A).
#   - Dynamic expert = frozen MoPE (last layer, E-03a config).
#   - Gate g         = learned content-driven scalar gate (D-15 b): image_embeds + g*out.
#   - No auxiliary task loss; MoPE layer selection unchanged.
#
# Two-stage training (same recipe as E-10/E-03a, D-09/D-10):
#   - Stage 1 (TRAIN_STAGE=stage1): freeze LLM, train projector + gate.
#       start = E-00b projector-only checkpoint.
#       --tune_mm_llm False
#   - Stage 2 (TRAIN_STAGE=stage2, DEFAULT): warm-start joint training.
#       start = E-10b Stage-1 checkpoint.
#       --tune_mm_llm True
#
# Usage:
#   TRAIN_STAGE=stage1 MODEL_SIZE=4b bash train_e10b_router_v1_gatefix.sh
#   TRAIN_STAGE=stage2 MODEL_SIZE=4b bash train_e10b_router_v1_gatefix.sh   # default
#
# Supported MODEL_SIZE values: 4b  8b
# Output dir: e10b_router_v1_{size} (stage2) / e10b_router_v1_stage1_{size}
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../env/activate.sh"

# ---------------------------------------------------------------------------
# Model size + stage switch
# ---------------------------------------------------------------------------
MODEL_SIZE=${MODEL_SIZE:-4b}
TRAIN_STAGE=${TRAIN_STAGE:-stage2}   # stage1 | stage2 (default: stage2 warm-start)

# ---------------------------------------------------------------------------
# Distributed training configuration
# ---------------------------------------------------------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Single-node training only; multi-node not supported.
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# ---------------------------------------------------------------------------
# NCCL runtime workarounds (new server huirui: driver 570 / CUDA 12.8,
# torch 2.6.0+cu124, NCCL 2.28.9). Copied verbatim from train_e10_router_v1.sh
# — required for multi-card DeepSpeed on the new server. See that script for
# the full root-cause notes.
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
# E-10b gate anti-collapse hyperparameters (the ONLY thing that differs from
# E-10). All overridable via env for ablations.
# ---------------------------------------------------------------------------
GATE_INIT_BIAS=${GATE_INIT_BIAS:-0.0}             # A1: g starts ≈0.5 (non-saturated)
GATE_LASTW_STD=${GATE_LASTW_STD:-0.5}             # v2.2: final-layer weight init std = content seed (E-10 used 1e-3 ≈ near-zero)
GATE_ZLOSS_COEF=${GATE_ZLOSS_COEF:-0.0}           # A2: z-loss OFF by default (set >0 to guard logit runaway)
GATE_ENTROPY_COEF=${GATE_ENTROPY_COEF:-1e-2}      # A3: lambda_max for MI objective
GATE_ENTROPY_WARMUP=${GATE_ENTROPY_WARMUP:-0}     # v2.2: NO warmup (MI at full strength from step 0)
GATE_DIAG_EVERY=${GATE_DIAG_EVERY:-10}            # change B: [gate-diag] interval (every 10 steps)

# ---------------------------------------------------------------------------
# Per-size configuration. Global batch is pinned to E-03a/E-10 (target=48 for 4B,
# 96 for 8B) and grad_accum is DERIVED from the GPU count, so global batch is
# constant regardless of NPROC (the gate fix stays the only variable vs E-10).
# ---------------------------------------------------------------------------
if [ "${MODEL_SIZE}" = "4b" ]; then
    batch_size=2
    TARGET_GLOBAL_BATCH=48
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero2.json}
elif [ "${MODEL_SIZE}" = "8b" ]; then
    batch_size=1
    TARGET_GLOBAL_BATCH=96
    DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${SPACE_ROOT}/configs/zero3.json}
    echo "WARNING: 8B E-10b is experimental — monitor VRAM usage carefully." >&2
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
# Stage-specific configuration: start checkpoint, tune_mm_llm, output dir.
# Stage 1: freeze LLM, train projector+gate, start from E-00b.
# Stage 2: warm-start joint, train LLM+projector+gate, start from E-10b Stage1.
# ---------------------------------------------------------------------------
if [ "${TRAIN_STAGE}" = "stage1" ]; then
    TUNE_MM_LLM=False
    # Stage 1 starts from the E-00b projector-only checkpoint (same source as E-10/E-03a Stage 1).
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e00b_mope_projector_only_${MODEL_SIZE}}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e10b_router_v1_stage1_${MODEL_SIZE}}"
    run_name="space_e10b_router_v1_stage1_${MODEL_SIZE}_lr1e-5"
    exp_name="e10b_router_v1_stage1_${MODEL_SIZE}"
elif [ "${TRAIN_STAGE}" = "stage2" ]; then
    TUNE_MM_LLM=True
    # Stage 2 (warm-start joint) starts from the E-10b Stage-1 checkpoint.
    GUIDE_CKPT_PATH=${GUIDE_CKPT_PATH:-${SPACE_OUTPUT_ROOT}/train/e10b_router_v1_stage1_${MODEL_SIZE}}
    output_dir="${OUTPUT_DIR:-${SPACE_OUTPUT_ROOT}/train/e10b_router_v1_${MODEL_SIZE}}"
    run_name="space_e10b_router_v1_${MODEL_SIZE}_lr1e-5"
    exp_name="e10b_router_v1_${MODEL_SIZE}"
else
    echo "ERROR: Unknown TRAIN_STAGE='${TRAIN_STAGE}'. Must be 'stage1' or 'stage2'." >&2
    exit 1
fi

# Resume support (only auto-resume from this run's own output dir).
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-${output_dir}/checkpoint-4000}
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
# Training arguments
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
    --save_steps 1000 \
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

echo "=== E-10b Router v1.1 gatefix Training (MODEL_SIZE=${MODEL_SIZE}, STAGE=${TRAIN_STAGE}) ==="
echo "Start ckpt   : ${GUIDE_CKPT_PATH}"
echo "Output       : ${output_dir}"
echo "Log          : ${LOG_FILE}"
echo "Fusion       : crossattn + learned content-driven gate, batch=${batch_size}, accum=${grad_accum_steps}, NPROC=${NPROC_PER_NODE}, global_batch=${global_batch} (target=${TARGET_GLOBAL_BATCH})"
echo "Gate fix     : v2.2 seed-init | anticollapse=True init_bias=${GATE_INIT_BIAS} lastw_std=${GATE_LASTW_STD}(seed) zloss=${GATE_ZLOSS_COEF}(OFF) entropy=MI(coef=${GATE_ENTROPY_COEF}) warmup=${GATE_ENTROPY_WARMUP} diag_every=${GATE_DIAG_EVERY}"
if [ "${TRAIN_STAGE}" = "stage1" ]; then
    echo "Trainable    : MoPEProjectorCrossAttn + gate (LLM frozen, Stage 1)"
else
    echo "Trainable    : LLM + MoPEProjectorCrossAttn + gate (warm-start joint, Stage 2)"
fi

python -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} \
         > "${LOG_FILE}" 2>&1
