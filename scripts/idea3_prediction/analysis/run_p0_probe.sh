#!/bin/bash
# =============================================================================
# run_p0_probe.sh — P-0 linear-probe 对照 runner (paper2_design §4.1 / D-20)
#
# 依次对若干 checkpoint 跑 p0_linear_probe.py（"从 LLM 每帧 hidden 线性还原
# 绝对时间位置"的可还原度），把各自 JSON 合并成一个对照表打印。
#
# 解读（详见 p0_linear_probe.py 头部）：
#   干净判据 = E-16 vs E-03a（都 MoPE-ON，唯一变量=辅助头）：
#     E-16 的 bin_acc / pos_r2 显著 < E-03a（且 E-16 接近 chance=0.25）
#       → 位置擦除 → 走 E-16c（目标端补绝对时间位置）。
#     E-16 与 E-03a 相近 → 负迁移 → 走调 loss 权重 / E-16d 顺序头。
#   ⚠ E-15 vs E-03a 仅作 WARN-1 无泄漏臂旁证（E-15=MoPE-off，含"辅助头+MoPE-off"
#     双 confound），不并入干净判据。
#
# GPU 注意：轻量 inference，但需一张 GPU。若训练占满 cards 0-3，先
#   `export CUDA_VISIBLE_DEVICES=<空闲卡>`（脚本内 --device cuda:0 指进程内第0卡），
#   或等训练完再跑。
#
# 用法：
#   bash run_p0_probe.sh
#   NUM_VIDEOS=150 FRAMES=8 CUDA_VISIBLE_DEVICES=4 bash run_p0_probe.sh
# =============================================================================
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../../_common/env/activate.sh"

# --- New-server runtime workaround: drop stray videorepa cuDNN paths --------
if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v 'videorepa' | paste -sd: -)"
fi

# ---------------------------------------------------------------------------
# Paths (env-derived; same vars as eval_e16_vsibench.sh / profile_*.sh).
# ---------------------------------------------------------------------------
SPACE_ROOT=${SPACE_ROOT:-"/data2/wlx/projects/space"}
GUIDE_LMMS_EVAL=${GUIDE_LMMS_EVAL:-${SPACE_ROOT}/src/vendor/lmms-eval}
GUIDE_TRAIN_ROOT=${SPACE_ROOT}/src

# VSI-Bench videos (probe reuses them; path = eval script default, overridable).
# Layout: ${VSIBENCH_VIDEO_ROOT}/{arkitscenes,scannet,scannetpp}/*.mp4
VSIBENCH_VIDEO_ROOT=${VSIBENCH_VIDEO_ROOT:-/data2/wlx/data/VSIBench}

# Output dir for per-checkpoint JSONs + merged table.
OUT_DIR=${OUT_DIR:-${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}/analysis/p0_probe}
mkdir -p "${OUT_DIR}"

# Knobs
NUM_VIDEOS=${NUM_VIDEOS:-150}
FRAMES=${FRAMES:-8}
SEED=${SEED:-0}
DEVICE=${DEVICE:-cuda:0}
ATTN=${ATTN:-flash_attention_2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---------------------------------------------------------------------------
# Checkpoints to probe.  model_type: crossattn = MoPE-ON (E-03a / E-16);
# qwen3_vl_my = MoPE-OFF (E-15 predict-only, matches its eval).
# Each entry: "TAG|MODEL_TYPE|CKPT_PATH".  E-15 added once trained (path guessed
# from train_e15_lfp_predonly.sh output_dir).
# ---------------------------------------------------------------------------
SPACE_OUTPUT_ROOT=${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}
CKPT_E03A=${CKPT_E03A:-${SPACE_OUTPUT_ROOT}/train/e03a_mope_crossattn_two_stage_4b}
CKPT_E16=${CKPT_E16:-${SPACE_OUTPUT_ROOT}/train/e16_lfp_both_4b}
CKPT_E15=${CKPT_E15:-${SPACE_OUTPUT_ROOT}/train/e15_lfp_predonly_4b}

ENTRIES=(
    "E-03a|qwen3_vl_mope_crossattn|${CKPT_E03A}"
    "E-16|qwen3_vl_mope_crossattn|${CKPT_E16}"
)
# E-15 only if the checkpoint exists (trained later).
if [ -d "${CKPT_E15}" ]; then
    ENTRIES+=("E-15|qwen3_vl_my|${CKPT_E15}")
else
    echo "[run_p0] NOTE: E-15 checkpoint not found (${CKPT_E15}); skipping. Re-run after E-15 finishes, or set CKPT_E15=..."
fi

# ---------------------------------------------------------------------------
# Env for imports — identical to eval scripts.
# ---------------------------------------------------------------------------
export LMMS_EVAL_PLUGINS="src.eval"
export PYTHONPATH="${GUIDE_LMMS_EVAL}:${GUIDE_TRAIN_ROOT}:${SPACE_ROOT}:${PYTHONPATH}"
export NCCL_NVLS_ENABLE=0

PROBE_PY="${SPACE_ROOT}/scripts/idea3_prediction/analysis/p0_linear_probe.py"

echo "=== P-0 linear-probe runner ==="
echo "Video root : ${VSIBENCH_VIDEO_ROOT}"
echo "Num videos : ${NUM_VIDEOS}  Frames: ${FRAMES}  Seed: ${SEED}"
echo "Device     : ${DEVICE}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "Out dir    : ${OUT_DIR}"
echo "================================"

JSON_PATHS=()
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r TAG MTYPE CKPT <<< "${entry}"
    if [ ! -d "${CKPT}" ]; then
        echo "[run_p0] SKIP ${TAG}: checkpoint dir not found: ${CKPT}"
        continue
    fi
    OUT_JSON="${OUT_DIR}/p0_${TAG//-/}.json"
    echo ""
    echo ">>> Probing ${TAG} (${MTYPE}) : ${CKPT}"
    # W2: protect each probe under `set -e` so one checkpoint's non-zero exit
    # does NOT abort the whole comparison. Run in a subshell (|| true) and only
    # register the JSON on success, so partial runs still produce a table.
    if python "${PROBE_PY}" \
        --checkpoint "${CKPT}" \
        --model_type "${MTYPE}" \
        --video_dir "${VSIBENCH_VIDEO_ROOT}" \
        --num_videos "${NUM_VIDEOS}" \
        --frames "${FRAMES}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --attn_implementation "${ATTN}" \
        --space_root "${SPACE_ROOT}" \
        --output "${OUT_JSON}"; then
        JSON_PATHS+=("${OUT_JSON}")
    else
        echo "[run_p0] FAILED ${TAG} (exit $?); skipping, continuing with remaining checkpoints."
    fi
done

# ---------------------------------------------------------------------------
# Merge all per-checkpoint JSONs into one comparison table + combined JSON.
# ---------------------------------------------------------------------------
MERGED="${OUT_DIR}/p0_compare.json"
echo ""
echo "=== P-0 comparison ==="
python - "$MERGED" "${JSON_PATHS[@]}" <<'PY'
import json, sys
merged_path, paths = sys.argv[1], sys.argv[2:]
rows = []
for p in paths:
    try:
        with open(p) as fh:
            rows.append(json.load(fh))
    except Exception as e:
        print(f"[warn] could not read {p}: {e}")
with open(merged_path, "w") as fh:
    json.dump(rows, fh, indent=2)
hdr = f"{'checkpoint':40s} {'model_type':26s} {'n_vid':>6s} {'n_frm':>7s} {'bin_acc':>8s} {'macroF1':>8s} {'pos_r2':>8s} {'chance':>7s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    ck = r.get('checkpoint', '')
    ck = ck if len(ck) <= 40 else "..." + ck[-37:]
    print(f"{ck:40s} {r.get('model_type',''):26s} "
          f"{r.get('n_videos',0):6d} {r.get('n_frames_total',0):7d} "
          f"{r.get('bin_acc',float('nan')):8.4f} {r.get('bin_macro_f1',float('nan')):8.4f} "
          f"{r.get('pos_r2',float('nan')):8.4f} {r.get('chance_acc',0.25):7.4f}")

# --- collapse diagnostics sub-table (ADD-ONLY; backward compatible) ---------
print()
print("=== Collapse diagnostics (E-16 vs E-03a; on per-frame hidden X) ===")
hdr2 = (f"{'checkpoint':40s} {'stbl_rank':>9s} {'eff_rank':>9s} "
        f"{'frac_lowvar':>11s} {'vicreg_var':>10s} {'vicreg_cov':>10s} {'mean_std':>9s}")
print(hdr2); print("-" * len(hdr2))
for r in rows:
    ck = r.get('checkpoint', '')
    ck = ck if len(ck) <= 40 else "..." + ck[-37:]
    print(f"{ck:40s} {r.get('stable_rank',float('nan')):9.3f} "
          f"{r.get('effective_rank',float('nan')):9.3f} "
          f"{r.get('frac_low_var_dims',float('nan')):11.4f} "
          f"{r.get('vicreg_var_term',float('nan')):10.4f} "
          f"{r.get('vicreg_cov_term',float('nan')):10.4f} "
          f"{r.get('hidden_mean_std',float('nan')):9.3f}")
print("坍缩解读: E-16 的 stable_rank/effective_rank ≈ E-03a、frac_lowvar 不升、")
print("          vicreg 两项 ≈ E-03a => 无额外坍缩 => E-16b(VICReg 防坍缩)不需要;")
print("          反之(rank 明显更低 / frac_lowvar 升高 / vicreg 两项更大)才需要 E-16b。")
print()
print("解读(干净判据 = E-16 vs E-03a, 二者都 MoPE-ON, 唯一变量=辅助头):")
print("  E-16 的 bin_acc/pos_r2 显著 < E-03a (且 E-16 接近 chance=0.25) => 位置擦除 => E-16c;")
print("  E-16 与 E-03a 相近 => 负迁移 => 调 loss 权重 / E-16d 顺序头。")
print("警告: E-15 vs E-03a 仅 WARN-1 无泄漏臂旁证 (E-15=MoPE-off, 含'辅助头+MoPE-off'双 confound),")
print("      不并入干净判据 -- 其低可还原度可能部分来自 MoPE-off 而非辅助头。")
print(f"\nMerged JSON: {merged_path}")
PY

echo ""
echo "=== Done. Per-checkpoint JSONs + p0_compare.json in ${OUT_DIR} ==="
