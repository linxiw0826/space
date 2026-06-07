"""mope_probe_layers.py — per-layer linear probe over MoPE features (验证③ / D-13).

服务于 D-13 取层决策（MoPE 取层 + 融合算子）。
================================================================================
背景
----
读 `mope_layer_features.py` 产出的 `features.npy [N, n_layers, 768]` +
`labels.npy [N]`，对**每一层**单独训一个 sklearn `LogisticRegression`（多类，
标签=运动类别），用 StratifiedKFold 交叉验证算 accuracy / macro-F1。输出每层
acc 曲线，定位峰值层（"Physics Emergence Zone"），并量化**末层 vs 峰值层**的差距
——给出"是否值得改抽中层"的信号。验证 FAIR《Interpreting Physics》(2602.07050)
结论是否在 MoPE 上成立（物理/方向信息中层峰值、向输出层退化）。

GPU / CPU 哪步需要
------------------
本脚本**纯 CPU**，只依赖 numpy + sklearn，登录节点直接跑。
（特征抽取那步 `mope_layer_features.py` 才需要 GPU。）

输入数据来自哪
--------------
`--feat-dir` 指向 `mope_layer_features.py` 的输出目录（含 features.npy /
labels.npy / meta.json）。标签建议来自 VLM4D 运动类别。

用法
----
    export PYTHONPATH=/u/lwu9/Space_sensing/projects/space:$PYTHONPATH  # so `src` is importable
    python -m src.analysis.mope_probe_layers \\
        --feat-dir /abs/path/out/mope_layer_features \\
        --folds 5 \\
        --out-csv /abs/path/out/probe_acc_by_layer.csv \\
        --out-png /abs/path/out/probe_acc_by_layer.png   # 可选，需 matplotlib

约束
----
- 纯 CPU、只依赖 numpy + sklearn（matplotlib 仅在 --out-png 时按需 import）。
- 不修改任何 vendor 代码。
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PENDING[D-13]: probe 在所有 n_layers 层上各训一个线性探针并对比末层 vs 峰值层。
# D-13 取层决策（MoPE 改抽哪一层）正是本脚本要提供的证据；解除后此脚本无需改动，
# 其输出（峰值层 idx + 末层差距）即为决策依据。
N_CLASS_MIN = 2  # 少于 2 类无法做分类探针


def _fit_layer_probe(X, y, n_splits, seed):
    """Train a multi-class LogisticRegression on one layer's features via CV.

    Returns (mean_acc, std_acc, mean_macro_f1, std_macro_f1).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Standardise (per-layer scale differs); liblinear-free lbfgs handles
    # multinomial multi-class. max_iter generous since features are 768-d.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = ["accuracy", "f1_macro"]
    res = cross_validate(clf, X, y, cv=skf, scoring=scoring, n_jobs=1)
    return (
        float(res["test_accuracy"].mean()),
        float(res["test_accuracy"].std()),
        float(res["test_f1_macro"].mean()),
        float(res["test_f1_macro"].std()),
    )


def main():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Per-layer linear probe over MoPE features (D-13 layer choice)."
    )
    parser.add_argument(
        "--feat-dir", required=True,
        help="Directory with features.npy / labels.npy / meta.json "
             "(output of mope_layer_features.py).",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="StratifiedKFold splits (default 5). Auto-reduced if a class is rare.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for CV shuffling.",
    )
    parser.add_argument(
        "--out-csv", default=None,
        help="Optional CSV path to save the per-layer metrics table.",
    )
    parser.add_argument(
        "--out-png", default=None,
        help="Optional PNG path for the acc-vs-layer curve (needs matplotlib).",
    )
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    features = np.load(feat_dir / "features.npy")  # [N, n_layers, 768]
    labels = np.load(feat_dir / "labels.npy")      # [N]
    meta_path = feat_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}

    if features.ndim != 3:
        raise ValueError(
            f"Expected features [N, n_layers, 768], got shape {features.shape}"
        )
    N, n_layers, embed_dim = features.shape
    logger.info("Loaded features %s, labels %s.", features.shape, labels.shape)

    label_names = meta.get("label_names")
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    logger.info(
        "N=%d  n_layers=%d  embed_dim=%d  n_classes=%d  class_counts=%s",
        N, n_layers, embed_dim, n_classes, dict(zip(classes.tolist(), counts.tolist())),
    )
    if n_classes < N_CLASS_MIN:
        raise RuntimeError(
            f"Only {n_classes} class(es) present — need >= {N_CLASS_MIN} for a probe. "
            "Check --label-key when extracting features."
        )

    # Cap folds at the smallest class count so StratifiedKFold is valid.
    min_count = int(counts.min())
    n_splits = max(2, min(args.folds, min_count))
    if n_splits != args.folds:
        logger.warning(
            "Reduced folds %d -> %d (smallest class has %d samples).",
            args.folds, n_splits, min_count,
        )

    # Chance level = majority-class fraction (accuracy baseline).
    chance = float(counts.max() / N)
    logger.info("Majority-class (chance) accuracy = %.4f", chance)

    rows = []  # (layer, acc, acc_std, f1, f1_std)
    for li in range(n_layers):
        X = features[:, li, :]
        acc, acc_std, f1, f1_std = _fit_layer_probe(X, labels, n_splits, args.seed)
        rows.append((li, acc, acc_std, f1, f1_std))
        logger.info(
            "layer %2d | acc=%.4f (±%.4f) | macroF1=%.4f (±%.4f)",
            li, acc, acc_std, f1, f1_std,
        )

    accs = np.array([r[1] for r in rows])
    peak_layer = int(accs.argmax())
    last_layer = n_layers - 1
    peak_acc = float(accs[peak_layer])
    last_acc = float(accs[last_layer])
    gap = peak_acc - last_acc

    # ---- Pretty table to stdout ----
    print("\n=== Per-layer linear probe (MoPE) — D-13 evidence ===")
    print(f"{'layer':>5} | {'acc':>7} | {'acc_std':>7} | {'macroF1':>7} | {'f1_std':>7}")
    print("-" * 48)
    for li, acc, acc_std, f1, f1_std in rows:
        marker = ""
        if li == peak_layer:
            marker = "  <- PEAK"
        elif li == last_layer:
            marker = "  <- LAST (current MoPE output)"
        print(f"{li:>5} | {acc:>7.4f} | {acc_std:>7.4f} | {f1:>7.4f} | {f1_std:>7.4f}{marker}")

    print("\n--- Summary (chance = %.4f) ---" % chance)
    print(f"Peak layer       : {peak_layer}  (acc={peak_acc:.4f})")
    print(f"Last layer       : {last_layer}  (acc={last_acc:.4f})  <- current MoPE抽这层")
    print(f"Peak - Last gap  : {gap:+.4f} ({gap * 100:+.2f} pp)")
    moe = meta.get("moe_layer_indices")
    if moe:
        print(f"MoE layers       : {moe}  (top-1/3)")
    if peak_layer < last_layer and gap > 0:
        print(
            "Signal           : 峰值在更早的层且优于末层 → 与 FAIR '中层峰值、"
            "向输出退化' 一致；改抽中层 (D-13) 有量化收益。"
        )
    elif peak_layer == last_layer:
        print(
            "Signal           : 末层即峰值 → 当前 MoPE 取末层已最优；改中层无收益 (D-13)。"
        )
    else:
        print(
            "Signal           : 峰值与末层接近/不显著 → 改层收益小，按其他因素定 (D-13)。"
        )
    if label_names:
        print(f"Labels           : {label_names}")

    # ---- Optional CSV ----
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w") as fh:
            fh.write("layer,acc,acc_std,macro_f1,macro_f1_std\n")
            for li, acc, acc_std, f1, f1_std in rows:
                fh.write(f"{li},{acc:.6f},{acc_std:.6f},{f1:.6f},{f1_std:.6f}\n")
        logger.info("Wrote per-layer table to %s", out_csv)

    # ---- Optional PNG ----
    if args.out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [r[0] for r in rows]
            plt.figure(figsize=(7, 4))
            plt.plot(xs, accs, marker="o", label="probe acc")
            plt.axhline(chance, color="gray", ls="--", label=f"chance={chance:.3f}")
            plt.axvline(peak_layer, color="green", ls=":", label=f"peak L{peak_layer}")
            plt.axvline(last_layer, color="red", ls=":", label=f"last L{last_layer}")
            plt.xlabel("transformer block index")
            plt.ylabel("CV accuracy")
            plt.title("MoPE per-layer linear probe (D-13)")
            plt.legend()
            plt.tight_layout()
            out_png = Path(args.out_png)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=120)
            logger.info("Wrote curve to %s", out_png)
        except ImportError:
            logger.warning("matplotlib not available — skipping --out-png.")


if __name__ == "__main__":
    main()
