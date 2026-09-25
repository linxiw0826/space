#!/usr/bin/env python3
"""Find VSI-Bench cases where model A got it right and model B got it wrong.

Joins two lmms-eval samples.jsonl files on doc_id, decides correctness by
comparing `prediction` against `ground_truth` (numeric-aware), and prints the
video path + question + both predictions for cases where A is correct and B
is not.

Usage:
    python find_vsibench_diff_cases.py \
        --correct /data2/wlx/output/eval/vsibench/e05a_mope73_full_llm_4b/e05a_mope73_full_llm_4b_samples.jsonl \
        --wrong /data2/wlx/output/eval/vsibench/baseline_quarter_4b/train__baseline_quarter_4b/20260914_204602_samples_vsibench.jsonl \
        --num-cases 10
"""
import argparse
import json
import random
from pathlib import Path

VIDEO_ROOT = "/data2/wlx/data/VSIBench"


def load_by_doc_id(path: Path) -> dict:
    rows = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["doc_id"]] = row["vsibench_score"]
    return rows


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_correct(pred: str, ground_truth: str) -> bool:
    if pred is None or ground_truth is None:
        return False
    if str(pred).strip().lower() == str(ground_truth).strip().lower():
        return True
    pf, gf = to_float(pred), to_float(ground_truth)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct", required=True, help="samples.jsonl for the model that should be right")
    parser.add_argument("--wrong", required=True, help="samples.jsonl for the model that should be wrong")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-root", default=VIDEO_ROOT)
    args = parser.parse_args()

    correct_rows = load_by_doc_id(Path(args.correct))
    wrong_rows = load_by_doc_id(Path(args.wrong))

    common_ids = sorted(set(correct_rows) & set(wrong_rows))
    if not common_ids:
        raise RuntimeError("No overlapping doc_id between the two samples files — are they the same eval set?")

    cases = []
    for doc_id in common_ids:
        c = correct_rows[doc_id]
        w = wrong_rows[doc_id]
        gt = c["ground_truth"]
        if is_correct(c["prediction"], gt) and not is_correct(w["prediction"], gt):
            video_path = f"{args.video_root}/{c['dataset']}/{c['scene_name']}.mp4"
            cases.append(
                {
                    "doc_id": doc_id,
                    "video_path": video_path,
                    "question_type": c["question_type"],
                    "question": c["question"],
                    "ground_truth": gt,
                    "correct_model_prediction": c["prediction"],
                    "wrong_model_prediction": w["prediction"],
                }
            )

    print(f"Total overlapping questions: {len(common_ids)}")
    print(f"Cases where --correct is right and --wrong is wrong: {len(cases)}\n")

    random.Random(args.seed).shuffle(cases)
    selected = cases[: args.num_cases]

    for i, case in enumerate(selected, 1):
        print(f"--- case {i} (doc_id={case['doc_id']}, type={case['question_type']}) ---")
        print(f"video: {case['video_path']}")
        print(f"question: {case['question']}")
        print(f"ground_truth: {case['ground_truth']}")
        print(f"correct model prediction: {case['correct_model_prediction']}")
        print(f"wrong model prediction: {case['wrong_model_prediction']}")
        print()


if __name__ == "__main__":
    main()
