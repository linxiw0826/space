#!/usr/bin/env python3
"""Per-question oracle routing headroom for VSI-Bench (Paper-2 Validation ①).

Purpose
-------
Estimate the *upper bound* of a perfect per-question router between two models:
  * E-01   : Qwen3-VL SFT, no MoPE        (wrapper ``qwen3_vl_my``)
  * E-03a  : Qwen3-VL + MoPE cross-attn   (wrapper ``qwen3_vl_mope_crossattn``)

For every question we take ``max(score_E01, score_E03a)`` (a router that always
picks the better model on that exact question). Aggregating these gives the
*per-question oracle*, which is a tighter upper bound than the already-known
*sub-task-level oracle = 70.33%* (which can only pick a whole model per subtask).
Hence the per-question oracle must be >= 70.33%.

Inputs
------
Two ``{date}_samples_vsibench.jsonl`` files, one per model. These are produced
automatically by lmms-eval's ``--log_samples`` flag when re-running:
  * scripts/eval/eval_e01_vsibench.sh   -> E-01 samples jsonl
  * scripts/eval/eval_e03a_vsibench.sh  -> E-03a samples jsonl
They land in ``${SPACE_OUTPUT_ROOT}/eval/vsibench/<EXP>/<model_sanitized>/``.

Each sample line is a JSON object with at least::

    {
      "doc_id": <int>,                 # dataset row index == alignment key
      "target": <str>,                 # ground truth (== ground_truth)
      "filtered_resps": [...],         # model prediction text
      "vsibench_score": {              # = the whole doc dict + prediction + score
          "question_type": <str>,      # e.g. object_rel_distance / object_counting
          "ground_truth": <str>,
          "prediction": <str>,
          "accuracy": 0.0|1.0,         # present for MCA question types
          "MRA:.5:.95:.05": <float>,   # present for NA question types
          ...
      }
    }

Per-question score key & question_type values are taken verbatim from
``lmms_eval/tasks/vsibench/utils.py`` (vsibench_process_results / the
METRICS_FOR_MCA / METRICS_FOR_NA / *_QUESTION_TYPES constants). See the
constants below.

Alignment
---------
Two evals use the SAME test.jsonl and DO NOT shuffle (``process_docs`` only
shuffles when env ``LMMS_EVAL_SHUFFLE_DOCS`` is set, which neither eval script
sets), so ``doc_id`` denotes the same question across both files. Multi-process
writes may reorder physical lines, so we ALWAYS join on ``doc_id``, never on
line order.

Overall metric
--------------
Mirrors ``vsibench_aggregate_results`` in utils.py exactly:
  1. Per subtask: mean of its per-question scores
     (accuracy for MCA, MRA for NA).
  2. The three ``object_rel_direction_{easy,medium,hard}`` subtasks are first
     collapsed into ONE ``object_rel_direction`` subtask = mean of the three
     subtask means.
  3. ``overall`` = simple (unweighted) mean over the resulting 8 subtask values.
This is a mean-of-subtask-means, NOT a per-question mean.

Usage
-----
    python -m src.analysis.oracle_route_headroom \
        --e01  /path/to/<date>_samples_vsibench.jsonl \
        --e03a /path/to/<date>_samples_vsibench.jsonl \
        [--out /path/to/report.md]   # or .json (extension decides format)

Pure Python + stdlib only (numpy NOT required). CPU-only, no GPU, no lmms-eval
import. Runnable on a login node once both jsonl files are present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("oracle_route_headroom")

# ---------------------------------------------------------------------------
# Constants copied VERBATIM from lmms_eval/tasks/vsibench/utils.py
# (MCA_QUESTION_TYPES, NA_QUESTION_TYPES, METRICS_FOR_MCA, METRICS_FOR_NA).
# Keep in sync if the vendor task ever changes these.
# ---------------------------------------------------------------------------
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

# Per-question score key for each family (the dict keys vsibench writes into
# the doc / vsibench_score dict).
MCA_SCORE_KEY = "accuracy"            # 0.0 / 1.0
NA_SCORE_KEY = "MRA:.5:.95:.05"      # continuous in [0, 1]

# The three rel_direction difficulty buckets get merged into this one subtask
# (matches the output.pop(...) merge in vsibench_aggregate_results).
REL_DIRECTION_BUCKETS = (
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
)
REL_DIRECTION_MERGED = "object_rel_direction"

# Known coarse-grained (sub-task-level) oracle, for sanity comparison.
SUBTASK_LEVEL_ORACLE_PCT = 70.33


class ParseError(Exception):
    """Raised for unrecoverable input problems."""


# ---------------------------------------------------------------------------
# Loading & scoring
# ---------------------------------------------------------------------------
def _question_family(question_type: str) -> str:
    """Return 'MCA' or 'NA' for a question_type, else raise."""
    if question_type in MCA_QUESTION_TYPES:
        return "MCA"
    if question_type in NA_QUESTION_TYPES:
        return "NA"
    raise ParseError(f"Unknown question_type: {question_type!r}")


def _extract_score(score_dict: dict, question_type: str, doc_id, src: str) -> float:
    """Pull the per-question score (accuracy for MCA, MRA for NA)."""
    family = _question_family(question_type)
    key = MCA_SCORE_KEY if family == "MCA" else NA_SCORE_KEY
    if key not in score_dict:
        raise ParseError(
            f"[{src}] doc_id={doc_id} question_type={question_type!r} (family "
            f"{family}) missing expected score key {key!r}. "
            f"Available keys: {sorted(score_dict.keys())}"
        )
    val = score_dict[key]
    try:
        return float(val)
    except (TypeError, ValueError) as e:
        raise ParseError(
            f"[{src}] doc_id={doc_id} score key {key!r} is not numeric: {val!r}"
        ) from e


def load_samples(path: str, src: str) -> Dict[int, dict]:
    """Parse a *_samples_vsibench.jsonl into {doc_id: record}.

    record = {"question_type": str, "score": float}

    Corrupt lines are warned-and-skipped; structural problems on otherwise
    valid lines raise ParseError (loud, not silent).
    """
    import os

    if not os.path.isfile(path):
        raise ParseError(f"[{src}] input file does not exist: {path}")

    out: Dict[int, dict] = {}
    n_lines = 0
    n_bad = 0
    n_dup = 0
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            n_lines += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                n_bad += 1
                logger.warning("[%s] line %d: corrupt JSON, skipping (%s)", src, lineno, e)
                continue

            if "doc_id" not in obj:
                n_bad += 1
                logger.warning("[%s] line %d: missing 'doc_id', skipping", src, lineno)
                continue
            doc_id = obj["doc_id"]

            score_dict = obj.get("vsibench_score")
            if not isinstance(score_dict, dict):
                raise ParseError(
                    f"[{src}] line {lineno} doc_id={doc_id}: missing/invalid "
                    f"'vsibench_score' dict (got {type(score_dict).__name__})"
                )
            qtype = score_dict.get("question_type")
            if qtype is None:
                raise ParseError(
                    f"[{src}] line {lineno} doc_id={doc_id}: vsibench_score has "
                    f"no 'question_type'"
                )

            score = _extract_score(score_dict, qtype, doc_id, src)

            if doc_id in out:
                n_dup += 1
                logger.warning(
                    "[%s] duplicate doc_id=%s (line %d); keeping first occurrence",
                    src, doc_id, lineno,
                )
                continue
            out[doc_id] = {"question_type": qtype, "score": score}

    if not out:
        raise ParseError(f"[{src}] no usable samples parsed from {path}")
    logger.info(
        "[%s] parsed %d samples from %d non-empty lines (%d corrupt, %d dup) -> %s",
        src, len(out), n_lines, n_bad, n_dup, path,
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation (mirrors vsibench_aggregate_results)
# ---------------------------------------------------------------------------
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_overall(per_subtask_mean: Dict[str, float]) -> float:
    """Collapse rel_direction buckets, then simple-mean over 8 subtasks.

    `per_subtask_mean` maps raw question_type -> mean score (the 3 rel_direction
    difficulty buckets present as separate keys, like utils.py before the pop).
    """
    merged = dict(per_subtask_mean)
    buckets = [merged.pop(b) for b in REL_DIRECTION_BUCKETS if b in merged]
    if buckets:
        merged[REL_DIRECTION_MERGED] = _mean(buckets)
    return _mean(list(merged.values()))


def compute(
    e01: Dict[int, dict],
    e03a: Dict[int, dict],
) -> dict:
    """Compute per question_type means for E-01, E-03a, and per-question oracle.

    Returns a dict with:
      - 'subtask_rows': OrderedDict raw_qtype -> {e01, e03a, oracle} (means over
        the shared doc_ids, raw difficulty buckets still separate)
      - 'display_rows': OrderedDict merged_qtype -> {e01, e03a, oracle} (rel_dir
        buckets collapsed; this is what the markdown table shows)
      - 'overall': {e01, e03a, oracle}
      - 'n_common', 'n_missing_e01', 'n_missing_e03a'
    """
    ids_e01 = set(e01)
    ids_e03a = set(e03a)
    common = ids_e01 & ids_e03a
    miss_in_e01 = ids_e03a - ids_e01   # present in e03a, absent in e01
    miss_in_e03a = ids_e01 - ids_e03a

    if ids_e01 != ids_e03a:
        logger.warning(
            "doc_id sets differ: |E01|=%d |E03a|=%d |common|=%d "
            "(missing from E01: %d, missing from E03a: %d). Using INTERSECTION.",
            len(ids_e01), len(ids_e03a), len(common),
            len(miss_in_e01), len(miss_in_e03a),
        )
    if not common:
        raise ParseError("no common doc_id between the two files; cannot align.")

    # Accumulate per-question_type scores over the common set.
    # Use the question_type from E-01 (must match E-03a; warn on mismatch).
    acc: "OrderedDict[str, dict]" = OrderedDict()
    n_qtype_mismatch = 0
    for doc_id in sorted(common):
        r1 = e01[doc_id]
        r3 = e03a[doc_id]
        qt = r1["question_type"]
        if r3["question_type"] != qt:
            n_qtype_mismatch += 1
            logger.warning(
                "doc_id=%s question_type mismatch: E01=%r E03a=%r; using E01's.",
                doc_id, qt, r3["question_type"],
            )
        s1 = r1["score"]
        s3 = r3["score"]
        bucket = acc.setdefault(qt, {"e01": [], "e03a": [], "oracle": []})
        bucket["e01"].append(s1)
        bucket["e03a"].append(s3)
        bucket["oracle"].append(max(s1, s3))

    if n_qtype_mismatch:
        logger.warning("%d doc_ids had question_type mismatch between files", n_qtype_mismatch)

    # Order subtask rows: MCA types first (in canonical order), then NA.
    canonical = MCA_QUESTION_TYPES + NA_QUESTION_TYPES
    ordered_qtypes = [q for q in canonical if q in acc] + [q for q in acc if q not in canonical]

    subtask_rows: "OrderedDict[str, dict]" = OrderedDict()
    e01_means: Dict[str, float] = {}
    e03a_means: Dict[str, float] = {}
    oracle_means: Dict[str, float] = {}
    for qt in ordered_qtypes:
        b = acc[qt]
        m01, m03, mor = _mean(b["e01"]), _mean(b["e03a"]), _mean(b["oracle"])
        subtask_rows[qt] = {"e01": m01, "e03a": m03, "oracle": mor, "n": len(b["e01"])}
        e01_means[qt] = m01
        e03a_means[qt] = m03
        oracle_means[qt] = mor

    overall = {
        "e01": aggregate_overall(e01_means),
        "e03a": aggregate_overall(e03a_means),
        "oracle": aggregate_overall(oracle_means),
    }

    # Display rows: collapse rel_direction buckets into one row.
    display_rows: "OrderedDict[str, dict]" = OrderedDict()
    rel_e01, rel_e03a, rel_or, rel_n = [], [], [], 0
    for qt, row in subtask_rows.items():
        if qt in REL_DIRECTION_BUCKETS:
            rel_e01.append(row["e01"])
            rel_e03a.append(row["e03a"])
            rel_or.append(row["oracle"])
            rel_n += row["n"]
            continue
        display_rows[qt] = row
    if rel_e01:
        merged_row = {
            "e01": _mean(rel_e01),
            "e03a": _mean(rel_e03a),
            "oracle": _mean(rel_or),
            "n": rel_n,
        }
        # Insert merged rel_direction in canonical position (before object_rel_distance).
        new_display: "OrderedDict[str, dict]" = OrderedDict()
        inserted = False
        for qt, row in display_rows.items():
            if qt == "object_rel_distance" and not inserted:
                new_display[REL_DIRECTION_MERGED] = merged_row
                inserted = True
            new_display[qt] = row
        if not inserted:
            new_display[REL_DIRECTION_MERGED] = merged_row
        display_rows = new_display

    return {
        "subtask_rows": subtask_rows,
        "display_rows": display_rows,
        "overall": overall,
        "n_common": len(common),
        "n_missing_e01": len(miss_in_e01),
        "n_missing_e03a": len(miss_in_e03a),
        "n_qtype_mismatch": n_qtype_mismatch,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(x: float) -> float:
    return 100.0 * x


def build_markdown(result: dict) -> str:
    rows = result["display_rows"]
    overall = result["overall"]
    lines: List[str] = []
    lines.append("# VSI-Bench per-question oracle routing headroom (Validation ①)")
    lines.append("")
    lines.append(
        f"- Aligned questions (common doc_id): **{result['n_common']}**"
    )
    if result["n_missing_e01"] or result["n_missing_e03a"]:
        lines.append(
            f"- WARNING: doc_id mismatch — missing from E-01: "
            f"{result['n_missing_e01']}, missing from E-03a: "
            f"{result['n_missing_e03a']} (used intersection only)"
        )
    if result.get("n_qtype_mismatch"):
        lines.append(
            f"- WARNING: {result['n_qtype_mismatch']} question_type mismatches "
            f"between files (used E-01's type)"
        )
    lines.append(
        "- Scores: MCA -> accuracy (0/1); NA -> MRA:.5:.95:.05. "
        "Oracle = per-question max(E-01, E-03a)."
    )
    lines.append("")
    lines.append(
        "| question_type | E-01 | E-03a | oracle(per-question) | oracle-E03a (headroom) |"
    )
    lines.append("|---|---|---|---|---|")
    for qt, row in rows.items():
        head = row["oracle"] - row["e03a"]
        lines.append(
            f"| {qt} | {_pct(row['e01']):.2f} | {_pct(row['e03a']):.2f} | "
            f"{_pct(row['oracle']):.2f} | {_pct(head):+.2f} |"
        )
    head_overall = overall["oracle"] - overall["e03a"]
    lines.append(
        f"| **overall** | **{_pct(overall['e01']):.2f}** | "
        f"**{_pct(overall['e03a']):.2f}** | **{_pct(overall['oracle']):.2f}** | "
        f"**{_pct(head_overall):+.2f}** |"
    )
    lines.append("")
    lines.append("## Overall summary")
    lines.append(f"- E-01 overall:                   {_pct(overall['e01']):.2f}%")
    lines.append(f"- E-03a overall:                  {_pct(overall['e03a']):.2f}%")
    lines.append(f"- per-question oracle overall:    {_pct(overall['oracle']):.2f}%")
    lines.append(
        f"- headroom vs E-03a:              {_pct(overall['oracle'] - overall['e03a']):+.2f} pp"
    )
    lines.append(
        f"- headroom vs E-01:               {_pct(overall['oracle'] - overall['e01']):+.2f} pp"
    )
    lines.append(
        f"- sub-task-level oracle (known):  {SUBTASK_LEVEL_ORACLE_PCT:.2f}%  "
        f"(per-question oracle should be >= this)"
    )
    sanity = "OK" if _pct(overall["oracle"]) + 1e-6 >= SUBTASK_LEVEL_ORACLE_PCT else "VIOLATION"
    lines.append(
        f"- sanity (per-question >= sub-task oracle): {sanity}"
    )
    lines.append("")
    return "\n".join(lines)


def build_json(result: dict) -> str:
    def _row(row):
        return {
            "e01": row["e01"],
            "e03a": row["e03a"],
            "oracle": row["oracle"],
            "oracle_minus_e03a": row["oracle"] - row["e03a"],
            "n": row.get("n"),
        }

    payload = {
        "n_common": result["n_common"],
        "n_missing_e01": result["n_missing_e01"],
        "n_missing_e03a": result["n_missing_e03a"],
        "n_qtype_mismatch": result.get("n_qtype_mismatch", 0),
        "subtask_level_oracle_pct": SUBTASK_LEVEL_ORACLE_PCT,
        "per_question_type": {qt: _row(r) for qt, r in result["display_rows"].items()},
        "overall": {
            "e01": result["overall"]["e01"],
            "e03a": result["overall"]["e03a"],
            "oracle": result["overall"]["oracle"],
            "oracle_minus_e03a": result["overall"]["oracle"] - result["overall"]["e03a"],
            "oracle_minus_e01": result["overall"]["oracle"] - result["overall"]["e01"],
        },
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-question oracle routing headroom for VSI-Bench (E-01 vs E-03a).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--e01", required=True,
        help="Path to E-01 (qwen3_vl_my) {date}_samples_vsibench.jsonl",
    )
    p.add_argument(
        "--e03a", required=True,
        help="Path to E-03a (qwen3_vl_mope_crossattn) {date}_samples_vsibench.jsonl",
    )
    p.add_argument(
        "--out", default=None,
        help="Optional output path. Extension decides format: .json -> JSON, "
             "anything else (e.g. .md) -> markdown. If omitted, only stdout.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        e01 = load_samples(args.e01, "E-01")
        e03a = load_samples(args.e03a, "E-03a")
        result = compute(e01, e03a)
    except ParseError as e:
        logger.error("FATAL: %s", e)
        return 2

    md = build_markdown(result)
    # Always print the human-readable table to stdout.
    print(md)

    if args.out:
        try:
            if args.out.lower().endswith(".json"):
                content = build_json(result)
            else:
                content = md
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(content)
            logger.info("Wrote report to %s", args.out)
        except OSError as e:
            logger.error("Failed to write --out %s: %s", args.out, e)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
