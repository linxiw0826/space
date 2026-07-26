#!/usr/bin/env python3
"""Audit Part A Core coverage under coordinate and instance-safety gates."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate-confidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accepted-coordinate-levels",
        nargs="+",
        default=["high"],
        choices=["high", "medium", "low"],
    )
    return parser.parse_args()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarize(items, confidence, accepted_levels):
    totals = Counter()
    accepted = Counter()
    accepted_question_types = Counter()
    accepted_categories = Counter()
    accepted_singleton_categories = Counter()
    coordinate_levels = Counter()
    scene_rows = []

    for item in items:
        scene_id = item["scene_id"]
        nodes = item.get("nodes", [])
        qa = item.get("qa", [])
        views = item.get("candidate_views", [])

        category_counts = Counter(node["category"] for node in nodes)
        singleton_node_ids = {
            node["node_id"]
            for node in nodes
            if category_counts[node["category"]] == 1
        }
        all_node_ids = {node["node_id"] for node in nodes}

        visible_observations = sum(
            len(view.get("visible_nodes", [])) for view in views
        )
        singleton_visible_observations = sum(
            sum(
                visible["node_id"] in singleton_node_ids
                for visible in view.get("visible_nodes", [])
            )
            for view in views
        )

        totals["scenes"] += 1
        totals["qa_rows"] += len(qa)
        totals["nodes"] += len(nodes)
        totals["visible_observations"] += visible_observations
        totals["possible_node_pairs"] += len(nodes) * (len(nodes) - 1) // 2

        confidence_record = confidence.get(scene_id, {})
        level = confidence_record.get("coordinate_confidence", "missing")
        coordinate_levels[level] += 1
        scene_accepted = level in accepted_levels

        if scene_accepted:
            singleton_count = len(singleton_node_ids)
            accepted["scenes"] += 1
            accepted["qa_rows"] += len(qa)
            accepted["all_nodes"] += len(all_node_ids)
            accepted["singleton_nodes"] += singleton_count
            accepted["all_visible_observations"] += visible_observations
            accepted[
                "singleton_visible_observations"
            ] += singleton_visible_observations
            accepted["all_possible_node_pairs"] += (
                len(nodes) * (len(nodes) - 1) // 2
            )
            accepted["singleton_possible_node_pairs"] += (
                singleton_count * (singleton_count - 1) // 2
            )
            if singleton_count >= 1:
                accepted["scenes_with_singleton_node"] += 1
            if singleton_count >= 2:
                accepted["scenes_with_singleton_graph"] += 1

            accepted_question_types.update(
                row.get("question_type", "unknown") for row in qa
            )
            accepted_categories.update(node["category"] for node in nodes)
            accepted_singleton_categories.update(
                node["category"]
                for node in nodes
                if node["node_id"] in singleton_node_ids
            )

        scene_rows.append({
            "scene_id": scene_id,
            "coordinate_confidence": level,
            "offset_difference_m": confidence_record.get(
                "offset_difference_m"
            ),
            "accepted_coordinate": scene_accepted,
            "qa_rows": len(qa),
            "nodes": len(nodes),
            "singleton_nodes": len(singleton_node_ids),
            "ambiguous_multicategory_nodes": (
                len(nodes) - len(singleton_node_ids)
            ),
            "singleton_possible_node_pairs": (
                len(singleton_node_ids)
                * (len(singleton_node_ids) - 1)
                // 2
            ),
            "visible_observations": visible_observations,
            "singleton_visible_observations": (
                singleton_visible_observations
            ),
        })

    coverage = {
        "accepted_scene_rate": safe_ratio(
            accepted["scenes"], totals["scenes"]
        ),
        "accepted_qa_rate": safe_ratio(
            accepted["qa_rows"], totals["qa_rows"]
        ),
        "singleton_node_rate_within_accepted": safe_ratio(
            accepted["singleton_nodes"], accepted["all_nodes"]
        ),
        "singleton_observation_rate_within_accepted": safe_ratio(
            accepted["singleton_visible_observations"],
            accepted["all_visible_observations"],
        ),
        "singleton_pair_rate_within_accepted": safe_ratio(
            accepted["singleton_possible_node_pairs"],
            accepted["all_possible_node_pairs"],
        ),
    }

    return {
        "schema_version": "scannet_parta_core_coverage_v1",
        "policy": {
            "accepted_coordinate_levels": sorted(accepted_levels),
            "instance_safe_definition": (
                "A node is instance-safe only when its category occurs once "
                "among the manifest nodes for that scene. Multi-instance "
                "categories are retained for QA/category visibility but are "
                "not counted as verified instance geometry."
            ),
        },
        "coordinate_level_counts": dict(coordinate_levels),
        "totals": dict(totals),
        "accepted": dict(accepted),
        "coverage": coverage,
        "accepted_question_types": dict(
            accepted_question_types.most_common()
        ),
        "accepted_all_node_categories": dict(
            accepted_categories.most_common()
        ),
        "accepted_singleton_node_categories": dict(
            accepted_singleton_categories.most_common()
        ),
        "scene_rows": scene_rows,
    }


def main():
    args = parse_args()
    items = load_jsonl(args.manifest)
    with args.coordinate_confidence.open("r", encoding="utf-8") as handle:
        confidence = json.load(handle)

    report = summarize(
        items,
        confidence,
        set(args.accepted_coordinate_levels),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    printable = {
        key: value
        for key, value in report.items()
        if key != "scene_rows"
    }
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
