import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/preprocess/match_scannet_instances_to_bboxes.py"
)
SPEC = importlib.util.spec_from_file_location("instance_matcher", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Args:
    min_score = 0.35
    min_margin = 0.08
    high_score = 0.60
    high_margin = 0.15


def make_features(trajectory):
    return [
        {"visibility": float(value > 0), "area": float(value)}
        for value in trajectory
    ]


def test_hungarian_recovers_crossed_temporal_signatures():
    first = np.asarray([9, 8, 4, 0, 0, 0], dtype=float)
    second = np.asarray([0, 0, 0, 4, 8, 9], dtype=float)
    instances = {11: first * 100, 22: second * 100}
    box_features = [make_features(second), make_features(first)]

    assignments, matrix, instance_ids = MODULE.match_category(
        "chair",
        instances,
        [{}, {}],
        box_features,
        Args(),
    )

    mapping = {
        item["instance_id"]: item["bbox_index"] for item in assignments
    }
    assert instance_ids == [11, 22]
    assert np.asarray(matrix).shape == (2, 2)
    assert mapping == {11: 1, 22: 0}
    assert all(
        item["mapping_confidence"] == "high" for item in assignments
    )


def test_identical_signatures_are_marked_ambiguous():
    trajectory = np.asarray([1, 1, 0, 0], dtype=float)
    instances = {1: trajectory * 100, 2: trajectory * 80}
    box_features = [
        make_features(trajectory),
        make_features(trajectory),
    ]

    assignments, _, _ = MODULE.match_category(
        "table",
        instances,
        [{}, {}],
        box_features,
        Args(),
    )

    assert all(item["match_margin"] == 0 for item in assignments)
    assert all(
        item["mapping_status"] == "ambiguous" for item in assignments
    )
