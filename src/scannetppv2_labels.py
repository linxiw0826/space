"""Frozen ScanNet++ V2 official-label to VSI category normalization."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "scannetppv2_vsi_label_normalization_v1"

# These aliases are the complete set of non-identity official->VSI mappings
# observed across the 855-scene VSI-590K ScanNet++ V2 subset.  Unknown future
# differences remain fail-closed in the full-contract audit.
LABEL_ALIASES = {
    "bench cushion": "cushion",
    "beverage crate": "crate",
    "beverage crates": "crate",
    "ceiling lamp": "ceiling light",
    "couch": "sofa",
    "desk": "table",
    "fridge": "refrigerator",
    "mouse": "computer mouse",
    "mug": "cup",
    "office chair": "chair",
    "office desk": "table",
    "office table": "table",
    "picutre": "picture",
    "shoe": "shoes",
    "slipper": "slippers",
    "trash bin": "trash can",
    "trashbin": "trash can",
}


def normalize_scannetppv2_label(value: Any) -> str:
    label = " ".join(str(value).strip().lower().split())
    return LABEL_ALIASES.get(label, label)
