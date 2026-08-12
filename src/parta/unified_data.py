"""Frozen three-source manifests and engineering-subset contracts for Part A."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from parta_data_contract import ContractError, content_sha256, read_jsonl, stable_json

from .canonical_data import PartACanonicalDataset, PartASample, TargetSelectionAudit, build_state_targets

UNIFIED_SCHEMA_VERSION = "parta_unified_manifest_v2"
ENGINEERING_SUBSET_SCHEMA_VERSION = "parta_engineering_subset_v1"
FROZEN_SOURCE_REGISTRY = ("adt", "hypersim", "scannetppv2")
FROZEN_SOURCE_INVENTORY: Mapping[str, Mapping[str, int]] = {
    "adt": {"qa": 60_207, "scenes": 183},
    "hypersim": {"qa": 176_774, "scenes": 317},
    "scannetppv2": {"qa": 138_701, "scenes": 855},
}
FROZEN_TOTAL_INVENTORY = {"qa": 375_682, "scenes": 1_355}
SPLITS = ("train", "val")
EXACT_QA_FILENAME = "qa_manifest_exact_verified.jsonl"
REQUIRED_CANONICAL_FILENAMES = (
    "qa_manifest_exact_verified.jsonl",
    "scene_states.jsonl",
    "frame_states.jsonl",
)
SUPPORT_CERTIFICATE_FILENAMES: Mapping[str, str] = {
    "adt": "adt_support_certificates.jsonl",
    "hypersim": "hypersim_support_certificates.jsonl",
    "scannetppv2": "scannetppv2_support_certificates.jsonl",
}
SUPPORT_CERTIFICATE_REQUIRED = frozenset({"adt", "scannetppv2"})


@dataclass(frozen=True)
class SplitDefaults:
    """D-62 split constants; alternate values are rejected, not overridden."""

    seed: int = 42
    val_fraction: float = 0.10

    def validate(self) -> None:
        if self.seed != 42 or self.val_fraction != 0.10:
            raise ContractError("D-62 freezes seed=42 and val_fraction=0.10")


def _require_frozen_sources(sources: Iterable[str], *, context: str) -> None:
    actual = {str(source) for source in sources}
    expected = set(FROZEN_SOURCE_REGISTRY)
    if actual != expected:
        raise ContractError(
            f"{context} must use exactly {list(FROZEN_SOURCE_REGISTRY)}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def validate_source_inventory(
    qa_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_inventory: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, dict[str, int]]:
    """Validate exact production counts, or an explicitly supplied fixture contract."""
    _require_frozen_sources(qa_by_source, context="source inventory")
    actual = {
        source: {
            "qa": len(qa_by_source[source]),
            "scenes": len({str(row["scene_id"]) for row in qa_by_source[source]}),
        }
        for source in FROZEN_SOURCE_REGISTRY
    }
    if expected_inventory is not None:
        _require_frozen_sources(expected_inventory, context="expected source inventory")
        normalized_expected = {
            source: {
                "qa": int(expected_inventory[source]["qa"]),
                "scenes": int(expected_inventory[source]["scenes"]),
            }
            for source in FROZEN_SOURCE_REGISTRY
        }
        if actual != normalized_expected:
            raise ContractError(
                f"source inventory differs from frozen contract: expected={normalized_expected}, "
                f"actual={actual}"
            )
    return actual


def stable_scene_split(source: str, scene_id: str, defaults: SplitDefaults) -> str:
    """Assign one scene independently of the surrounding source inventory."""
    defaults.validate()
    if source not in FROZEN_SOURCE_REGISTRY:
        raise ContractError(f"source is outside the D-62 registry: {source}")
    digest = hashlib.sha256(
        f"{UNIFIED_SCHEMA_VERSION}\0{defaults.seed}\0{source}\0{scene_id}".encode()
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < defaults.val_fraction:
        return "val"
    return "train"


def exact_input_binding(qa: Mapping[str, Any]) -> dict[str, Any]:
    """Return the question-independent visual binding persisted for audits."""
    frame_indices = qa.get("actual_frame_indices")
    frame_keys = qa.get("actual_frame_keys")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise ContractError(f"QA lacks exact actual_frame_indices: {qa.get('qa_id')}")
    if not isinstance(frame_keys, list) or len(frame_keys) != len(frame_indices):
        raise ContractError(f"QA has inconsistent actual_frame_keys: {qa.get('qa_id')}")
    if not qa.get("vsi_media"):
        raise ContractError(f"QA lacks exact media binding: {qa.get('qa_id')}")
    return {
        "source_dataset": str(qa["source_dataset"]),
        "scene_id": str(qa["scene_id"]),
        "qa_id": str(qa["qa_id"]),
        "vsi_media": qa.get("vsi_media"),
        "media_kind": qa.get("media_kind"),
        "actual_frame_indices": frame_indices,
        "actual_frame_keys": frame_keys,
        "actual_frame_binding_sha256": qa.get("actual_frame_binding_sha256"),
        "support_certificate_sha256": qa.get("support_certificate_sha256"),
    }


def build_unified_rows(
    qa_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    defaults: SplitDefaults,
    *,
    expected_inventory: Mapping[str, Mapping[str, int]] | None = FROZEN_SOURCE_INVENTORY,
) -> list[dict[str, Any]]:
    """Create deterministic QA index rows with fields needed by balanced sampling."""
    defaults.validate()
    _require_frozen_sources(qa_by_source, context="unified QA registry")
    validate_source_inventory(qa_by_source, expected_inventory)
    rows: list[dict[str, Any]] = []
    seen_qa: set[str] = set()
    source_counts = {source: len(source_rows) for source, source_rows in qa_by_source.items()}
    if any(count == 0 for count in source_counts.values()):
        raise ContractError("every requested source must contain at least one QA row")
    for source in sorted(qa_by_source):
        for qa in qa_by_source[source]:
            if qa.get("source_dataset") != source:
                raise ContractError(f"source mismatch for qa_id={qa.get('qa_id')}")
            qa_id = str(qa["qa_id"])
            if qa_id in seen_qa:
                raise ContractError(f"duplicate cross-source qa_id: {qa_id}")
            seen_qa.add(qa_id)
            scene_id = str(qa["scene_id"])
            rows.append(
                {
                    "schema_version": UNIFIED_SCHEMA_VERSION,
                    "source_dataset": source,
                    "scene_id": scene_id,
                    "qa_id": qa_id,
                    "split": stable_scene_split(source, scene_id, defaults),
                    "source_sampling_key": source,
                    "source_qa_count": source_counts[source],
                    "canonical_qa_content_sha256": content_sha256(qa),
                    "exact_input_binding_sha256": content_sha256(exact_input_binding(qa)),
                }
            )
    rows = sorted(rows, key=lambda row: (row["source_dataset"], row["scene_id"], row["qa_id"]))
    split_counts = Counter((row["source_dataset"], row["split"]) for row in rows)
    for row in rows:
        split_count = split_counts[(row["source_dataset"], row["split"])]
        row["source_split_qa_count"] = split_count
        row["source_balanced_weight"] = 1.0 / split_count
    return rows


def summarize_unified_rows(
    rows: Sequence[Mapping[str, Any]],
    defaults: SplitDefaults,
    *,
    expected_inventory: Mapping[str, Mapping[str, int]] | None = FROZEN_SOURCE_INVENTORY,
) -> dict[str, Any]:
    defaults.validate()
    validate_unified_rows(rows)
    scene_splits: dict[tuple[str, str], str] = {}
    qa_counts: Counter[tuple[str, str]] = Counter()
    scene_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        if split not in SPLITS:
            raise ContractError(f"unknown split: {split}")
        key = (str(row["source_dataset"]), str(row["scene_id"]))
        previous = scene_splits.setdefault(key, split)
        if previous != split:
            raise ContractError(f"scene leakage across splits: {key}")
        scene_sets[split].add(key)
        qa_counts[(str(row["source_dataset"]), split)] += 1
    intersections = {
        f"{left}__{right}": len(scene_sets[left] & scene_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(intersections.values()):
        raise ContractError(f"scene leakage detected: {intersections}")
    sources = list(FROZEN_SOURCE_REGISTRY)
    source_summary = {
        source: {
            split: {
                "scenes": len({scene for src, scene in scene_sets[split] if src == source}),
                "qa": qa_counts[(source, split)],
            }
            for split in SPLITS
        }
        for source in sources
    }
    actual_inventory = {
        source: {
            "qa": sum(source_summary[source][split]["qa"] for split in SPLITS),
            "scenes": sum(source_summary[source][split]["scenes"] for split in SPLITS),
        }
        for source in sources
    }
    if expected_inventory is not None:
        normalized_expected = {
            source: dict(expected_inventory[source]) for source in FROZEN_SOURCE_REGISTRY
        }
        if actual_inventory != normalized_expected:
            raise ContractError(
                f"manifest inventory differs from frozen contract: expected={normalized_expected}, "
                f"actual={actual_inventory}"
            )
        if normalized_expected == dict(FROZEN_SOURCE_INVENTORY):
            actual_total = {
                "qa": len(rows),
                "scenes": len(scene_splits),
            }
            if actual_total != FROZEN_TOTAL_INVENTORY:
                raise ContractError(
                    f"total inventory differs from D-62: expected={FROZEN_TOTAL_INVENTORY}, "
                    f"actual={actual_total}"
                )
    production_inventory = actual_inventory == dict(FROZEN_SOURCE_INVENTORY)
    return {
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "contract_status": (
            "frozen_by_D-62" if production_inventory else "fixture_only_explicit_inventory"
        ),
        "source_registry": sources,
        "source_registry_sha256": content_sha256(sources),
        "seed": defaults.seed,
        "val_fraction": defaults.val_fraction,
        "manifest_rows_sha256": content_sha256(list(rows)),
        "total_scenes": len(scene_splits),
        "total_qa": len(rows),
        "frozen_source_inventory": actual_inventory,
        "frozen_total_inventory": {
            "qa": len(rows),
            "scenes": len(scene_splits),
        },
        "scene_intersections": intersections,
        "sources": source_summary,
    }


def validate_unified_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on any v2 registry, identity, split, or weighting drift."""
    if not rows:
        raise ContractError("unified manifest is empty")
    _require_frozen_sources(
        (str(row.get("source_dataset")) for row in rows),
        context="unified manifest source registry",
    )
    seen_qa: set[str] = set()
    scene_splits: dict[tuple[str, str], str] = {}
    split_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row.get("schema_version") != UNIFIED_SCHEMA_VERSION:
            raise ContractError("unified manifest schema mismatch")
        source = str(row.get("source_dataset"))
        split = str(row.get("split"))
        if split not in SPLITS:
            raise ContractError(f"unknown split: {split}")
        qa_id = str(row.get("qa_id"))
        if not qa_id or qa_id == "None" or qa_id in seen_qa:
            raise ContractError(f"duplicate or missing qa_id: {qa_id}")
        seen_qa.add(qa_id)
        key = (source, str(row.get("scene_id")))
        previous = scene_splits.setdefault(key, split)
        if previous != split:
            raise ContractError(f"scene leakage across splits: {key}")
        if row.get("source_sampling_key") != source:
            raise ContractError(f"invalid source_sampling_key for {qa_id}")
        if not isinstance(row.get("canonical_qa_content_sha256"), str):
            raise ContractError(f"missing canonical QA content binding for {qa_id}")
        if not isinstance(row.get("exact_input_binding_sha256"), str):
            raise ContractError(f"missing exact input binding for {qa_id}")
        split_counts[(source, split)] += 1
    source_counts = Counter(str(row["source_dataset"]) for row in rows)
    for row in rows:
        source = str(row["source_dataset"])
        split = str(row["split"])
        expected_split_count = split_counts[(source, split)]
        if row.get("source_qa_count") != source_counts[source]:
            raise ContractError(f"source_qa_count drift for {row['qa_id']}")
        if row.get("source_split_qa_count") != expected_split_count:
            raise ContractError(f"source_split_qa_count drift for {row['qa_id']}")
        if row.get("source_balanced_weight") != 1.0 / expected_split_count:
            raise ContractError(f"source-balanced weight drift for {row['qa_id']}")


def load_unified_rows(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    rows = list(read_jsonl(Path(path)))
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    validate_unified_rows(rows)
    return [row for row in rows if split is None or row.get("split") == split]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_exact_input_registry(
    source_roots: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    _require_frozen_sources(source_roots, context="exact canonical input registry")
    registry: dict[str, dict[str, Any]] = {}
    for source in sorted(source_roots):
        root = Path(source_roots[source]).resolve()
        files: dict[str, dict[str, Any]] = {}
        for filename in REQUIRED_CANONICAL_FILENAMES:
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"missing canonical input for {source}: {path}")
            stat = path.stat()
            files[filename] = {
                "path": str(path),
                "size_bytes": stat.st_size,
                "sha256": file_sha256(path),
                "required": True,
                "present": True,
            }
        support_filename = SUPPORT_CERTIFICATE_FILENAMES[source]
        support_path = root / support_filename
        support_required = source in SUPPORT_CERTIFICATE_REQUIRED
        if support_required and not support_path.is_file():
            raise FileNotFoundError(
                f"missing required support certificate for {source}: {support_path}"
            )
        support_entry: dict[str, Any] = {
            "path": str(support_path),
            "required": support_required,
            "present": support_path.is_file(),
        }
        if support_path.is_file():
            stat = support_path.stat()
            support_entry.update(
                {"size_bytes": stat.st_size, "sha256": file_sha256(support_path)}
            )
        files[support_filename] = support_entry
        registry[source] = {
            "root": str(root),
            "files": files,
            "files_sha256": content_sha256(files),
        }
    return registry


def validate_exact_input_registry(
    source_roots: Mapping[str, str | Path],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_frozen_sources(source_roots, context="canonical source roots")
    _require_frozen_sources(expected, context="persisted exact-input registry")
    if set(source_roots) != set(expected):
        raise ContractError(
            "canonical source roots do not match the persisted exact-input registry"
        )
    actual = build_exact_input_registry(source_roots)
    for source in sorted(expected):
        if stable_json(actual[source]) != stable_json(expected[source]):
            raise ContractError(f"exact canonical input changed for {source}")


def _engineering_scene_rank(source: str, scene_id: str) -> str:
    return hashlib.sha256(
        f"{ENGINEERING_SUBSET_SCHEMA_VERSION}\0{42}\0{source}\0{scene_id}".encode()
    ).hexdigest()


def build_engineering_subset_artifact(
    unified_rows: Sequence[Mapping[str, Any]],
    qa_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    scenes_per_source: int,
    exact_canonical_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic train-only subset without changing train weights."""
    if type(scenes_per_source) is not int or scenes_per_source <= 0:
        raise ContractError("scenes_per_source must be an explicit positive integer")
    validate_unified_rows(unified_rows)
    _require_frozen_sources(qa_by_source, context="engineering QA registry")
    _require_frozen_sources(exact_canonical_inputs, context="engineering exact-input registry")
    qa_lookup: dict[str, Mapping[str, Any]] = {}
    for source in FROZEN_SOURCE_REGISTRY:
        for qa in qa_by_source[source]:
            qa_id = str(qa["qa_id"])
            if qa_id in qa_lookup:
                raise ContractError(f"duplicate canonical QA while building engineering subset: {qa_id}")
            qa_lookup[qa_id] = qa
    train_scenes: dict[str, set[str]] = defaultdict(set)
    for row in unified_rows:
        if row["split"] == "train":
            train_scenes[str(row["source_dataset"])].add(str(row["scene_id"]))
    selected_scene_ids: dict[str, list[str]] = {}
    for source in FROZEN_SOURCE_REGISTRY:
        ranked = sorted(
            train_scenes[source],
            key=lambda scene: (_engineering_scene_rank(source, scene), scene),
        )
        if len(ranked) < scenes_per_source:
            raise ContractError(
                f"engineering subset requests {scenes_per_source} train scenes for {source}, "
                f"but only {len(ranked)} are available"
            )
        selected_scene_ids[source] = ranked[:scenes_per_source]
    selected_sets = {source: set(scenes) for source, scenes in selected_scene_ids.items()}
    selected_rows = [
        row
        for row in unified_rows
        if row["split"] == "train"
        and str(row["scene_id"]) in selected_sets[str(row["source_dataset"])]
    ]
    qa_records = []
    for row in selected_rows:
        qa_id = str(row["qa_id"])
        qa = qa_lookup.get(qa_id)
        if qa is None:
            raise ContractError(f"engineering subset references missing canonical QA: {qa_id}")
        if content_sha256(qa) != row["canonical_qa_content_sha256"]:
            raise ContractError(f"canonical QA content drift for engineering QA: {qa_id}")
        binding = exact_input_binding(qa)
        if content_sha256(binding) != row["exact_input_binding_sha256"]:
            raise ContractError(f"exact input binding drift for engineering QA: {qa_id}")
        qa_records.append(
            {
                "source_dataset": row["source_dataset"],
                "scene_id": row["scene_id"],
                "qa_id": qa_id,
                "canonical_qa_content_sha256": row["canonical_qa_content_sha256"],
                "exact_input": binding,
                "exact_input_binding_sha256": row["exact_input_binding_sha256"],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": ENGINEERING_SUBSET_SCHEMA_VERSION,
        "contract_status": "size_explicitly_frozen_at_build_time",
        "selection": {
            "seed": 42,
            "unit": "source_dataset+scene_id",
            "policy": "stable_rank_within_each_source_train_split",
            "scenes_per_source": scenes_per_source,
            "question_loss_performance_fields_read_for_selection": False,
        },
        "source_registry": list(FROZEN_SOURCE_REGISTRY),
        "unified_manifest_rows_sha256": content_sha256(list(unified_rows)),
        "exact_canonical_inputs": dict(exact_canonical_inputs),
        "selected_scene_ids": selected_scene_ids,
        "selected_qa": qa_records,
        "selected_scene_counts": {
            source: len(selected_scene_ids[source]) for source in FROZEN_SOURCE_REGISTRY
        },
        "selected_qa_counts": dict(Counter(record["source_dataset"] for record in qa_records)),
        "formal_train_reuse": {
            "subset_rows_remain_in_train_manifest": True,
            "source_balanced_weights_unchanged": True,
            "extra_sampling_weight": False,
        },
        "transaction_promotion": {
            "promotable_to_formal_training": False,
            "discard": ["model", "optimizer", "scheduler", "rng", "sampler"],
            "formal_restart_optimizer_step": 0,
        },
    }
    payload["artifact_payload_sha256"] = content_sha256(payload)
    return payload


def validate_engineering_subset_artifact(
    artifact: Mapping[str, Any],
    unified_rows: Sequence[Mapping[str, Any]],
    qa_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exact_canonical_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Rebuild the frozen subset and reject any artifact or input tampering."""
    if artifact.get("schema_version") != ENGINEERING_SUBSET_SCHEMA_VERSION:
        raise ContractError("engineering subset schema mismatch")
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping):
        raise ContractError("engineering subset lacks selection contract")
    scenes_per_source = selection.get("scenes_per_source")
    expected = build_engineering_subset_artifact(
        unified_rows,
        qa_by_source,
        scenes_per_source=scenes_per_source,
        exact_canonical_inputs=exact_canonical_inputs,
    )
    if stable_json(dict(artifact)) != stable_json(expected):
        raise ContractError("engineering subset artifact or bound inputs changed")


def load_engineering_subset_artifact(
    path: str | Path,
    unified_rows: Sequence[Mapping[str, Any]],
    qa_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exact_canonical_inputs: Mapping[str, Mapping[str, Any]],
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Load an artifact only when both its file hash and rebuilt content agree."""
    artifact_path = Path(path)
    if file_sha256(artifact_path) != expected_file_sha256:
        raise ContractError("engineering subset artifact file hash mismatch")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_engineering_subset_artifact(
        artifact,
        unified_rows,
        qa_by_source,
        exact_canonical_inputs=exact_canonical_inputs,
    )
    return artifact


def validate_unified_report(
    report: Mapping[str, Any],
    *,
    expected_inventory: Mapping[str, Mapping[str, int]] | None = FROZEN_SOURCE_INVENTORY,
) -> None:
    if report.get("schema_version") != UNIFIED_SCHEMA_VERSION:
        raise ContractError("unified report schema mismatch")
    expected_status = (
        "frozen_by_D-62"
        if expected_inventory is not None
        and {
            source: dict(expected_inventory[source])
            for source in FROZEN_SOURCE_REGISTRY
        }
        == dict(FROZEN_SOURCE_INVENTORY)
        else "fixture_only_explicit_inventory"
    )
    if report.get("contract_status") != expected_status:
        raise ContractError("unified report production/fixture contract status mismatch")
    if report.get("source_registry") != list(FROZEN_SOURCE_REGISTRY):
        raise ContractError("unified report source registry mismatch")
    if report.get("source_registry_sha256") != content_sha256(list(FROZEN_SOURCE_REGISTRY)):
        raise ContractError("unified report source registry hash mismatch")
    if report.get("seed") != 42 or report.get("val_fraction") != 0.10:
        raise ContractError("unified report split contract mismatch")
    if "smoke_fraction" in report:
        raise ContractError("v2 unified report must not contain a smoke contract")
    if expected_inventory is not None:
        frozen_inventory = {
            source: dict(expected_inventory[source]) for source in FROZEN_SOURCE_REGISTRY
        }
        if report.get("frozen_source_inventory") != frozen_inventory:
            raise ContractError("unified report frozen source inventory mismatch")
        expected_total = {
            "qa": sum(item["qa"] for item in frozen_inventory.values()),
            "scenes": sum(item["scenes"] for item in frozen_inventory.values()),
        }
        if report.get("frozen_total_inventory") != expected_total:
            raise ContractError("unified report frozen total inventory mismatch")
        if frozen_inventory == dict(FROZEN_SOURCE_INVENTORY):
            if expected_total != FROZEN_TOTAL_INVENTORY:
                raise ContractError("D-62 frozen total inventory constant mismatch")
    registry = report.get("exact_canonical_inputs")
    if not isinstance(registry, Mapping):
        raise ContractError("unified report lacks exact_canonical_inputs registry")
    _require_frozen_sources(registry, context="unified report exact-input registry")
    if report.get("exact_canonical_inputs_registry_sha256") != content_sha256(registry):
        raise ContractError("exact canonical input registry hash mismatch")


class PartAUnifiedDataset:
    """Select a persisted unified split from validated canonical roots."""

    def __init__(
        self,
        source_roots: Mapping[str, str | Path],
        manifest_path: str | Path,
        *,
        split: str,
        report_path: str | Path,
        expected_inventory: Mapping[str, Mapping[str, int]] | None = FROZEN_SOURCE_INVENTORY,
    ) -> None:
        index_rows = load_unified_rows(manifest_path, split=split)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        validate_unified_report(report, expected_inventory=expected_inventory)
        registry = report.get("exact_canonical_inputs")
        assert isinstance(registry, Mapping)
        validate_exact_input_registry(source_roots, registry)
        all_rows = load_unified_rows(manifest_path)
        if content_sha256(all_rows) != report.get("manifest_rows_sha256"):
            raise ContractError("unified manifest rows hash differs from report")
        if file_sha256(Path(manifest_path)) != report.get("manifest_file_sha256"):
            raise ContractError("unified manifest file hash differs from report")
        required_sources = {str(row["source_dataset"]) for row in index_rows}
        if required_sources - set(source_roots):
            raise ContractError(f"missing canonical roots: {sorted(required_sources - set(source_roots))}")
        canonical = PartACanonicalDataset(
            {source: source_roots[source] for source in sorted(required_sources)},
            fixture_only=False,
            require_fixtures=False,
        )
        sample_map: dict[str, PartASample] = {}
        for sample in canonical.samples:
            qa_id = str(sample.qa["qa_id"])
            if qa_id in sample_map:
                raise ContractError(f"ambiguous duplicate canonical qa_id: {qa_id}")
            sample_map[qa_id] = sample
        requested_ids = [str(row["qa_id"]) for row in index_rows]
        if len(requested_ids) != len(set(requested_ids)):
            raise ContractError("duplicate qa_id in persisted unified split")
        missing = sorted(set(requested_ids) - set(sample_map))
        if missing:
            raise ContractError(f"unified manifest references missing QA rows: {missing[:5]}")
        samples = []
        for row in index_rows:
            if row.get("source_sampling_key") != row.get("source_dataset"):
                raise ContractError(f"invalid source_sampling_key for {row.get('qa_id')}")
            sample = sample_map[str(row["qa_id"])]
            expected_identity = (
                str(row["source_dataset"]),
                str(row["scene_id"]),
                str(row["qa_id"]),
            )
            actual_identity = (
                str(sample.qa["source_dataset"]),
                str(sample.qa["scene_id"]),
                str(sample.qa["qa_id"]),
            )
            if actual_identity != expected_identity:
                raise ContractError(
                    f"unified/canonical identity mismatch: expected={expected_identity}, "
                    f"actual={actual_identity}"
                )
            samples.append(sample)
        self.index_rows = tuple(index_rows)
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PartASample:
        return self.samples[index]


@dataclass(frozen=True)
class PartACPUStateBatch:
    samples: tuple[PartASample, ...]
    targets: tuple[Any, ...]
    selection_audits: tuple[TargetSelectionAudit, ...]
    source_sampling_keys: tuple[str, ...]


class PartACPUStateCollator:
    """CPU contract collator; visual processor/model collation remains separate."""

    def __call__(self, samples: Sequence[PartASample]) -> PartACPUStateBatch:
        targets = []
        audits = []
        for sample in samples:
            target, audit = build_state_targets(sample, max_objects=384)
            targets.append(target)
            audits.append(audit)
        return PartACPUStateBatch(
            samples=tuple(samples),
            targets=tuple(targets),
            selection_audits=tuple(audits),
            source_sampling_keys=tuple(str(sample.qa["source_dataset"]) for sample in samples),
        )


def iter_source_balanced_indices(
    rows: Sequence[Mapping[str, Any]], *, seed: int, epoch: int = 0
) -> Iterator[int]:
    """Yield deterministic round-robin indices, cycling smaller sources."""
    import random

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["source_sampling_key"])].append(index)
    if not grouped:
        return
    rng = random.Random(f"{UNIFIED_SCHEMA_VERSION}:{seed}:{epoch}")
    for values in grouped.values():
        rng.shuffle(values)
    width = max(map(len, grouped.values()))
    sources = sorted(grouped)
    for offset in range(width):
        for source in sources:
            values = grouped[source]
            yield values[offset % len(values)]
