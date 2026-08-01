#!/usr/bin/env python3
"""Run the real five-fixture Part A T0-A smoke gate (never formal training)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np
from transformers import AutoConfig

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.canonical_data import (  # noqa: E402
    ExactMediaLoader,
    PartACanonicalDataset,
    PartASample,
    build_state_targets,
)
from parta.provenance import (  # noqa: E402
    atomic_json_dump,
    sha256_file,
    stable_sha256,
    ResolvedRunContract,
    write_run_status,
)
from parta.checkpoint import is_state_head_key, load_head_free_checkpoint  # noqa: E402
from parta.state_head import build_state_tap_from_packed  # noqa: E402
from parta.state_head import StateHeadConfig  # noqa: E402
from parta.state_loss import ObjectStateSetLoss, StateLossConfig, StateTargets  # noqa: E402
from parta.t0 import (  # noqa: E402
    FIXTURE_SCENE_IDS,
    T0_A_REQUIRED_CHECKS,
    T0Report,
    assert_finite_tensors,
    compare_tensors,
    component_shared_gradient_norms,
)
from parta.t0_runtime import PartAT0Collator, forward_visual_tap  # noqa: E402
from parta.training import attach_a1o_state_head, run_a1o_side_branch  # noqa: E402
from parta_data_contract import CANONICAL_CATEGORIES  # noqa: E402
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: E402
from qwenvl.model.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402
from qwenvl.model.geometry_encoders.vggt_encoder import VGGTEncoder  # noqa: E402

GUIDE_MIN_PIXELS = 8192
GUIDE_MAX_PIXELS = 268324
TENSOR_DIAGNOSTIC_CHUNK_ELEMENTS = 8_388_608


def _geometry_encoder_contract(config) -> dict[str, str]:
    explicit = getattr(config, "geometry_encoder_type", None)
    if explicit is None:
        return {
            "geometry_encoder_type_effective": "vggt",
            "geometry_encoder_type_source": "model_default_missing_legacy_field",
        }
    if str(explicit).lower() != "vggt":
        raise RuntimeError(f"Part A requires VGGT geometry encoder, got {explicit!r}")
    return {
        "geometry_encoder_type_effective": "vggt",
        "geometry_encoder_type_source": "config_explicit",
    }


def _geometry_encoder_module(model):
    encoder = getattr(model, "geometry_encoder", None)
    if encoder is None and getattr(model, "model", None) is not None:
        encoder = getattr(model.model, "geometry_encoder", None)
    return encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adt-root", type=Path, required=True)
    parser.add_argument("--hypersim-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vggt-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed != 42:
        parser.error("D-55 freezes T0-A seed exactly to 42")
    return args


def _load_local(model_path: Path, vggt_path: Path, dtype: torch.dtype, device: str):
    resolved = model_path.resolve()
    resolved_vggt = vggt_path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"local checkpoint directory required: {resolved}")
    if not resolved_vggt.is_dir():
        raise FileNotFoundError(f"local VGGT checkpoint directory required: {resolved_vggt}")
    config = AutoConfig.from_pretrained(resolved, local_files_only=True)
    geometry_contract = _geometry_encoder_contract(config)
    required = {
        "use_geometry_encoder": True,
        "use_feature_fusion_module": True,
        "use_patch_size_alin": True,
        "use_geometry_inputs": True,
        "geometry_deepstack_indexes_pro": [
            [7, [0]], [10, [1]], [13, [2]], [16, [3]], [19, [4]], [22, [5]],
        ],
        "use_deepstack_importance_gate": "all",
        "use_deepstack_global_gate": "all",
    }
    wrong = {
        name: getattr(config, name, None)
        for name, expected in required.items()
        if getattr(config, name, None) != expected
    }
    if wrong:
        raise RuntimeError(f"checkpoint is not reproduced GUIDE E-01 configuration: {wrong}")
    if getattr(config, "use_mope", False):
        raise RuntimeError("Part A baseline must be GUIDE with MoPE disabled")
    if not getattr(config, "geometry_deepstack_indexes_pro", None):
        raise RuntimeError("GUIDE checkpoint lacks geometry_deepstack_indexes_pro")
    processor = Qwen3VLProcessor.from_pretrained(resolved, local_files_only=True)
    image_processor = processor.image_processor
    pixel_sources = {}
    for name, expected in (("min_pixels", GUIDE_MIN_PIXELS), ("max_pixels", GUIDE_MAX_PIXELS)):
        actual = getattr(image_processor, name, None)
        if actual is None:
            setattr(image_processor, name, expected)
            pixel_sources[name] = "runner_frozen_missing_artifact_field"
        elif int(actual) != expected:
            raise RuntimeError(f"GUIDE processor {name} mismatch: {actual} != {expected}")
        else:
            pixel_sources[name] = "processor_artifact_explicit"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        resolved,
        config=config,
        dtype=dtype,
        local_files_only=True,
        geometry_encoder_path=str(resolved_vggt),
        attn_implementation="flash_attention_2",
    )
    if model.__class__.__module__ != "qwenvl.model.modeling_qwen3_vl":
        raise RuntimeError(f"unexpected model class module: {model.__class__.__module__}")
    if getattr(model.config, "_attn_implementation", None) != "flash_attention_2":
        raise RuntimeError("T0-A requires GUIDE's flash_attention_2 implementation")
    if not isinstance(_geometry_encoder_module(model), VGGTEncoder):
        raise RuntimeError("loaded GUIDE geometry encoder is not VGGTEncoder")
    model.to(torch.device(device))
    return processor, model, {
        **geometry_contract,
        "processor_min_pixels": GUIDE_MIN_PIXELS,
        "processor_max_pixels": GUIDE_MAX_PIXELS,
        "processor_pixel_field_sources": pixel_sources,
    }


def _startup_resource_preflight(
    output_parent: Path,
    guide_artifact: dict[str, object],
    vggt_artifact: dict[str, object],
    device: str,
) -> dict[str, object]:
    weight_bytes = sum(
        int(item["size_bytes"])
        for artifact in (guide_artifact, vggt_artifact)
        for item in artifact["ordered_shards"]
    )
    required_disk = weight_bytes + max(weight_bytes // 20, 512 * 1024**2)
    disk_free = shutil.disk_usage(output_parent).free
    failures = []
    if disk_free < required_disk:
        failures.append(
            f"insufficient disk: free={disk_free}, required={required_disk}"
        )
    if not device.startswith("cuda") or not torch.cuda.is_available():
        failures.append("T0-A requires an available CUDA device")
        index, capability, free_cuda, total_cuda = None, None, None, None
    else:
        cuda_device = torch.device(device)
        index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            failures.append(f"CUDA device index out of range: {index}")
            capability, free_cuda, total_cuda = None, None, None
        else:
            capability = torch.cuda.get_device_capability(index)
            free_cuda, total_cuda = torch.cuda.mem_get_info(index)
            if capability[0] < 8:
                failures.append(f"CUDA capability {capability} is below required major 8")
            if free_cuda <= 0:
                failures.append("selected CUDA device reports no free memory")
    try:
        import flash_attn
        flash_version = getattr(flash_attn, "__version__", "unknown")
    except Exception as error:
        flash_version = None
        failures.append(f"FlashAttention2 import failed: {error}")
    return {
        "passed": not failures,
        "failures": failures,
        "estimated_smoke_checkpoint_bytes": weight_bytes,
        "required_transaction_disk_bytes": required_disk,
        "disk_free_bytes": disk_free,
        "cuda_device_index": index,
        "cuda_capability": list(capability) if capability is not None else None,
        "cuda_memory_free_bytes": free_cuda,
        "cuda_memory_total_bytes": total_cuda,
        "flash_attn_version": flash_version,
    }


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Read all and only model shards from a local T0 A1 checkpoint."""
    resolved = path.resolve()
    if resolved.is_dir():
        indexes = sorted(resolved.glob("*.index.json"))
        if indexes:
            if len(indexes) != 1:
                raise RuntimeError("A1 checkpoint has ambiguous model indexes")
            with indexes[0].open(encoding="utf-8") as handle:
                index = json.load(handle)
            files = [resolved / name for name in sorted(set(index["weight_map"].values()))]
        else:
            files = sorted(resolved.glob("*.safetensors"))
            if not files:
                files = [candidate for candidate in sorted(resolved.glob("*.bin")) if "optimizer" not in candidate.name]
    else:
        files = [resolved]
    if not files or any(not candidate.is_file() for candidate in files):
        raise FileNotFoundError("A1 checkpoint model shards are missing")
    state: dict[str, torch.Tensor] = {}
    for candidate in files:
        if candidate.suffix == ".safetensors":
            from safetensors.torch import load_file
            shard = load_file(str(candidate), device="cpu")
        else:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            shard = payload.get("model", payload) if isinstance(payload, dict) else payload
        if not isinstance(shard, dict) or not all(isinstance(value, torch.Tensor) for value in shard.values()):
            raise TypeError(f"A1 shard is not a tensor state_dict: {candidate}")
        duplicate = set(state).intersection(shard)
        if duplicate:
            raise RuntimeError(f"duplicate A1 checkpoint keys: {sorted(duplicate)[:5]}")
        state.update(shard)
    return state


def _checkpoint_artifact_provenance(path: Path) -> dict[str, object]:
    """D-55 fingerprint for indexed or explicit single-weight checkpoints."""
    resolved = path.resolve()
    if not resolved.is_dir():
        record = {
            "name": resolved.name, "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
        return {
            "mode": "no_index_explicit_manifest", "index": None,
            "config_files": [], "ordered_shards": [record],
            "artifact_sha256": stable_sha256({
                "mode": "no_index_explicit_manifest", "config_files": [],
                "ordered_shards": [record],
            }),
        }
    indexes = sorted(resolved.glob("*.index.json"))
    if len(indexes) > 1:
        raise RuntimeError("D-55 checkpoint has ambiguous index JSON files")
    config_paths = sorted(
        item for item in resolved.glob("*.json") if not item.name.endswith(".index.json")
    )
    config_records = [{
        "name": item.name, "size_bytes": item.stat().st_size, "sha256": sha256_file(item),
    } for item in config_paths]
    if not indexes:
        weights = sorted(resolved.glob("*.safetensors"))
        if not weights:
            weights = [
                item for item in sorted(resolved.glob("*.bin"))
                if "optimizer" not in item.name
            ]
        if not weights:
            raise FileNotFoundError("checkpoint has no explicit local weight file")
        if len(weights) != 1:
            raise RuntimeError("sharded checkpoint requires an index JSON")
        shard_records = [{
            "name": item.name, "size_bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        } for item in weights]
        payload = {
            "mode": "no_index_explicit_manifest", "config_files": config_records,
            "ordered_shards": shard_records,
        }
        return {
            **payload, "index": None, "artifact_sha256": stable_sha256(payload),
        }
    index_path = indexes[0]
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("A1 checkpoint index has no nonempty weight_map")
    if not all(isinstance(key, str) and key for key in weight_map):
        raise RuntimeError("A1 checkpoint index contains invalid parameter keys")
    ordered_names = list(dict.fromkeys(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name for name in ordered_names):
        raise RuntimeError("A1 checkpoint index contains unsafe shard names")
    candidates = sorted(resolved.glob("*.safetensors")) or [
        item for item in sorted(resolved.glob("*.bin")) if "optimizer" not in item.name
    ]
    if {item.name for item in candidates} != set(ordered_names):
        raise RuntimeError("A1 checkpoint index/shard set mismatch")
    records = [{
        "name": name,
        "size_bytes": (resolved / name).stat().st_size,
        "sha256": sha256_file(resolved / name),
    } for name in ordered_names]
    index_record = {
        "name": index_path.name,
        "size_bytes": index_path.stat().st_size,
        "sha256": sha256_file(index_path),
        "ordered_weight_map_sha256": stable_sha256(sorted(weight_map.items())),
    }
    return {
        "mode": "indexed_weight_map", "index": index_record,
        "config_files": config_records,
        "ordered_shards": records,
        "artifact_sha256": stable_sha256({
            "mode": "indexed_weight_map", "index": index_record,
            "config_files": config_records, "ordered_shards": records,
        }),
    }


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    records = []
    for key, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        import hashlib
        records.append((key, list(value.shape), str(value.dtype), hashlib.sha256(raw).hexdigest()))
    return stable_sha256(records)


def _validate_manifest_sampling(
    path: Path, source: str, expected_base_interval: float
) -> dict[str, object]:
    if source not in {"adt", "hypersim"}:
        raise ValueError(f"unsupported manifest source: {source}")
    rows = 0
    binding_hashes = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            location = f"{path}:{line_number}"
            if row.get("source_dataset") != source:
                raise RuntimeError(f"manifest source mismatch at {location}")
            frame_indices = row.get("actual_frame_indices")
            if source == "adt":
                actual = row.get("sampling_base_interval")
                if actual is None:
                    actual = row.get("sampling_parameters", {}).get("base_interval")
                if actual is None or float(actual) != expected_base_interval:
                    raise RuntimeError(
                        f"manifest base_interval mismatch at {location}: {actual}"
                    )
                if (
                    row.get("media_kind") != "video"
                    or not isinstance(frame_indices, list)
                    or not 16 <= len(frame_indices) <= 32
                ):
                    raise RuntimeError(f"ADT exact video frame contract mismatch at {location}")
            else:
                top_level_interval = row.get("sampling_base_interval")
                sampling_parameters = row.get("sampling_parameters")
                if sampling_parameters is None:
                    nested_interval = None
                elif isinstance(sampling_parameters, dict):
                    nested_interval = sampling_parameters.get("base_interval")
                else:
                    raise RuntimeError(
                        f"Hypersim sampling_parameters contract mismatch at {location}"
                    )
                if top_level_interval is not None or nested_interval is not None:
                    raise RuntimeError(
                        f"Hypersim base_interval is inapplicable at {location}: "
                        f"top_level={top_level_interval}, nested={nested_interval}"
                    )
                if (
                    row.get("media_kind") != "image"
                    or not isinstance(frame_indices, list)
                    or len(frame_indices) != 1
                ):
                    raise RuntimeError(f"Hypersim single-frame contract mismatch at {location}")
                evidence = row.get("evidence_frame_indices")
                if (
                    row.get("qa_evidence_scope") != "frame_verified"
                    or row.get("qa_visual_support_verified") is not True
                    or evidence != frame_indices
                ):
                    raise RuntimeError(
                        f"Hypersim frame-verified evidence contract mismatch at {location}"
                    )
            binding = row.get("frame_binding_sha256")
            if not isinstance(binding, str) or len(binding) != 64:
                raise RuntimeError(f"manifest binding provenance missing at {location}")
            binding_hashes.append(binding)
            rows += 1
    if rows == 0:
        raise RuntimeError(f"empty exact manifest: {path}")
    return {
        "path": str(path.resolve()), "rows": rows,
        "source_dataset": source,
        "sampling_contract": (
            "guide_video_base_interval_v1" if source == "adt"
            else "single_frame_verified_v1"
        ),
        "base_interval": expected_base_interval if source == "adt" else None,
        "content_sha256": sha256_file(path),
        "ordered_binding_sha256": stable_sha256(binding_hashes),
    }


def _tensor_diagnostic(tensor: torch.Tensor) -> dict[str, object]:
    # QA logits for a 32-frame fixture can contain billions of values.  A
    # whole-tensor `.float()` plus boolean indexing temporarily needs several
    # extra GiB and can OOM after an otherwise successful forward/backward.
    # Compute the exact same audit statistics in bounded chunks instead.
    flat = tensor.detach().reshape(-1)
    nonfinite_count = 0
    minimum = None
    maximum = None
    squared_norm = 0.0
    with torch.no_grad():
        for start in range(0, flat.numel(), TENSOR_DIAGNOSTIC_CHUNK_ELEMENTS):
            chunk = flat[
                start : start + TENSOR_DIAGNOSTIC_CHUNK_ELEMENTS
            ].float()
            finite = torch.isfinite(chunk)
            finite_count = int(finite.sum().item())
            nonfinite_count += chunk.numel() - finite_count
            if finite_count:
                values = chunk[finite]
                piece_min = float(values.min().cpu())
                piece_max = float(values.max().cpu())
                minimum = piece_min if minimum is None else min(minimum, piece_min)
                maximum = piece_max if maximum is None else max(maximum, piece_max)
                squared_norm += float(
                    torch.sum(values * values, dtype=torch.float64).cpu()
                )
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "finite": nonfinite_count == 0,
        "nonfinite_count": nonfinite_count,
        "min": minimum,
        "max": maximum,
        "l2_norm": squared_norm**0.5 if nonfinite_count == 0 else None,
    }


def _context_preflight(processed, model) -> dict[str, object]:
    sequence_length = int(processed.model_kwargs["input_ids"].shape[1])
    visual_tokens = sum(processed.frame_token_counts)
    text_config = getattr(model.config, "text_config", model.config)
    context_limit = getattr(text_config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit <= 0:
        raise RuntimeError("model does not expose a positive context limit")
    if sequence_length > context_limit:
        raise RuntimeError(
            f"processed sequence exceeds model context: {sequence_length} > {context_limit}"
        )
    return {
        "frame_visual_tokens": list(processed.frame_token_counts),
        "total_visual_tokens": visual_tokens,
        "sequence_length": sequence_length,
        "context_limit": context_limit,
    }


def _e01_trainability_audit(model) -> dict[str, object]:
    frozen_vggt = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "geometry_encoder.vggt." in name
    ]
    if not frozen_vggt:
        raise RuntimeError("loaded GUIDE exposes no VGGT backbone parameters")
    frozen_violations = [
        name for name, parameter in frozen_vggt
        if parameter.requires_grad or parameter.grad is not None
    ]
    group_markers = {
        "guide_geometry_adapters": (
            "geometry_merger.", "deepstack_geo_merger_list.",
            "deepstack_geo_pro_merger_list.", "feature_fusion_module.",
            "deepstack_importance_gates.", "deepstack_global_gates.",
        ),
        "llm": ("language_model.",),
    }
    norms = {}
    matched_names = {}
    for group, markers in group_markers.items():
        squared = 0.0
        names = []
        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
                and any(marker in name for marker in markers)
                and "parta_state_head" not in name
            ):
                squared += float(parameter.grad.detach().float().square().sum().cpu())
                names.append(name)
        norms[group] = squared**0.5
        matched_names[group] = names
    passed = (
        not frozen_violations
        and all(names for names in matched_names.values())
        and all(np.isfinite(value) and value > 1e-12 for value in norms.values())
    )
    return {
        "passed": passed,
        "frozen_vggt_parameter_count": len(frozen_vggt),
        "frozen_vggt_violations": frozen_violations,
        "trainable_gradient_norms": norms,
        "trainable_gradient_parameter_names": matched_names,
    }


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    raise TypeError(f"unsupported model input type: {type(value).__name__}")


def _to_device(kwargs, device):
    return {key: _move(value, device) for key, value in kwargs.items()}


def _target_to(target: StateTargets, device: torch.device) -> StateTargets:
    values = vars(target).copy()
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            values[key] = value.to(device)
    return StateTargets(**values)


def _select_fixtures(dataset: PartACanonicalDataset) -> list[PartASample]:
    by_scene = {}
    for sample in dataset.samples:
        by_scene.setdefault(sample.qa["scene_id"], sample)
    if tuple(scene for scene in FIXTURE_SCENE_IDS if scene not in by_scene):
        raise RuntimeError("dataset lost a frozen fixture after validation")
    return [by_scene[scene] for scene in FIXTURE_SCENE_IDS]


def _reorder_target(target: StateTargets, order: torch.Tensor) -> StateTargets:
    values = vars(target).copy()
    for key in (
        "categories", "centers_world_m", "extents_m", "visibility",
        "category_valid", "center_valid", "extent_valid", "visibility_valid",
    ):
        values[key] = values[key][order]
    return StateTargets(**values)


def _empty_target(target: StateTargets) -> StateTargets:
    device = target.categories.device
    frames = target.visibility.shape[1]
    return StateTargets(
        categories=torch.empty(0, dtype=torch.long, device=device),
        centers_world_m=torch.empty(0, 3, device=device),
        extents_m=torch.empty(0, 3, device=device),
        visibility=torch.empty(0, frames, device=device),
        category_valid=torch.empty(0, dtype=torch.bool, device=device),
        center_valid=torch.empty(0, dtype=torch.bool, device=device),
        extent_valid=torch.empty(0, dtype=torch.bool, device=device),
        visibility_valid=torch.empty(0, frames, dtype=torch.bool, device=device),
        scene_scale_m=target.scene_scale_m,
        source_dataset=target.source_dataset,
        scene_id=target.scene_id,
    )


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("D-55 freezes T0-A seed exactly to 42")
    final_output = args.output.resolve()
    if final_output.exists():
        raise FileExistsError(f"transactional T0 output already exists: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.running-", dir=final_output.parent))
    args.output = staging
    failure_path = args.output / "t0_a_failure.json"
    config = None
    try:
        dtype = getattr(torch, args.dtype)
        torch.manual_seed(args.seed)
        config = {
            "schema_version": "parta_t0_a_config_v1",
            "adt_root": str(args.adt_root.resolve()),
            "hypersim_root": str(args.hypersim_root.resolve()),
            "media_root": str(args.media_root.resolve()),
            "model_path": str(args.model_path.resolve()),
            "vggt_path": str(args.vggt_path.resolve()),
            "smoke_checkpoint_role": "initialization_no_optimizer_updates",
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
            "batch_size": 1,
            "num_slots": 384,
            "video_min_frames": 16,
            "video_max_frames": 32,
            "base_interval": 1.0,
            "t0_b_status": "blocked_not_implemented",
            "fixture_count": 5,
            "qa_backward_steps": 5,
            "state_backward_steps": 5,
            "formal_training": False,
        }
        guide_artifact = _checkpoint_artifact_provenance(args.model_path)
        vggt_artifact = _checkpoint_artifact_provenance(args.vggt_path)
        resource_preflight = _startup_resource_preflight(
            final_output.parent, guide_artifact, vggt_artifact, args.device
        )
        config["resource_preflight"] = resource_preflight
        atomic_json_dump(config, args.output / "resolved_config.json")
        if not resource_preflight.get("passed", True):
            raise RuntimeError(
                f"startup resource preflight failed: {resource_preflight['failures']}"
            )
        manifest_provenance = {
            source: _validate_manifest_sampling(
                root / "qa_manifest_exact_verified.jsonl", source, config["base_interval"]
            )
            for source, root in (("adt", args.adt_root), ("hypersim", args.hypersim_root))
        }
        dataset = PartACanonicalDataset(
            {"adt": args.adt_root, "hypersim": args.hypersim_root}
        )
        samples = _select_fixtures(dataset)
        loader = ExactMediaLoader(args.media_root)
        processor, model, runtime_contract = _load_local(
            args.model_path, args.vggt_path, dtype, args.device
        )
        # Match the reproduced GUIDE training memory contract.  In particular,
        # GradientCheckpointingLayer only checkpoints while the model is in
        # training mode; leaving the smoke runner in eval mode retains every
        # decoder activation for the 32-frame ADT fixture and exceeds a 96 GiB
        # H20 before the first backward pass.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        config.update(runtime_contract)
        config.update({
            "use_cache": False,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "memory_contract": "guide_training_equivalent_v1",
        })
        atomic_json_dump(config, args.output / "resolved_config.json")
        model.train()
        device = torch.device(args.device)
        collator = PartAT0Collator(processor)
        hidden_size = int(model.config.text_config.hidden_size)
        head = attach_a1o_state_head(
            model,
            StateHeadConfig(hidden_size=hidden_size, num_categories=len(CANONICAL_CATEGORIES)),
        ).to(device=device, dtype=dtype)
        expected_head_keys = tuple(
            sorted(key for key in model.state_dict() if is_state_head_key(key))
        )
        if not expected_head_keys:
            raise RuntimeError("seeded T0 A1-O initialization has no state-head keys")
        initialization_before_sha256 = _state_digest(dict(model.state_dict()))
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
        manifest_digest = stable_sha256(manifest_provenance)
        exact_binding_digest = stable_sha256([
            {"qa_id": sample.qa["qa_id"], "sha256": sample.qa["frame_binding_sha256"]}
            for sample in samples
        ])
        run_contract = ResolvedRunContract(
            run_id="t0-a-five-fixture", experiment="parta-t0-a", seed=args.seed,
            resolved_config=config, manifest_sha256=manifest_digest,
            initialization_sha256=stable_sha256({
                "guide": guide_artifact["artifact_sha256"],
                "vggt": vggt_artifact["artifact_sha256"],
            }),
            code_revision=git_revision, exact_frame_binding_sha256=exact_binding_digest,
            output_dir=str(final_output), status="running",
        )
        status_path = args.output / "run_status.json"
        write_run_status(run_contract, status_path)
        report = T0Report("t0-a-five-fixture", stable_sha256(config), phase="t0-a")
        report.add_boolean(
            "fixed_fixtures",
            [sample.qa["scene_id"] for sample in samples] == list(FIXTURE_SCENE_IDS),
            scene_ids=[sample.qa["scene_id"] for sample in samples],
        )
        report.add_boolean("slot_count_k384", head.config.num_slots == 384)
        fixture_metrics = []
        first_processed = first_output = first_branch = first_target = None
        component_norms = {}
        active_components = {"existence"}
        prefix_measurements = []
        exact_measurements = []
        qa_losses = []
        finite_diagnostics = {}
        for sample in samples:
            images = loader.load(sample)
            processed = collator(sample, images)
            processed = type(processed)(
                model_kwargs=_to_device(processed.model_kwargs, device),
                sample=processed.sample,
                images=processed.images,
                frame_token_counts=processed.frame_token_counts,
                frame_ids=processed.frame_ids,
                media_kind=processed.media_kind,
                visual_prefix_before_question=processed.visual_prefix_before_question,
                visual_token_mask=processed.visual_token_mask.to(device),
                question_token_span=processed.question_token_span,
            )
            context_metrics = _context_preflight(processed, model)
            output = forward_visual_tap(model, processed)
            if not isinstance(getattr(output, "loss", None), torch.Tensor):
                raise RuntimeError("authoritative answer labels did not produce QA loss")
            qa_losses.append(float(output.loss.detach()))
            prefix_measurements.append({
                "scene_id": sample.qa["scene_id"],
                "passed": processed.visual_prefix_before_question,
                "visual_last": int(processed.visual_token_mask[0].nonzero()[-1]),
                "question_span": list(processed.question_token_span),
            })
            target, selection = build_state_targets(sample)
            target = _target_to(target, device)
            if target.category_valid.any():
                active_components.add("category")
            if target.center_valid.any():
                active_components.add("center")
            if target.extent_valid.any():
                active_components.add("extent")
            if target.visibility_valid.any():
                active_components.add("visibility")
            branch = run_a1o_side_branch(
                model,
                output.visual_state_hidden,
                output.visual_state_valid_mask,
                [processed.frame_token_counts],
                [processed.frame_ids],
                [processed.media_kind],
                [target],
                StateLossConfig(),
            )
            # This second pass only proves that A0 and A1-O bind the same
            # frames/masks/spans.  It contributes no loss and must not retain a
            # second full 32-frame autograd graph alongside the authoritative
            # QA/state graph above.
            with torch.no_grad():
                bypass_output = forward_visual_tap(model, processed)
            tap_a0 = build_state_tap_from_packed(
                output.visual_state_hidden, output.visual_state_valid_mask,
                [processed.frame_token_counts], [processed.frame_ids], [processed.media_kind],
            )
            tap_a1 = build_state_tap_from_packed(
                bypass_output.visual_state_hidden, bypass_output.visual_state_valid_mask,
                [processed.frame_token_counts], [processed.frame_ids], [processed.media_kind],
            )
            from parta.t0 import assert_exact_frame_contract
            assert_exact_frame_contract(
                tap_a0.frame_ids, tap_a1.frame_ids, tap_a0.valid_mask, tap_a1.valid_mask,
                tap_a0.frame_token_spans, tap_a1.frame_token_spans,
            )
            exact_measurements.append({"scene_id": sample.qa["scene_id"], "passed": True})
            losses = {name.removeprefix("loss_"): value for name, value in branch.losses.items() if name.startswith("loss_") and name != "loss_state"}
            shared = [parameter for name, parameter in model.named_parameters() if not name.startswith("parta_state_head.")]
            norms = component_shared_gradient_norms(losses, shared)
            enabled_here = {
                "existence": True,
                "category": bool(target.category_valid.any()),
                "center": bool(target.center_valid.any()),
                "extent": bool(target.extent_valid.any()),
                "visibility": bool(target.visibility_valid.any()),
            }
            for name, value in norms.items():
                if enabled_here.get(name, False):
                    component_norms.setdefault(name, []).append(value)
            (branch.losses["loss_state"] + output.loss).backward()
            assert_finite_tensors({
                "qa_logits": output.logits,
                "tap": output.visual_state_hidden,
                "state_loss": branch.losses["loss_state"],
                "qa_loss": output.loss,
            })
            diagnostic_tensors = {
                "qa_logits": output.logits,
                "visual_state_hidden": output.visual_state_hidden,
                **{
                    f"prediction.{field}": getattr(branch.predictions, field)
                    for field in (
                        "existence_logits", "category_logits", "center_world_normalized",
                        "extent_normalized", "visibility_logits", "slots",
                    )
                },
                **{
                    f"loss.{name}": value
                    for name, value in branch.losses.items()
                    if isinstance(value, torch.Tensor)
                },
                "loss.qa": output.loss,
            }
            fixture_diag = {name: _tensor_diagnostic(value) for name, value in diagnostic_tensors.items()}
            if not all(item["finite"] for item in fixture_diag.values()):
                raise AssertionError(f"nonfinite fixture diagnostics: {sample.qa['qa_id']}")
            finite_diagnostics[sample.qa["qa_id"]] = fixture_diag
            fixture_metrics.append({
                "scene_id": sample.qa["scene_id"],
                "qa_id": sample.qa["qa_id"],
                "frame_binding_sha256": sample.qa["frame_binding_sha256"],
                "media_kind": processed.media_kind,
                "frame_ids": list(processed.frame_ids),
                "frame_token_counts": list(processed.frame_token_counts),
                "context_preflight": context_metrics,
                "tap_valid_tokens": int(output.visual_state_valid_mask.sum().item()),
                "frame_token_spans": branch.tap.frame_token_spans[0].detach().cpu().tolist(),
                "actual_input_visible_object_count": selection.actual_input_visible_object_count,
                "truncated_object_ids": list(selection.truncated_object_ids),
            })
            if first_processed is None:
                first_processed, first_output, first_branch, first_target = (
                    processed, output, branch, target
                )
        report.add_boolean("visual_before_question", all(x["passed"] for x in prefix_measurements), fixtures=prefix_measurements)
        report.add_boolean("exact_frame_contract", len(exact_measurements) == len(samples), fixtures=exact_measurements)
        shape_ok = all(
            item["tap_valid_tokens"] == sum(item["frame_token_counts"])
            and item["frame_token_spans"][-1][1] == item["tap_valid_tokens"]
            and len(item["frame_ids"]) == len(item["frame_token_spans"])
            for item in fixture_metrics
        )
        report.add_boolean("shape_mask_frame_span", shape_ok, fixtures=fixture_metrics)
        e01_audit = _e01_trainability_audit(model)
        parameter_gradients = {}
        missing_required_gradients = []
        component_markers = {
            "existence": ("parta_state_head.existence.",),
            "category": ("parta_state_head.category.",),
            "center": ("parta_state_head.center.",),
            "extent": ("parta_state_head.extent.",),
            "visibility": ("parta_state_head.visibility.",),
        }
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            required = (
                any(marker in name for component in active_components for marker in component_markers[component])
                or name.startswith("parta_state_head.decoder")
                or name.startswith("parta_state_head.output_norm")
                or name == "parta_state_head.slot_queries"
                or "geometry_merger" in name
                or "feature_fusion" in name
                or "deepstack_" in name
            )
            if parameter.grad is None:
                if required:
                    missing_required_gradients.append(name)
                continue
            parameter_gradients[name] = _tensor_diagnostic(parameter.grad)
        invalid_gradients = [
            name for name, item in parameter_gradients.items() if not item["finite"]
        ]
        finite_passed = (
            bool(finite_diagnostics) and not missing_required_gradients and not invalid_gradients
        )
        report.add_boolean(
            "finite", finite_passed, tensors=finite_diagnostics,
            parameter_gradients=parameter_gradients,
            missing_required_parameter_gradients=missing_required_gradients,
            nonfinite_parameter_gradients=invalid_gradients,
        )
        if not finite_passed:
            raise AssertionError(
                f"finite/parameter-gradient gate failed: missing={missing_required_gradients}, "
                f"nonfinite={invalid_gradients}"
            )
        report.add_boolean(
            "qa_supervision_and_e01_trainability",
            bool(qa_losses) and all(np.isfinite(loss) for loss in qa_losses)
            and bool(e01_audit["passed"]),
            qa_losses=qa_losses,
            e01_trainability_audit=e01_audit,
        )
        if not report.payload["checks"]["qa_supervision_and_e01_trainability"]["passed"]:
            raise AssertionError("QA supervision/E-01 trainability smoke gate failed")
        enabled = {
            name: min(component_norms.get(name, (0.0,)))
            for name in sorted(active_components)
        }
        report.add_component_gradient_check(enabled, sorted(active_components))
        criterion = ObjectStateSetLoss(StateLossConfig())
        reference_losses = criterion(first_branch.predictions, [first_target])
        order = torch.arange(first_target.num_objects - 1, -1, -1, device=device)
        permuted_losses = criterion(
            first_branch.predictions, [_reorder_target(first_target, order)]
        )
        permutation_components = {
            name: compare_tensors(reference_losses[name], permuted_losses[name])
            for name in reference_losses if name.startswith("loss_")
        }
        report.add_boolean(
            "gt_permutation_invariance",
            all(item.passed for item in permutation_components.values()),
            components={name: vars(item) for name, item in permutation_components.items()},
        )
        empty_losses = criterion(
            first_branch.predictions, [_empty_target(first_target)]
        )
        empty_loss = empty_losses["loss_state"]
        report.add_boolean(
            "empty_gt", bool(torch.isfinite(empty_loss))
            and all(float(empty_losses[f"loss_{name}"].detach()) == 0.0 for name in ("category", "center", "extent", "visibility"))
            and float(empty_losses["loss_existence"].detach()) > 0.0,
            losses={name: float(value.detach()) for name, value in empty_losses.items() if name.startswith("loss_")},
        )
        masked_values = vars(first_target).copy()
        for key in ("category_valid", "center_valid", "extent_valid", "visibility_valid"):
            masked_values[key] = torch.zeros_like(masked_values[key])
        masked_losses = criterion(first_branch.predictions, [StateTargets(**masked_values)])
        masked_loss = masked_losses["loss_state"]
        report.add_boolean(
            "all_masked_no_object",
            bool(torch.isfinite(masked_loss)) and all(
                float(masked_losses[f"loss_{name}"].detach()) == 0.0
                for name in ("category", "center", "extent", "visibility")
            ),
            loss=float(masked_loss.detach()),
        )
        alternate = collator.with_question(
            first_processed, "Describe the same scene differently."
        )
        alternate = type(alternate)(**{**vars(alternate), "model_kwargs": _to_device(alternate.model_kwargs, device)})
        with torch.no_grad():
            alternate_output = forward_visual_tap(model, alternate)
            alternate_branch = run_a1o_side_branch(
                model, alternate_output.visual_state_hidden, alternate_output.visual_state_valid_mask,
                [alternate.frame_token_counts], [alternate.frame_ids], [alternate.media_kind],
                [first_target], StateLossConfig(),
            )
        alternate_tap = build_state_tap_from_packed(
            alternate_output.visual_state_hidden, alternate_output.visual_state_valid_mask,
            [alternate.frame_token_counts], [alternate.frame_ids], [alternate.media_kind],
        )
        first_tap = first_branch.tap
        from parta.t0 import assert_exact_frame_contract
        assert_exact_frame_contract(
            first_tap.frame_ids, alternate_tap.frame_ids,
            first_tap.valid_mask, alternate_tap.valid_mask,
            first_tap.frame_token_spans, alternate_tap.frame_token_spans,
        )
        question_comparisons = {
            "visual_state_hidden": compare_tensors(
                first_output.visual_state_hidden, alternate_output.visual_state_hidden
            ),
            **{
                f"state.{field}": compare_tensors(
                    getattr(first_branch.predictions, field),
                    getattr(alternate_branch.predictions, field),
                )
                for field in (
                    "existence_logits", "category_logits", "center_world_normalized",
                    "extent_normalized", "visibility_logits", "slots",
                )
            },
        }
        report.add_boolean(
            "question_invariance",
            all(item.passed for item in question_comparisons.values()),
            comparisons={name: vars(item) for name, item in question_comparisons.items()},
            qa_logits_intentionally_unconstrained=True,
        )
        initialization_after_sha256 = _state_digest(dict(model.state_dict()))
        if initialization_after_sha256 != initialization_before_sha256:
            raise AssertionError("T0-A changed model parameters despite having no optimizer step")
        resume_path = args.output / "t0_a1o_smoke_no_update.pt"
        torch.save(
            {
                "schema_version": "parta_t0_a1o_smoke_checkpoint_v1",
                "checkpoint_role": "initialization_no_optimizer_updates",
                "optimizer_steps": 0,
                "seed": args.seed,
                "model": model.state_dict(),
                "state_head_config": vars(head.config),
                "expected_state_head_keys": list(expected_head_keys),
                "parameter_sha256_before_backward": initialization_before_sha256,
                "parameter_sha256_after_backward": initialization_after_sha256,
            },
            resume_path,
        )
        smoke_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(smoke_payload, dict) or smoke_payload.get("checkpoint_role") != (
            "initialization_no_optimizer_updates"
        ):
            raise RuntimeError("serialized T0 smoke checkpoint role is invalid")
        if smoke_payload.get("optimizer_steps") != 0 or smoke_payload.get("seed") != args.seed:
            raise RuntimeError("serialized T0 smoke checkpoint optimizer/seed contract is invalid")
        if tuple(smoke_payload.get("expected_state_head_keys", ())) != expected_head_keys:
            raise RuntimeError("serialized T0 smoke checkpoint head-key manifest differs")
        if (
            smoke_payload.get("parameter_sha256_before_backward") != initialization_before_sha256
            or smoke_payload.get("parameter_sha256_after_backward") != initialization_after_sha256
        ):
            raise RuntimeError("serialized T0 smoke checkpoint parameter digests differ")
        a1_state = smoke_payload.get("model")
        if not isinstance(a1_state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in a1_state.items()
        ):
            raise RuntimeError("serialized T0 smoke checkpoint model state is invalid")
        a1_artifact = _checkpoint_artifact_provenance(resume_path)
        a1_state_sha256 = _state_digest(a1_state)
        if a1_state_sha256 != initialization_before_sha256:
            raise RuntimeError("serialized T0 smoke checkpoint changed initialized parameters")
        a1_shared_state_sha256 = _state_digest({
            key: value for key, value in a1_state.items() if not is_state_head_key(key)
        })
        reference_logits = first_output.logits.detach().cpu()
        head_config = head.config
        reference_predictions = {
            field: getattr(first_branch.predictions, field).detach().clone()
            for field in ("existence_logits", "category_logits", "center_world_normalized", "extent_normalized", "visibility_logits")
        }
        del (
            first_output,
            first_branch,
            head,
            model,
            output,
            branch,
            alternate_output,
            alternate_branch,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _, clean_model, _ = _load_local(args.model_path, args.vggt_path, dtype, args.device)
        clean_model.eval()
        audit = load_head_free_checkpoint(
            clean_model, a1_state, expected_state_head_keys=expected_head_keys
        )
        if not audit.passed:
            raise RuntimeError(f"head-free A1 restore audit failed: {audit}")
        restored_backbone_sha256 = _state_digest(dict(clean_model.state_dict()))
        backbone_digest_matches = restored_backbone_sha256 == a1_shared_state_sha256
        if not backbone_digest_matches:
            raise RuntimeError("head-free restored backbone digest differs from A1 shared digest")
        with torch.no_grad():
            clean_output = forward_visual_tap(clean_model, first_processed)
        head_free_cmp = compare_tensors(reference_logits, clean_output.logits.detach().cpu())
        report.add_boolean(
            "head_free_equivalence", head_free_cmp.passed and backbone_digest_matches,
            logits=vars(head_free_cmp), expected_shared_sha256=a1_shared_state_sha256,
            restored_backbone_sha256=restored_backbone_sha256,
            backbone_digest_matches=backbone_digest_matches,
        )
        del clean_model, clean_output, a1_state
        _, resumed_model, _ = _load_local(args.model_path, args.vggt_path, dtype, args.device)
        attach_a1o_state_head(resumed_model, head_config).to(device=device, dtype=dtype)
        resumed_state = torch.load(resume_path, map_location="cpu", weights_only=True)["model"]
        resumed_model.load_state_dict(resumed_state, strict=True)
        resumed_model.eval()
        with torch.no_grad():
            resumed_output = forward_visual_tap(resumed_model, first_processed)
            resumed_branch = run_a1o_side_branch(
                resumed_model, resumed_output.visual_state_hidden, resumed_output.visual_state_valid_mask,
                [first_processed.frame_token_counts], [first_processed.frame_ids],
                [first_processed.media_kind], [first_target], StateLossConfig(),
            )
        resume_checks = [compare_tensors(reference_logits, resumed_output.logits.detach().cpu())]
        for field in ("existence_logits", "category_logits", "center_world_normalized", "extent_normalized", "visibility_logits"):
            resume_checks.append(compare_tensors(reference_predictions[field], getattr(resumed_branch.predictions, field)))
        report.add_boolean(
            "save_resume_equivalence", all(item.passed for item in resume_checks),
            checkpoint_sha256=sha256_file(resume_path), comparisons=[vars(item) for item in resume_checks],
        )
        missing = sorted(T0_A_REQUIRED_CHECKS - set(report.payload["checks"]))
        if missing:
            raise AssertionError(f"runner omitted T0-A checks: {missing}")
        payload = report.finalize(str(args.output / "t0_a_report.json"))
        environment = {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        provenance = {
            "schema_version": "parta_t0_a_provenance_v1",
            "status": payload["status"],
            "resolved_config_sha256": stable_sha256(config),
            "checkpoint_sha256": guide_artifact["artifact_sha256"],
            "guide_checkpoint_artifact": guide_artifact,
            "geometry_encoder_type_effective": runtime_contract[
                "geometry_encoder_type_effective"
            ],
            "geometry_encoder_type_source": runtime_contract[
                "geometry_encoder_type_source"
            ],
            "processor_context_contract": {
                key: runtime_contract[key]
                for key in (
                    "processor_min_pixels",
                    "processor_max_pixels",
                    "processor_pixel_field_sources",
                )
            },
            "resource_preflight": resource_preflight,
            "a1_checkpoint_state_sha256": a1_state_sha256,
            "a1_checkpoint_role": "initialization_no_optimizer_updates",
            "a1_checkpoint_optimizer_steps": 0,
            "parameter_sha256_before_backward": initialization_before_sha256,
            "parameter_sha256_after_backward": initialization_after_sha256,
            "a1_loaded_shared_state_sha256": a1_shared_state_sha256,
            "head_free_restored_backbone_sha256": restored_backbone_sha256,
            "head_free_load_audit": vars(audit),
            "vggt_checkpoint_sha256": vggt_artifact["artifact_sha256"],
            "vggt_checkpoint_artifact": vggt_artifact,
            "manifest_sha256": {
                source: details["content_sha256"]
                for source, details in manifest_provenance.items()
            },
            "manifest_provenance": manifest_provenance,
            "a1_checkpoint_artifact": a1_artifact,
            "git_revision": git_revision,
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True)),
            "exact_frame_binding_sha256": exact_binding_digest,
            "environment": environment,
            "environment_sha256": stable_sha256(environment),
            "t0_b_status": "blocked_not_implemented",
        }
        atomic_json_dump(provenance, args.output / "provenance.json")
        write_run_status(
            replace(run_contract, status="complete", checkpoint_sha256=a1_state_sha256),
            status_path,
        )
        os.replace(args.output, final_output)
    except BaseException as error:
        failure = {
            "schema_version": "parta_t0_a_failure_v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "resolved_config_sha256": stable_sha256(config) if config is not None else None,
            "t0_b_status": "blocked_not_implemented",
        }
        atomic_json_dump(failure, failure_path)
        atomic_json_dump(failure, args.output / "t0_a_report.json")
        atomic_json_dump(
            {
                "schema_version": "parta_t0_a_provenance_v1",
                "status": "failed",
                "resolved_config_sha256": failure["resolved_config_sha256"],
                "git_revision": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=PROJECT,
                    text=True,
                    capture_output=True,
                ).stdout.strip() or None,
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                },
            },
            args.output / "provenance.json",
        )
        if "run_contract" in locals() and "status_path" in locals() and status_path.exists():
            with status_path.open(encoding="utf-8") as handle:
                current_status = json.load(handle).get("status")
            if current_status == "running":
                write_run_status(replace(run_contract, status="failed"), status_path)
        if args.output.exists() and not final_output.exists():
            os.replace(args.output, final_output)
        raise


if __name__ == "__main__":
    main()
