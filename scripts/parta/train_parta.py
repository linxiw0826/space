#!/usr/bin/env python3
"""Matched A0/A1-O formal training entrypoint.

Numerical defaults remain unfrozen until D-62 execution evidence and Gate@CONFIG.
This script is intentionally not a T0-B, overfit, profiling, or eval runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.canonical_data import ExactMediaLoader, PartASample, build_state_targets  # noqa: E402
from parta.checkpoint import export_head_free_checkpoint  # noqa: E402
from parta.checkpoint_selection import RULE as CHECKPOINT_SELECTION_RULE  # noqa: E402
from parta.checkpoint_selection import select_validation_checkpoint  # noqa: E402
from parta.distributed import barrier, initialize_distributed, synchronize_failure  # noqa: E402
from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from parta.gate_orchestration import validate_formal_training_authorization  # noqa: E402
from parta.runner import (  # noqa: E402
    PartATrainBatch,
    PartATrainConfig,
    PartATrainer,
    SourceBalancedCursor,
    SharedForwardOutput,
    config_sha256,
    assert_matched_fairness,
    attach_a1o_head_without_advancing_shared_rng,
    matched_fairness_payload,
    seed_matched_run,
    validate_single_step_execution_contract,
)
from parta.state_head import StateHeadConfig  # noqa: E402
from parta.training import (  # noqa: E402
    consume_a1o_forward_result,
    install_a1o_forward_integration,
    prepare_a1o_forward_request,
)
from parta.training_log import JsonlTrainingLogger  # noqa: E402
from parta.unified_data import (  # noqa: E402
    PartAUnifiedDataset,
    file_sha256,
    iter_source_balanced_indices,
    load_engineering_subset_artifact,
    load_unified_rows,
)
from parta.resource_profile_contract import (normalize_profile_worker_argv,
    normalized_contract_sha256)  # noqa: E402
from parta_data_contract import CANONICAL_CATEGORIES  # noqa: E402

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO
)
LOGGER = logging.getLogger("parta.train")


def _write_rank_failure(error: BaseException, stage: str, output_dir: Path) -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    failure_dir = output_dir / "rank_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}") if cuda_available else None
    message = str(error)
    atomic_json_dump({
        "schema_version": "parta_rank_failure_v1", "rank": rank, "local_rank": local_rank,
        "stage": stage, "error_type": type(error).__name__, "reason": message,
        "oom": isinstance(error, torch.OutOfMemoryError) or "out of memory" in message.lower(),
        "device_name": torch.cuda.get_device_name(device) if device else None,
        "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device else None,
        "finite": None,
    }, failure_dir / f"rank-{rank}.json")


def _coordinate_step_failure(error: BaseException | None, stage: str, args,
                             context) -> None:
    """Keep formal failure coordination while profile ranks fail without collectives."""
    if error is not None:
        _write_rank_failure(error, stage, args.output_dir)
    if args.engineering_mode == "resource_profile":
        if error is not None:
            # Publish first, then let the parent timeout terminate peer ranks.
            raise error
        return
    if synchronize_failure(error is not None, context):
        if error is not None:
            raise error
        raise RuntimeError(f"peer rank failed during {stage}")


def _write_top_level_rank_failure_if_absent(error: BaseException,
                                            output_dir: Path) -> None:
    """Do not replace a more specific load/train-stage rank artifact."""
    rank = int(os.environ.get("RANK", "0"))
    existing = output_dir / "rank_failures" / f"rank-{rank}.json"
    if not existing.is_file():
        _write_rank_failure(error, "train_parta_main", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("a0", "a1o"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--source-root", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vggt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matched-contract", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-state", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributed-strategy", choices=("none", "ddp", "fsdp"), default="none")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run exactly two optimizer steps")
    parser.add_argument("--unified-gate", type=Path)
    parser.add_argument("--frozen-config-artifact", type=Path)
    parser.add_argument("--engineering-subset", type=Path)
    parser.add_argument(
        "--engineering-mode",
        choices=("overfit", "matched_runner", "resource_profile"),
    )
    parser.add_argument("--required-frame-count", choices=(32,), type=int)
    parser.add_argument("--val-batches-per-source", type=int, default=8)
    return parser.parse_args()


def _source_roots(values: list[str]) -> dict[str, Path]:
    roots = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source-root must be SOURCE=PATH, got {value!r}")
        source, path = value.split("=", 1)
        if not source or source in roots:
            raise ValueError(f"invalid or duplicate source root: {source!r}")
        roots[source] = Path(path)
    return roots


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _forward_adapter(model, model_inputs, return_tap):
    from parta.t0_runtime import forward_visual_tap

    fixture = model_inputs["fixture"]
    side_request = model_inputs.get("a1o_side_request")
    if side_request is not None:
        prepare_a1o_forward_request(model, **side_request)
    output = forward_visual_tap(model, fixture)
    return SharedForwardOutput(
        qa_loss=output.loss,
        visual_state_hidden=output.visual_state_hidden if return_tap else None,
        visual_state_valid_mask=output.visual_state_valid_mask if return_tap else None,
        a1o_side_branch=(consume_a1o_forward_result(model) if side_request is not None else None),
    )


def _resource_profile_sample(sample: PartASample, frame_count: int) -> PartASample:
    """Derive an auditable exact-view profile binding from a longer video sample."""
    indices = list(sample.qa.get("actual_frame_indices", ()))
    keys = list(sample.qa.get("actual_frame_keys", ()))
    if sample.qa.get("media_kind") != "video" or len(indices) < frame_count:
        raise ValueError("resource profile requires a video binding with enough exact views")
    positions = [round(index * (len(indices) - 1) / (frame_count - 1))
                 for index in range(frame_count)]
    if len(set(positions)) != frame_count:
        raise ValueError("resource profile frame derivation produced duplicate positions")
    selected_indices = [indices[position] for position in positions]
    selected_keys = [keys[position] for position in positions]
    frame_by_key = {str(frame["frame_key"]): frame for frame in sample.frames}
    selected_frames = tuple(frame_by_key[str(key)] for key in selected_keys)
    visible = sorted({
        str(item["object_id"])
        for frame in selected_frames
        for item in frame.get("visible_nodes", ())
        if item.get("evidence_present") is True and item.get("visible") is True
        and item.get("field_mask", {}).get("visibility") is True
    })
    qa = dict(sample.qa)
    qa.update({
        "actual_frame_indices": selected_indices,
        "actual_frame_keys": selected_keys,
        "actual_visible_object_ids": visible,
        "resource_profile_parent_frame_binding_sha256": sample.qa.get("frame_binding_sha256"),
        "resource_profile_sampling_policy": "uniform_positions_without_replacement_v1",
        "resource_profile_required_frame_count": frame_count,
    })
    qa.pop("selected_object_ids", None)
    qa.pop("truncated_object_ids", None)
    derived_binding = stable_sha256({
        "qa_id": qa["qa_id"], "actual_frame_indices": selected_indices,
        "actual_frame_keys": selected_keys, "parent": qa["resource_profile_parent_frame_binding_sha256"],
        "policy": qa["resource_profile_sampling_policy"],
    })
    qa["frame_binding_sha256"] = derived_binding
    qa["actual_frame_binding_sha256"] = derived_binding
    return PartASample(scene=sample.scene, frames=selected_frames, qa=qa)


def _fixture_on_device(fixture, device):
    return type(fixture)(
        model_kwargs=_to_device(fixture.model_kwargs, device), sample=fixture.sample,
        images=fixture.images, frame_token_counts=fixture.frame_token_counts,
        frame_ids=fixture.frame_ids, media_kind=fixture.media_kind,
        visual_prefix_before_question=fixture.visual_prefix_before_question,
        visual_token_mask=fixture.visual_token_mask.to(device),
        question_token_span=fixture.question_token_span,
    )


def _source_balanced_val_qa_loss(model, dataset, media_loader, collator, device, per_source):
    if per_source < 1:
        raise ValueError("val-batches-per-source must be positive")
    selected = {source: [] for source in ("adt", "hypersim", "scannetppv2")}
    for sample in dataset.samples:
        source = str(sample.qa["source_dataset"])
        if len(selected[source]) < per_source:
            selected[source].append(sample)
    if any(len(rows) != per_source for rows in selected.values()):
        raise ValueError("validation split lacks the frozen per-source evaluation budget")
    was_training = model.training
    model.eval()
    losses = {}
    with torch.no_grad():
        for source, samples in selected.items():
            values = []
            for sample in samples:
                fixture = _fixture_on_device(collator(sample, media_loader.load(sample)), device)
                output = _forward_adapter(model, {"fixture": fixture}, False)
                if output.qa_loss is None or not torch.isfinite(output.qa_loss).all():
                    raise ValueError(f"non-finite validation QA loss for {source}")
                values.append(float(output.qa_loss.detach().float().cpu()))
            losses[source] = sum(values) / len(values)
    if was_training:
        model.train()
    return losses


def main() -> None:
    args = parse_args()
    engineering = args.dry_run or args.engineering_mode is not None
    if not engineering and (args.unified_gate is None or args.frozen_config_artifact is None):
        raise ValueError("formal non-dry-run training requires --unified-gate and --frozen-config-artifact")
    if not engineering and args.engineering_subset is not None:
        raise ValueError("formal training must use full train without engineering-subset reweighting")
    if engineering and args.engineering_subset is None:
        raise ValueError("engineering runner requires the frozen train-internal subset artifact")
    if engineering and args.resume is not None:
        raise ValueError("engineering dry-run checkpoints are non-promotable and cannot be resumed")
    if args.required_frame_count is not None and args.engineering_mode != "resource_profile":
        raise ValueError("--required-frame-count is reserved for resource_profile engineering mode")
    context = initialize_distributed()
    if context.world_size > 1 and args.distributed_strategy == "none":
        raise ValueError("torchrun requires --distributed-strategy ddp or fsdp")
    if context.world_size == 1 and args.distributed_strategy != "none":
        raise ValueError("distributed strategy requires torchrun WORLD_SIZE>1")
    effective_device = f"cuda:{context.local_rank}" if context.world_size > 1 else args.device
    if args.engineering_mode == "resource_profile":
        if context.world_size != 4:
            raise ValueError("resource profile requires exactly four distributed ranks")
        if not torch.cuda.is_available() or "NVIDIA H20" not in torch.cuda.get_device_name(
            torch.device(effective_device)
        ):
            raise ValueError("resource profile requires NVIDIA H20 on every rank")
    if args.output_dir.exists() and args.resume is None and context.is_primary:
        raise FileExistsError(f"output exists; use --resume explicitly: {args.output_dir}")
    if context.is_primary:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(context)
    seed_matched_run(args.seed)
    source_roots = _source_roots(args.source_root)
    dataset = PartAUnifiedDataset(
        source_roots, args.manifest, split="train", report_path=args.manifest_report
    )
    val_dataset = None if engineering else PartAUnifiedDataset(
        source_roots, args.manifest, split="val", report_path=args.manifest_report
    )
    engineering_artifact = None
    if engineering:
        report_for_subset = json.loads(args.manifest_report.read_text(encoding="utf-8"))
        frozen_subset = report_for_subset.get("engineering_subset", {})
        if (
            not isinstance(frozen_subset, Mapping)
            or Path(str(frozen_subset.get("path", ""))).resolve() != args.engineering_subset.resolve()
            or not isinstance(frozen_subset.get("sha256"), str)
            or frozen_subset.get("promotable_to_formal_training") is not False
        ):
            raise ValueError("engineering subset is not the artifact frozen by manifest report")
        qa_by_source: dict[str, list[Mapping]] = {source: [] for source in source_roots}
        for sample in dataset.samples:
            qa_by_source[str(sample.qa["source_dataset"])].append(sample.qa)
        engineering_artifact = load_engineering_subset_artifact(
            args.engineering_subset,
            load_unified_rows(args.manifest),
            qa_by_source,
            exact_canonical_inputs=report_for_subset["exact_canonical_inputs"],
            expected_file_sha256=str(frozen_subset["sha256"]),
        )
        selected = {str(row["qa_id"]) for row in engineering_artifact.get("selected_qa", ())}
        if not selected:
            raise ValueError("engineering subset has no selected QA")
        pairs = [(row, sample) for row, sample in zip(dataset.index_rows, dataset.samples)
                 if str(row["qa_id"]) in selected]
        if len(pairs) != len(selected):
            raise ValueError("engineering subset is not an exact subset of train")
        if args.required_frame_count is not None:
            pairs = [
                (row, _resource_profile_sample(sample, args.required_frame_count))
                for row, sample in pairs
                if sample.qa.get("media_kind") == "video"
                and len(sample.qa.get("actual_frame_indices", ())) >= args.required_frame_count
            ]
            if not pairs:
                raise ValueError("engineering subset lacks a video eligible for the profile frame count")
        dataset.index_rows = tuple(pair[0] for pair in pairs)
        dataset.samples = tuple(pair[1] for pair in pairs)
    max_steps = 2 if args.dry_run else args.max_steps
    config = PartATrainConfig(
        arm=args.arm,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lambda_state=args.lambda_state,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        max_steps=max_steps,
        dtype=args.dtype,
    )
    config.validate()
    resolved = {**asdict(config), "num_workers": args.num_workers, "split": "train"}
    resolved.update({
        "manifest": str(args.manifest.resolve()),
        "manifest_report": str(args.manifest_report.resolve()),
        "source_roots": {key: str(value.resolve()) for key, value in sorted(source_roots.items())},
        "media_root": str(args.media_root.resolve()),
        "model_path": str(args.model_path.resolve()),
        "vggt_path": str(args.vggt_path.resolve()),
        "distributed_strategy": args.distributed_strategy,
        "world_size": context.world_size,
        "cuda_total_memory_bytes": (
            int(torch.cuda.get_device_properties(torch.device(effective_device)).total_memory)
            if torch.cuda.is_available() and str(effective_device).startswith("cuda") else None
        ),
        "val_batches_per_source": args.val_batches_per_source,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "effective_device": effective_device,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_rank_batch_size": 1,
        "effective_global_batch_size": context.world_size * config.gradient_accumulation_steps,
        "required_frame_count": args.required_frame_count,
        "engineering_mode": args.engineering_mode,
        "dry_run": args.dry_run,
        "engineering_subset": (
            str(args.engineering_subset.resolve()) if args.engineering_subset else None
        ),
    })
    if context.is_primary:
        atomic_json_dump(resolved, args.output_dir / "resolved_config.json")

    # Reuse the already-reviewed reproduced-GUIDE loader until it is promoted
    # into a shared runtime module. Importing here keeps CPU contract tests light.
    from run_t0_a import _checkpoint_artifact_provenance, _load_local  # noqa: PLC0415
    from parta.t0_runtime import PartAT0Collator

    if args.engineering_mode == "resource_profile":
        normalized_profile = normalize_profile_worker_argv(sys.argv)
        preflight = {
            "schema_version": "parta_profile_preexecution_matched_v1",
            "status": "complete_preexecution",
            "distributed_strategy": args.distributed_strategy,
            "normalized_execution_contract": normalized_profile,
            "normalized_execution_contract_sha256": normalized_contract_sha256(normalized_profile),
            "manifest": {"path": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest)},
            "manifest_report": {"path": str(args.manifest_report.resolve()),
                                "sha256": sha256_file(args.manifest_report)},
            "engineering_subset": {"path": str(args.engineering_subset.resolve()),
                                   "sha256": sha256_file(args.engineering_subset)},
            "guide": _checkpoint_artifact_provenance(args.model_path),
            "vggt": _checkpoint_artifact_provenance(args.vggt_path),
        }
        if context.is_primary:
            atomic_json_dump(preflight, args.output_dir / "profile_preflight_matched_contract.json")
        barrier(context)

    device = torch.device(effective_device)
    dtype = getattr(torch, args.dtype)
    processor, model, _ = _load_local(args.model_path, args.vggt_path, dtype, effective_device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    shared_names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if args.arm == "a1o":
        attach_a1o_head_without_advancing_shared_rng(
            model,
            StateHeadConfig(
                hidden_size=int(model.config.text_config.hidden_size),
                num_categories=len(CANONICAL_CATEGORIES),
            ),
            seed=args.seed,
        ).to(device=device, dtype=dtype)
        install_a1o_forward_integration(model)
    if context.world_size > 1:
        if args.distributed_strategy == "ddp":
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[context.local_rank], output_device=context.local_rank
            )
        else:
            from torch.distributed.fsdp import FullyShardedDataParallel
            model = FullyShardedDataParallel(
                model, device_id=context.local_rank, use_orig_params=True
            )
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manifest_digest = file_sha256(args.manifest)
    report = json.loads(args.manifest_report.read_text(encoding="utf-8"))
    exact_canonical_inputs = report.get("exact_canonical_inputs")
    if not isinstance(exact_canonical_inputs, Mapping) or not exact_canonical_inputs:
        raise ValueError("manifest report lacks exact_canonical_inputs registry")
    if any(not isinstance(item, Mapping) for item in exact_canonical_inputs.values()):
        raise ValueError("exact_canonical_inputs entries must be source registry mappings")
    # Preserve the complete source->{files,...} registry.  Reducing each source
    # to legacy top-level size_bytes/sha256 fields both loses provenance and is
    # incompatible with the current unified-data report schema.
    exact_canonical_inputs = {
        source: dict(item) for source, item in sorted(exact_canonical_inputs.items())
    }
    exact_digest = stable_sha256(exact_canonical_inputs)
    initialization_digest = stable_sha256(
        {
            "guide": _checkpoint_artifact_provenance(args.model_path)["artifact_sha256"],
            "vggt": _checkpoint_artifact_provenance(args.vggt_path)["artifact_sha256"],
        }
    )
    artifact_identity = {
        "manifest_sha256": manifest_digest,
        "manifest_report_sha256": sha256_file(args.manifest_report),
        "exact_canonical_inputs": exact_canonical_inputs,
        "initialization_sha256": initialization_digest,
    }
    execution_contract = {
        "distributed_strategy": args.distributed_strategy,
        "world_size": context.world_size,
        "per_rank_batch_size": 1,
        "effective_global_batch_size": context.world_size * config.gradient_accumulation_steps,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "source_content_identity": artifact_identity["exact_canonical_inputs"],
        "manifest_sha256": manifest_digest,
        "manifest_report_sha256": artifact_identity["manifest_report_sha256"],
    }
    validate_single_step_execution_contract(
        cli_gradient_accumulation_steps=args.gradient_accumulation_steps,
        config=config,
        execution_contract=execution_contract,
        world_size=context.world_size,
    )
    resolved_identity = {"resolved_execution": resolved, "artifact_identity": artifact_identity}
    resolved_identity_sha256 = stable_sha256(resolved_identity)
    matched = matched_fairness_payload(
        config,
        manifest_sha256=manifest_digest,
        initialization_sha256=initialization_digest,
        exact_frame_binding_sha256=exact_digest,
        trainable_shared_parameter_names=shared_names,
        execution_contract=execution_contract,
    )
    matched_hash = stable_sha256(matched)
    if args.matched_contract.exists():
        frozen_matched = json.loads(args.matched_contract.read_text(encoding="utf-8"))
        matched_hash = assert_matched_fairness(frozen_matched, matched)
    elif context.is_primary:
        atomic_json_dump(matched, args.matched_contract)
    barrier(context)
    if not args.matched_contract.exists():
        raise RuntimeError("rank 0 did not publish matched contract")
    matched_hash = assert_matched_fairness(
        json.loads(args.matched_contract.read_text(encoding="utf-8")), matched
    )
    authorization = None
    if not engineering:
        freeze_payload = json.loads(args.frozen_config_artifact.read_text(encoding="utf-8"))
        formal_training_config = {
            key: value for key, value in resolved.items()
            if key not in {"arm", "manifest", "manifest_report", "model_path", "vggt_path"}
        }
        authorization = validate_formal_training_authorization(
            args.unified_gate,
            args.frozen_config_artifact,
            manifest_sha256=manifest_digest,
            manifest_report_sha256=artifact_identity["manifest_report_sha256"],
            matched_contract_sha256=matched_hash,
            profile_report_sha256=str(freeze_payload.get("profile_report_sha256", "")),
            guide_artifact_sha256=_checkpoint_artifact_provenance(args.model_path)["artifact_sha256"],
            vggt_artifact_sha256=_checkpoint_artifact_provenance(args.vggt_path)["artifact_sha256"],
            code_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            resolved_training_config={
                "artifact_path": freeze_payload.get("resolved_training_config_path"),
                "training_config": formal_training_config,
            },
            formal_arm=args.arm,
        )
    if context.is_primary:
        atomic_json_dump(matched, args.output_dir / "matched_fairness_contract.json")
        artifact_registry = {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": manifest_digest},
            "manifest_report": {"path": str(args.manifest_report.resolve()), "sha256": sha256_file(args.manifest_report)},
            "exact_canonical_inputs": exact_canonical_inputs,
            "guide": _checkpoint_artifact_provenance(args.model_path),
            "vggt": _checkpoint_artifact_provenance(args.vggt_path),
        }
        provenance = {
            "schema_version": "parta_training_provenance_v1", "status": "running",
            "runner": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256_file(Path(__file__).resolve())},
            "arm": args.arm, "rank_world_size": context.world_size,
            "resolved_config_sha256": resolved_identity_sha256, "matched_contract_sha256": matched_hash,
            "artifacts": artifact_registry,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip()),
            "formal_training_authorization": authorization,
            "transaction_lifecycle": {
                "transaction_kind": "engineering" if engineering else "formal",
                "engineering_mode": args.engineering_mode or ("dry_run" if args.dry_run else None),
                "promotable": not engineering,
                "model_promotable": not engineering,
                "optimizer_promotable": not engineering,
                "scheduler_promotable": not engineering,
                "rng_promotable": not engineering,
                "sampler_promotable": not engineering,
                "formal_start_step": 0 if not engineering and args.resume is None else None,
                "engineering_subset_extra_weight": False,
            },
            "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        }
        atomic_json_dump(provenance, args.output_dir / "run_status.json")

        def _failure_hook(error_type, error, tb):
            failed = {**provenance, "status": "failed", "error_type": error_type.__name__,
                      "error": str(error), "traceback": "".join(traceback.format_exception(error_type, error, tb))}
            atomic_json_dump(failed, args.output_dir / "run_status.json")
            sys.__excepthook__(error_type, error, tb)
        sys.excepthook = _failure_hook
    trainer = PartATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        forward_adapter=_forward_adapter,
        logger=JsonlTrainingLogger(args.output_dir / "train_steps.jsonl", enabled=context.is_primary),
        manifest_sha256=manifest_digest,
        resolved_config_sha256=resolved_identity_sha256,
        matched_contract_sha256=matched_hash,
        is_primary=context.is_primary,
        transaction_kind="engineering" if engineering else "formal",
        promotable=not engineering,
    )
    if args.resume is not None:
        trainer.resume(str(args.resume))

    media_loader = ExactMediaLoader(args.media_root)
    collator = PartAT0Collator(processor)
    cursor = SourceBalancedCursor(dataset.index_rows, seed=args.seed, epoch=trainer.epoch,
                                  position=trainer.sampler_position)
    validation_candidates = []
    engineering_qa_ids_seen = set()

    def save_and_validate_candidate() -> None:
        checkpoint = args.output_dir / f"checkpoint-{trainer.global_step}.pt"
        trainer.save(str(checkpoint))
        barrier(context)
        if val_dataset is None:
            return
        losses = _source_balanced_val_qa_loss(
            model, val_dataset, media_loader, collator, device, args.val_batches_per_source
        )
        barrier(context)
        validation_candidates.append({
            "step": trainer.global_step,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_val_qa_loss": losses,
        })
    LOGGER.info("Starting %s training at step %d", args.arm, trainer.global_step)
    while trainer.global_step < config.max_steps:
        index = cursor.next_distributed_index(rank=context.rank, world_size=context.world_size)
        sample = dataset[index]
        if engineering:
            engineering_qa_ids_seen.add(str(sample.qa["qa_id"]))
        local_error = None
        try:
            images = media_loader.load(sample)
            fixture = collator(sample, images)
        except Exception as error:
            LOGGER.exception("Corrupt sample at index=%d qa_id=%s", index, sample.qa.get("qa_id"))
            local_error = error
        _coordinate_step_failure(local_error, "load_collate", args, context)
        fixture = _fixture_on_device(fixture, device)
        target, _ = build_state_targets(sample)
        target = type(target)(**{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in vars(target).items()
        })
        local_error = None
        try:
            trainer.train_step(PartATrainBatch(
                model_inputs={
                    "fixture": fixture,
                    **({"a1o_side_request": {
                        "frame_token_counts": [fixture.frame_token_counts],
                        "frame_ids": [fixture.frame_ids],
                        "media_kinds": [fixture.media_kind],
                        "targets": [target],
                        "loss_config": trainer.loss_config,
                    }} if args.arm == "a1o" else {}),
                },
                targets=[target],
                source_datasets=[str(sample.qa["source_dataset"])],
                frame_ids=[fixture.frame_ids],
                frame_token_counts=[fixture.frame_token_counts],
                media_kinds=[fixture.media_kind],
                expected_frame_binding_sha256=[str(sample.qa["frame_binding_sha256"])],
            ))
        except Exception as error:
            LOGGER.exception("Training step failed at qa_id=%s", sample.qa.get("qa_id"))
            local_error = error
        _coordinate_step_failure(local_error, "train_step", args, context)
        trainer.epoch = cursor.epoch
        trainer.sampler_position = cursor.position
        if trainer.global_step % config.save_steps == 0:
            save_and_validate_candidate()
    if engineering:
        final_path = args.output_dir / "checkpoint-final.pt"
        trainer.save(str(final_path))
        selected_step = trainer.global_step
    else:
        if not validation_candidates or validation_candidates[-1]["step"] != trainer.global_step:
            save_and_validate_candidate()
        selection = select_validation_checkpoint(validation_candidates)
        selection.update({
            "status": "complete_selected", "arm": args.arm,
            "manifest_sha256": manifest_digest,
            "manifest_report_sha256": artifact_identity["manifest_report_sha256"],
            "frozen_config_artifact_sha256": sha256_file(args.frozen_config_artifact),
            "source_registry": ["adt", "hypersim", "scannetppv2"],
            "val_batches_per_source": args.val_batches_per_source,
        })
        selected_step = int(selection["selected"]["step"])
        selected_source = Path(next(
            item["checkpoint_path"] for item in validation_candidates if item["step"] == selected_step
        ))
        final_path = args.output_dir / "checkpoint-selected.pt"
        if context.is_primary:
            shutil.copy2(selected_source, final_path)
            selection["selected"]["checkpoint_path"] = str(final_path.resolve())
            selection["selected"]["checkpoint_sha256"] = sha256_file(final_path)
            atomic_json_dump(selection, args.output_dir / "checkpoint_selection.json")
        barrier(context)
    local_cuda_peak = (
        int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None
    )
    local_cuda_reserved_peak = (
        int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else None
    )
    local_cuda_total = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if torch.cuda.is_available() else None
    )
    per_rank_cuda_peak = [{
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "peak_allocated_bytes": local_cuda_peak,
        "peak_reserved_bytes": local_cuda_reserved_peak,
        "total_memory_bytes": local_cuda_total,
    }]
    if context.world_size > 1:
        gathered: list[dict | None] = [None] * context.world_size
        torch.distributed.all_gather_object(gathered, per_rank_cuda_peak[0])
        per_rank_cuda_peak = sorted(
            (item for item in gathered if item is not None), key=lambda item: item["rank"]
        )
    if context.is_primary and args.arm == "a1o":
        drop_path = args.output_dir / "checkpoint-a1o-drop.pt"
        export_head_free_checkpoint(final_path, drop_path)
        atomic_json_dump(
            {"schema_version": "parta_head_free_audit_request_v1", "status": "pending_independent_loader",
             "artifact": str(drop_path), "reason": "training/FSDP module must not be mutated for eval audit"},
            args.output_dir / "head_free_load_audit_request.json",
        )
    if context.is_primary:
        atomic_json_dump(
        {"schema_version": "parta_training_completion_v1", "status": "complete",
         "global_step": trainer.global_step, "selected_step": selected_step,
         "checkpoint_sha256": sha256_file(final_path),
         "checkpoint_path": str(final_path.resolve()),
         "checkpoint_role": "engineering_final" if engineering else "selected_validation"},
        args.output_dir / "completion.json",
        )
        atomic_json_dump({**provenance, "status": "complete", "global_step": trainer.global_step,
                          "checkpoint_sha256": sha256_file(final_path),
                          "checkpoint_path": str(final_path.resolve()),
                          "selected_step": selected_step,
                          "checkpoint_role": "engineering_final" if engineering else "selected_validation",
                          "checkpoint_selection_sha256": (
                              None if engineering else sha256_file(args.output_dir / "checkpoint_selection.json")
                          )}, args.output_dir / "run_status.json")
        if engineering:
            steps = [json.loads(line) for line in (args.output_dir / "train_steps.jsonl").read_text(
                encoding="utf-8"
            ).splitlines() if line]
            state_losses = [float(row["state_loss"]) for row in steps]
            receipt = {
                "schema_version": "parta_engineering_runner_receipt_v1",
                "status": "complete",
                "transaction_kind": "engineering",
                "engineering_mode": args.engineering_mode or "dry_run",
                "promotable": False,
                "arm": args.arm,
                "manifest_sha256": manifest_digest,
                "engineering_subset_sha256": sha256_file(args.engineering_subset),
                "optimizer_steps": trainer.global_step,
                "actual_unique_examples": len(engineering_qa_ids_seen),
                "overfit_defaults": {
                    "status": "provisional_D62_execution_default_v1",
                    "minimum_unique_examples": 64,
                    "optimizer_steps": 100,
                    "minimum_state_loss_decrease_fraction": 0.20,
                },
                "optimizer_step_indices": [int(row["step"]) for row in steps],
                "actual_frame_counts": sorted({int(row["actual_frame_count"]) for row in steps}),
                "frame_binding_sha256": sorted({
                    binding for row in steps for binding in row.get("frame_binding_sha256", ())
                }),
                "exact_canonical_inputs_registry_sha256": stable_sha256(exact_canonical_inputs),
                "training_log_sha256": sha256_file(args.output_dir / "train_steps.jsonl"),
                "all_losses_finite": all(torch.isfinite(torch.tensor([
                    row["qa_loss"], row["state_loss"], row["total_loss"]
                ])).all().item() for row in steps),
                "initial_state_loss": state_losses[0] if state_losses else None,
                "final_state_loss": state_losses[-1] if state_losses else None,
                "state_loss_decrease_fraction": (
                    (state_losses[0] - state_losses[-1]) / state_losses[0]
                    if state_losses and state_losses[0] > 0 else None
                ),
                "checkpoint_sha256": sha256_file(final_path),
                "cuda_total_memory_bytes": (
                    int(torch.cuda.get_device_properties(device).total_memory)
                    if torch.cuda.is_available() else None
                ),
                "per_rank_cuda_peak_memory_bytes": per_rank_cuda_peak,
                "resolved_execution_contract": execution_contract,
            }
            atomic_json_dump(receipt, args.output_dir / "engineering_receipt.json")
    barrier(context)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Every torchrun rank writes independent evidence, including failures
        # before the primary run_status/resolved_config artifacts exist.
        try:
            output_text = sys.argv[sys.argv.index("--output-dir") + 1]
            _write_top_level_rank_failure_if_absent(error, Path(output_text))
        except BaseException:
            pass
        raise
