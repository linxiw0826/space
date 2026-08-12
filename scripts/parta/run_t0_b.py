#!/usr/bin/env python3
"""Run the source-balanced Part A T0-B gradient calibration gate.

``--cpu-mock`` validates report semantics only and always records
``awaiting_gpu``.  A formal invocation consumes the frozen unified manifest
and the same authoritative GUIDE forward/collator used by ``train_parta.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts" / "parta"))

from parta.canonical_data import ExactMediaLoader, build_state_targets  # noqa: E402
from parta.checkpoint import (  # noqa: E402
    ResumeContract, capture_rng_state, load_training_checkpoint, save_training_checkpoint
)
from parta.provenance import atomic_json_dump, sha256_file, stable_sha256  # noqa: E402
from parta.runner import (  # noqa: E402
    PartATrainBatch,
    SharedForwardOutput,
    SourceBalancedCursor,
    attach_a1o_head_without_advancing_shared_rng,
    seed_matched_run,
)
from parta.state_head import StateHeadConfig  # noqa: E402
from parta.state_loss import StateLossConfig  # noqa: E402
from parta.t0_b_runtime import (  # noqa: E402
    T0BBatchObservation,
    T0BThresholds,
    active_target_components,
    build_t0_b_report,
    finalize_t0_b_report,
    nested_state_digest,
    parameter_gradient_norm,
    validate_t0_a_initialization_transaction,
)
from parta.training import run_a1o_side_branch  # noqa: E402
from parta.unified_data import PartAUnifiedDataset, file_sha256  # noqa: E402
from parta_data_contract import CANONICAL_CATEGORIES  # noqa: E402
from run_t0_a import _checkpoint_artifact_provenance, _load_local  # noqa: E402
from train_parta import _forward_adapter, _source_roots, _to_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-report", type=Path)
    parser.add_argument("--source-root", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--vggt-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--minimum-batches", type=int, default=20)
    parser.add_argument("--maximum-batches", type=int, default=50)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-gradient-norm", type=float, default=1e-12)
    parser.add_argument("--cpu-mock", action="store_true")
    parser.add_argument("--t0-a-report", type=Path)
    parser.add_argument("--t0-a-provenance", type=Path)
    parser.add_argument("--t0-a-run-status", type=Path)
    parser.add_argument("--t0-a-checkpoint", type=Path)
    return parser.parse_args()


def _required_paths(args: argparse.Namespace) -> None:
    missing = [
        name for name in (
            "manifest", "manifest_report", "media_root", "model_path", "vggt_path",
            "t0_a_report", "t0_a_provenance", "t0_a_run_status", "t0_a_checkpoint",
        )
        if getattr(args, name) is None
    ]
    if missing or not args.source_root:
        raise ValueError(f"formal T0-B missing inputs: {missing + ([] if args.source_root else ['source_root'])}")


def _mock_observations(count: int) -> list[T0BBatchObservation]:
    sources = ("adt", "hypersim", "scannetppv2")
    return [
        T0BBatchObservation(
            batch_index=index,
            qa_id=f"mock:{index}",
            source_dataset=sources[index % len(sources)],
            qa_loss=1.0,
            state_loss=1.0,
            qa_gradient_norm=1.0,
            state_gradient_norm=1.0,
            shared_gradient_parameter_count=1,
            head_gradient_parameter_count=1,
            enabled_components=("existence", "category"),
            masked_components=("center", "extent", "visibility"),
            component_losses={"existence": 0.5, "category": 0.5},
            component_valid_counts={"existence": 1, "category": 1, "center": 0, "extent": 0, "visibility": 0},
            matching_valid=True,
            matched_pairs=1,
            gt_objects=1,
            exact_frame_consistent=True,
            actual_frame_count=16,
        )
        for index in range(count)
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    thresholds = T0BThresholds(
        minimum_batches=args.minimum_batches,
        maximum_batches=args.maximum_batches,
        minimum_valid_fraction=args.minimum_valid_fraction,
        minimum_gradient_norm=args.minimum_gradient_norm,
    )
    thresholds.validate(args.batches)
    resolved = {
        "schema_version": "parta_t0_b_config_v1",
        "seed": args.seed,
        "batches": args.batches,
        "device": args.device,
        "dtype": args.dtype,
        "thresholds": asdict(thresholds),
        "defaults_status": thresholds.defaults_status,
        "cpu_mock": args.cpu_mock,
    }
    resolved_sha = stable_sha256(resolved)
    atomic_json_dump(resolved, args.output_dir / "resolved_config.json")
    def make_provenance(config_sha256: str) -> dict:
        return {
        "schema_version": "parta_t0_b_provenance_v1",
        "run_id": "t0-b-three-source",
        "status": "running",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT, text=True
        ).strip()),
        "resolved_config_sha256": config_sha256,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        }
    provenance = make_provenance(resolved_sha)
    atomic_json_dump(provenance, args.output_dir / "provenance.json")
    try:
        if args.cpu_mock:
            payload = build_t0_b_report(
                _mock_observations(args.batches), requested_batches=args.batches,
                thresholds=thresholds, checkpoint_resume_passed=True,
                runtime_status="awaiting_gpu", resolved_config_sha256=resolved_sha,
                expected_sources=("adt", "hypersim", "scannetppv2"),
            )
            finalize_t0_b_report(payload, str(args.output_dir / "t0_b_report.json"))
            terminal = {**provenance, "status": "awaiting_gpu",
                        "report_sha256": sha256_file(args.output_dir / "t0_b_report.json")}
            atomic_json_dump(terminal, args.output_dir / "provenance.json")
            atomic_json_dump(terminal, args.output_dir / "run_status.json")
            return

        _required_paths(args)
        if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
            raise RuntimeError("formal T0-B requires CUDA; use --cpu-mock for contract testing")
        source_roots = _source_roots(args.source_root)
        dataset = PartAUnifiedDataset(
            source_roots, args.manifest, split="train", report_path=args.manifest_report
        )
        report = json.loads(args.manifest_report.read_text(encoding="utf-8"))
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
        guide_artifact = _checkpoint_artifact_provenance(args.model_path)
        vggt_artifact = _checkpoint_artifact_provenance(args.vggt_path)
        t0_a_transaction = validate_t0_a_initialization_transaction(
            report_path=args.t0_a_report,
            provenance_path=args.t0_a_provenance,
            run_status_path=args.t0_a_run_status,
            checkpoint_path=args.t0_a_checkpoint,
            current_code_revision=git_revision,
            guide_artifact_sha256=guide_artifact["artifact_sha256"],
            vggt_artifact_sha256=vggt_artifact["artifact_sha256"],
            current_manifest_inputs=report["exact_canonical_inputs"],
        )
        resolved.update({
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "manifest_report_sha256": sha256_file(args.manifest_report),
            "exact_canonical_inputs": report["exact_canonical_inputs"],
            "exact_canonical_inputs_registry_sha256": stable_sha256(
                report["exact_canonical_inputs"]
            ),
            "source_roots": {key: str(value.resolve()) for key, value in sorted(source_roots.items())},
            "model_path": str(args.model_path.resolve()),
            "vggt_path": str(args.vggt_path.resolve()),
        })
        resolved_sha = stable_sha256(resolved)
        atomic_json_dump(resolved, args.output_dir / "resolved_config.json")
        provenance = make_provenance(resolved_sha)
        provenance["artifacts"] = {
            "manifest_sha256": file_sha256(args.manifest),
            "manifest_report_sha256": sha256_file(args.manifest_report),
            "guide": guide_artifact,
            "vggt": vggt_artifact,
            "t0_a_initialization_transaction": t0_a_transaction,
        }
        atomic_json_dump(provenance, args.output_dir / "provenance.json")
        seed_matched_run(args.seed)
        dtype = getattr(torch, args.dtype)
        device = torch.device(args.device)
        processor, model, _ = _load_local(args.model_path, args.vggt_path, dtype, args.device)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        attach_a1o_head_without_advancing_shared_rng(
            model,
            StateHeadConfig(
                hidden_size=int(model.config.text_config.hidden_size),
                num_categories=len(CANONICAL_CATEGORIES),
            ),
            seed=args.seed,
        ).to(device=device, dtype=dtype)
        smoke_payload = torch.load(args.t0_a_checkpoint, map_location="cpu", weights_only=True)
        if smoke_payload.get("optimizer_steps") != 0 or smoke_payload.get("checkpoint_role") != (
            "initialization_no_optimizer_updates"
        ):
            raise ValueError("T0-A checkpoint payload is not a zero-step initialization")
        incompatible = model.load_state_dict(smoke_payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("strict T0-A initialization restore failed")
        if nested_state_digest(model.state_dict()) != t0_a_transaction["t0_a_checkpoint_state_sha256"]:
            raise ValueError("loaded T0-A initialization state digest mismatch")
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-5
        )
        media_loader = ExactMediaLoader(args.media_root)
        from parta.t0_runtime import PartAT0Collator
        collator = PartAT0Collator(processor)
        cursor = SourceBalancedCursor(dataset.index_rows, seed=args.seed)
        observations = []
        loss_config = StateLossConfig()
        for batch_index in range(args.batches):
            sample = dataset[cursor.next_index()]
            images = media_loader.load(sample)
            fixture = collator(sample, images)
            fixture = type(fixture)(
                model_kwargs=_to_device(fixture.model_kwargs, device), sample=fixture.sample,
                images=fixture.images, frame_token_counts=fixture.frame_token_counts,
                frame_ids=fixture.frame_ids, media_kind=fixture.media_kind,
                visual_prefix_before_question=fixture.visual_prefix_before_question,
                visual_token_mask=fixture.visual_token_mask.to(device),
                question_token_span=fixture.question_token_span,
            )
            target, _ = build_state_targets(sample)
            target = type(target)(**{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in vars(target).items()
            })
            batch = PartATrainBatch(
                model_inputs={"fixture": fixture}, targets=[target],
                source_datasets=[str(sample.qa["source_dataset"])], frame_ids=[fixture.frame_ids],
                frame_token_counts=[fixture.frame_token_counts], media_kinds=[fixture.media_kind],
                expected_frame_binding_sha256=[str(sample.qa["frame_binding_sha256"])],
            )
            batch.validate()
            output: SharedForwardOutput = _forward_adapter(model, batch.model_inputs, True)
            branch = run_a1o_side_branch(
                model, output.visual_state_hidden, output.visual_state_valid_mask,
                batch.frame_token_counts, batch.frame_ids, batch.media_kinds, batch.targets, loss_config,
            )
            shared = [p for name, p in model.named_parameters() if p.requires_grad and "parta_state_head." not in name]
            head = [p for name, p in model.named_parameters() if p.requires_grad and "parta_state_head." in name]
            qa_norm, qa_count = parameter_gradient_norm(output.qa_loss, shared, retain_graph=True)
            state_norm, state_count = parameter_gradient_norm(branch.losses["loss_state"], shared, retain_graph=True)
            _, head_count = parameter_gradient_norm(branch.losses["loss_state"], head, retain_graph=False)
            enabled, masked = active_target_components(target)
            assignments = branch.losses["assignments"][0]
            slot_indices, gt_indices = assignments
            matched = len(slot_indices)
            matching_valid = (
                matched == target.num_objects
                and len(set(slot_indices.tolist())) == matched
                and len(set(gt_indices.tolist())) == matched
                and (matched == 0 or (
                    int(slot_indices.min()) >= 0 and int(slot_indices.max()) < 384
                    and int(gt_indices.min()) >= 0 and int(gt_indices.max()) < target.num_objects
                ))
                and torch.isfinite(branch.losses["matching_mean_cost"]).all().item()
            )
            exact = tuple(fixture.frame_ids) == tuple(sample.qa["actual_frame_indices"])
            observations.append(T0BBatchObservation(
                batch_index=batch_index, qa_id=str(sample.qa["qa_id"]),
                source_dataset=str(sample.qa["source_dataset"]),
                qa_loss=float(output.qa_loss.detach().float()),
                state_loss=float(branch.losses["loss_state"].detach().float()),
                qa_gradient_norm=qa_norm, state_gradient_norm=state_norm,
                shared_gradient_parameter_count=min(qa_count, state_count),
                head_gradient_parameter_count=head_count, enabled_components=enabled,
                masked_components=masked,
                component_losses={name: float(branch.losses[f"loss_{name}"].detach().float()) for name in ("existence", "category", "center", "extent", "visibility")},
                component_valid_counts={
                    "existence": 1,
                    "category": int(target.category_valid.sum()),
                    "center": int(target.center_valid.sum()),
                    "extent": int(target.extent_valid.sum()),
                    "visibility": int(target.visibility_valid.sum()),
                },
                matching_valid=matching_valid,
                matched_pairs=matched, gt_objects=target.num_objects,
                exact_frame_consistent=exact, actual_frame_count=len(fixture.frame_ids),
            ))
            model.zero_grad(set_to_none=True)

        manifest_sha = file_sha256(args.manifest)
        contract = ResumeContract("a1o", manifest_sha, resolved_sha, stable_sha256(resolved))
        probe = args.output_dir / "checkpoint-resume-probe.pt"
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        save_training_checkpoint(
            probe, model=model, optimizer=optimizer, scheduler=scheduler, global_step=0,
            epoch=cursor.epoch, sampler_position=cursor.position, contract=contract,
        )
        expected_identity = {
            "model": nested_state_digest(model.state_dict()),
            "optimizer": nested_state_digest(optimizer.state_dict()),
            "scheduler": nested_state_digest(scheduler.state_dict()),
            "counters": (0, cursor.epoch, cursor.position),
            "rng": nested_state_digest(capture_rng_state()),
        }
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        torch.rand(7)
        counters = load_training_checkpoint(
            probe, model=model, optimizer=optimizer, scheduler=scheduler, expected_contract=contract
        )
        restored_identity = {
            "model": nested_state_digest(model.state_dict()),
            "optimizer": nested_state_digest(optimizer.state_dict()),
            "scheduler": nested_state_digest(scheduler.state_dict()),
            "counters": (counters["global_step"], counters["epoch"], counters["sampler_position"]),
            "rng": nested_state_digest(capture_rng_state()),
        }
        checkpoint_ok = expected_identity == restored_identity
        payload = build_t0_b_report(
            observations, requested_batches=args.batches, thresholds=thresholds,
            checkpoint_resume_passed=checkpoint_ok, runtime_status="gpu_complete",
            resolved_config_sha256=resolved_sha,
            manifest_sha256=file_sha256(args.manifest),
            manifest_report_sha256=sha256_file(args.manifest_report),
            exact_registry_sha256=stable_sha256(report["exact_canonical_inputs"]),
            expected_sources=tuple(sorted(report["exact_canonical_inputs"])),
        )
        finalize_t0_b_report(payload, str(args.output_dir / "t0_b_report.json"))
        terminal = {**provenance, "status": "complete_passed",
                    "report_sha256": sha256_file(args.output_dir / "t0_b_report.json"),
                    "manifest_sha256": payload["manifest_sha256"],
                    "manifest_report_sha256": payload["manifest_report_sha256"],
                    "exact_registry_sha256": payload["exact_registry_sha256"]}
        atomic_json_dump(terminal, args.output_dir / "provenance.json")
        atomic_json_dump(terminal, args.output_dir / "run_status.json")
    except BaseException as error:
        failure = {
            **provenance, "status": "complete_failed", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        }
        report_path = args.output_dir / "t0_b_report.json"
        failure["report_sha256"] = sha256_file(report_path) if report_path.is_file() else None
        atomic_json_dump(failure, args.output_dir / "provenance.json")
        atomic_json_dump(failure, args.output_dir / "run_status.json")
        raise


if __name__ == "__main__":
    main()
