#!/usr/bin/env python3
"""Run the ordered Part A GPU gates or emit a non-promotable CPU/plan artifact."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.gate_orchestration import (  # noqa: E402
    FORMAL_SOURCE_REGISTRY,
    PHASES,
    PhaseCommand,
    ProvisionalGateDefaults,
    awaiting_gpu_plan,
    run_phase,
    unified_gate_report,
)
from parta.provenance import atomic_json_dump, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    for phase in PHASES:
        parser.add_argument(f"--{phase.replace('_', '-')}-command-json", type=Path)
    parser.add_argument("--overfit-examples", type=int, default=64)
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument("--overfit-minimum-loss-decrease-fraction", type=float, default=0.20)
    parser.add_argument("--profile-frame-counts", type=int, nargs="+", default=(32,))
    parser.add_argument("--frozen-config-artifact", type=Path)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.finalize_only:
        raise FileExistsError(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.finalize_only)
    try:
        defaults = ProvisionalGateDefaults(
            overfit_examples=args.overfit_examples,
            overfit_steps=args.overfit_steps,
            overfit_minimum_loss_decrease_fraction=args.overfit_minimum_loss_decrease_fraction,
            profile_frame_counts=tuple(args.profile_frame_counts),
        )
        defaults.validate()
        if args.finalize_only:
            if not args.frozen_config_artifact:
                raise ValueError("--finalize-only requires --frozen-config-artifact")
            preauth = args.output_dir / "unified_gate_pre_authorization.json"
            if not preauth.is_file():
                legacy = args.output_dir / "unified_gate_report.json"
                if not legacy.is_file():
                    raise FileNotFoundError("missing pre-authorization unified gate report")
                legacy_payload = json.loads(legacy.read_text(encoding="utf-8"))
                if legacy_payload.get("training_authorized_by_this_artifact"):
                    # Already finalized before the preserved-preauth convention: idempotent success.
                    atomic_json_dump(legacy_payload, args.output_dir / "run_status.json")
                    return
                atomic_json_dump(legacy_payload, preauth)
            prior = json.loads(preauth.read_text(encoding="utf-8"))
            if prior.get("status") != "complete_passed" or prior.get(
                "training_authorized_by_this_artifact"
            ):
                raise ValueError("finalize requires the original passed, unauthorized gate")
            statuses = [
                json.loads((args.output_dir / phase / "status.json").read_text(encoding="utf-8"))
                for phase in PHASES
            ]
            frozen = json.loads(args.frozen_config_artifact.read_text(encoding="utf-8"))
            gate = unified_gate_report(
                statuses, defaults, frozen_config_artifact=frozen,
                prior_unified_gate_sha256=sha256_file(preauth),
            )
            if not gate["training_authorized_by_this_artifact"]:
                raise ValueError("freeze artifact does not bind the original GPU gate evidence")
            gate["finalized_from_pre_authorization_sha256"] = sha256_file(preauth)
            gate["freeze_artifact_sha256"] = sha256_file(args.frozen_config_artifact)
            atomic_json_dump(gate, args.output_dir / "unified_gate_report.json")
            atomic_json_dump(gate, args.output_dir / "run_status.json")
            return
        commands = {}
        for phase in PHASES:
            path = getattr(args, f"{phase}_command_json")
            commands[phase] = PhaseCommand.from_json(path, phase) if path else None
        plan = awaiting_gpu_plan(args.output_dir, defaults, commands)
        present_registries = [
            set(command.resolved_config.get("source_registry", ()))
            for command in commands.values() if command is not None
        ]
        required_registry = set(FORMAL_SOURCE_REGISTRY)
        data_ready = bool(present_registries) and all(
            registry == required_registry for registry in present_registries
        )
        if present_registries and not data_ready:
            plan["status"] = "awaiting_data"
            plan["formal_gpu_evidence"] = False
            plan["required_source_registry"] = list(FORMAL_SOURCE_REGISTRY)
            plan["observed_source_registries"] = [sorted(value) for value in present_registries]
            plan["source_registry_failure"] = "missing_or_extra_source"
        atomic_json_dump(plan, args.output_dir / "resolved_plan.json")
        if args.plan_only:
            atomic_json_dump(plan, args.output_dir / "run_status.json")
            print(json.dumps(plan, indent=2))
            return
        missing = [phase for phase, command in commands.items() if command is None]
        if missing:
            raise ValueError(f"execute mode requires command JSON for every phase: {missing}")
        if not data_ready:
            atomic_json_dump(plan, args.output_dir / "run_status.json")
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU gate execution requires CUDA; use --plan-only on CPU")
        statuses = []
        for phase in PHASES:
            status = run_phase(commands[phase], phase_dir=args.output_dir / phase, defaults=defaults)
            statuses.append(status)
            if status["status"] != "complete_passed":
                break
        frozen = (
            json.loads(args.frozen_config_artifact.read_text(encoding="utf-8"))
            if args.frozen_config_artifact else None
        )
        gate = unified_gate_report(statuses, defaults, frozen_config_artifact=frozen)
        atomic_json_dump(gate, args.output_dir / "unified_gate_report.json")
        if gate["status"] == "complete_passed" and not gate["training_authorized_by_this_artifact"]:
            atomic_json_dump(gate, args.output_dir / "unified_gate_pre_authorization.json")
        atomic_json_dump(gate, args.output_dir / "run_status.json")
        if gate["status"] != "complete_passed":
            raise SystemExit(1)
    except BaseException as error:
        status_path = args.output_dir / "run_status.json"
        if not status_path.exists():
            atomic_json_dump({
                "schema_version": "parta_unified_pretrain_gate_v1",
                "status": "complete_failed",
                "formal_gpu_evidence": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }, status_path)
        raise


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # argparse failures happen before main has an Namespace. Recover an explicit
        # output path when present so even CLI construction errors get a terminal artifact.
        if "--output-dir" in sys.argv:
            index = sys.argv.index("--output-dir")
            if index + 1 < len(sys.argv):
                output = Path(sys.argv[index + 1])
                output.mkdir(parents=True, exist_ok=True)
                status_path = output / "run_status.json"
                if not status_path.exists():
                    atomic_json_dump({
                        "schema_version": "parta_unified_pretrain_gate_v1",
                        "status": "complete_failed", "formal_gpu_evidence": False,
                        "error_type": type(error).__name__, "error": str(error),
                    }, status_path)
        raise
