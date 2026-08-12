#!/usr/bin/env python3
"""Load A1-O-drop into an independently constructed head-free GUIDE model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from parta.checkpoint import load_head_free_artifact  # noqa: E402
from parta.provenance import atomic_json_dump, sha256_file  # noqa: E402
from parta.canonical_data import ExactMediaLoader  # noqa: E402
from parta.unified_data import PartAUnifiedDataset  # noqa: E402


def producer_record(git_revision: str) -> dict[str, str]:
    producer = Path(__file__).resolve()
    return {"path": str(producer), "sha256": sha256_file(producer),
            "git_revision": git_revision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vggt-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--source-root", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--media-root", type=Path, required=True)
    args = parser.parse_args()

    from run_t0_a import _checkpoint_artifact_provenance, _load_local
    from train_parta import _source_roots, _to_device
    from parta.t0_runtime import PartAT0Collator, forward_visual_tap

    processor, model, _ = _load_local(
        args.model_path, args.vggt_path, getattr(torch, args.dtype), args.device
    )
    if hasattr(model, "parta_state_head"):
        raise RuntimeError("independent GUIDE loader unexpectedly instantiated state head")
    _, report = load_head_free_artifact(model, args.artifact)
    dataset = PartAUnifiedDataset(
        _source_roots(args.source_root), args.manifest, split="val",
        report_path=args.manifest_report,
    )
    sample = dataset[0]
    images = ExactMediaLoader(args.media_root).load(sample)
    fixture = PartAT0Collator(processor)(sample, images)
    device = torch.device(args.device)
    fixture = type(fixture)(
        model_kwargs=_to_device(fixture.model_kwargs, device), sample=fixture.sample,
        images=fixture.images, frame_token_counts=fixture.frame_token_counts,
        frame_ids=fixture.frame_ids, media_kind=fixture.media_kind,
        visual_prefix_before_question=fixture.visual_prefix_before_question,
        visual_token_mask=fixture.visual_token_mask.to(device),
        question_token_span=fixture.question_token_span,
    )
    model.eval()
    with torch.no_grad():
        output = forward_visual_tap(model, fixture)
    if output.loss is not None and not bool(torch.isfinite(output.loss)):
        raise FloatingPointError("head-free GUIDE forward produced non-finite loss")
    report["status"] = "complete_passed"
    report["producer"] = producer_record(subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip())
    report["independent_model_construction"] = True
    report["forward_passed"] = True
    report["fixture_split"] = "val"
    report["frame_binding_sha256"] = sample.qa["frame_binding_sha256"]
    report["head_free_artifact_sha256"] = sha256_file(args.artifact)
    report["guide_artifact"] = _checkpoint_artifact_provenance(args.model_path)
    report["vggt_artifact"] = _checkpoint_artifact_provenance(args.vggt_path)
    atomic_json_dump(report, args.report)


if __name__ == "__main__":
    main()
