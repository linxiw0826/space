"""Isolated training entry point for E-00b-new, E-02c-new and E-03a-new."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model.mope_new_encoder import (
    DEFAULT_SOURCE_ROOT,
    MoPENewEncoder,
    load_annotation_for_mope_new,
    load_projector_warmstart,
)


EXPERIMENTS = {"e00b-new", "e02c-new", "e03a-new"}


def _take_new_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mope_new_experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--mope_new_source_root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--mope_new_sampling_rate", type=int, default=4)
    parser.add_argument("--mope_new_input_size", type=int, default=224)
    parser.add_argument("--mope_new_pool_mode", choices=("none", "temporal", "mean"), default="none")
    parser.add_argument("--mope_projector_warmstart_path", default="")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def validate_resume_scope(resume: str, output_dir: str) -> None:
    if not resume:
        return
    resume_path = Path(resume).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if output_path not in resume_path.parents:
        raise ValueError(
            f"MoPE-new resume checkpoint must be inside its own output directory: "
            f"resume={resume_path}, output={output_path}"
        )


def configure_trainability(model, experiment: str) -> dict[str, int]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown MoPE-new experiment: {experiment}")
    inner = model.model
    for parameter in inner._mope_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in inner._mope_projector.parameters():
        parameter.requires_grad_(True)
    if experiment == "e00b-new":
        for name, parameter in model.named_parameters():
            if "._mope_projector." not in name:
                parameter.requires_grad_(False)
    counts = {
        "encoder": sum(p.numel() for p in inner._mope_encoder.parameters() if p.requires_grad),
        "projector": sum(p.numel() for p in inner._mope_projector.parameters() if p.requires_grad),
        "other": sum(
            p.numel() for name, p in model.named_parameters()
            if p.requires_grad and "._mope_projector." not in name and "._mope_encoder." not in name
        ),
    }
    if counts["encoder"] != 0 or counts["projector"] == 0:
        raise RuntimeError(f"invalid MoPE-new trainability: {counts}")
    if experiment == "e00b-new" and counts["other"] != 0:
        raise RuntimeError(f"E-00b-new must train only its projector: {counts}")
    return counts


def _arg_value(argv: list[str], name: str, default: str = "") -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    new_args, remaining = _take_new_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    checkpoint = _arg_value(remaining, "--mope_checkpoint_path")
    if not checkpoint:
        raise ValueError("--mope_checkpoint_path is required for MoPE-new")
    if Path(checkpoint).name != "checkpoint-50.pth" and os.environ.get("MOPE_NEW_ALLOW_FAKE_CKPT") != "1":
        raise ValueError(f"MoPE-new requires checkpoint-50.pth, got {checkpoint}")
    if _arg_value(remaining, "--mope_all_frames", "16") != "16":
        raise ValueError("MoPE-new requires --mope_all_frames 16")
    output_dir = _arg_value(remaining, "--output_dir")
    resume = _arg_value(remaining, "--resume_from_checkpoint")
    validate_resume_scope(resume, output_dir)
    if new_args.mope_new_experiment == "e03a-new" and not new_args.mope_projector_warmstart_path:
        raise ValueError("E-03a-new requires --mope_projector_warmstart_path")
    if new_args.mope_new_experiment != "e03a-new" and new_args.mope_projector_warmstart_path:
        raise ValueError("only E-03a-new may warm-start a projector")

    import train_framework.train_space as base
    import train_framework.data.mope_data_wrapper as data_wrapper
    from model.mope_projector import MoPEProjectorCrossAttn
    original_set_model = base.set_model

    def attach(model, mope_args):
        if mope_args.mope_fusion_mode != "crossattn":
            raise ValueError("MoPE-new experiments require crossattn fusion")
        inner = model.model
        encoder = MoPENewEncoder(
            checkpoint, source_root=new_args.mope_new_source_root,
            num_frames=16, sampling_rate=new_args.mope_new_sampling_rate,
            input_size=new_args.mope_new_input_size, pool_mode=new_args.mope_new_pool_mode,
        )
        config = model.config
        llm_dim = getattr(config, "hidden_size", None) or config.text_config.hidden_size
        projector = MoPEProjectorCrossAttn(mope_dim=768, llm_dim=llm_dim)
        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)
        if new_args.mope_projector_warmstart_path:
            norms = load_projector_warmstart(projector, new_args.mope_projector_warmstart_path)
            base.rank0_print(
                f"[MoPE-new] projector warm-start={new_args.mope_projector_warmstart_path}; "
                f"keys={sorted(norms)}; out_proj.weight.norm={norms['out_proj.weight']:.6f}"
            )
        base.rank0_print(
            f"[MoPE-new] attached experiment={new_args.mope_new_experiment}, frames=16, "
            f"sampling_rate={new_args.mope_new_sampling_rate}, input=224, "
            f"pool={new_args.mope_new_pool_mode}, expected_tokens="
            f"{1568 if new_args.mope_new_pool_mode == 'none' else 8 if new_args.mope_new_pool_mode == 'temporal' else 1}"
        )

    def load_frames(annotation, all_frames):
        if all_frames != 16:
            raise ValueError(f"MoPE-new data wrapper expected 16 frames, got {all_frames}")
        return load_annotation_for_mope_new(
            annotation, num_frames=16, sampling_rate=new_args.mope_new_sampling_rate,
            input_size=new_args.mope_new_input_size,
        )

    def set_model_and_verify(model_args, model):
        original_set_model(model_args, model)
        counts = configure_trainability(model, new_args.mope_new_experiment)
        base.rank0_print(f"[MoPE-new] verified trainable parameter counts: {counts}")

    base._attach_mope_to_model = attach
    base.set_model = set_model_and_verify
    data_wrapper._load_mope_frames = load_frames
    base.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
