"""Training entry point for the final515k E-02c-new Paper 1 experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from model.mope_new_encoder import (
    DEFAULT_SOURCE_ROOT,
    MoPENewEncoder,
    load_annotation_for_mope_new,
)


EXPERIMENTS = {"e02c-new"}


def _take_new_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mope_new_experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--mope_new_source_root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--mope_new_groups", type=int, default=4)
    parser.add_argument("--mope_new_frames_per_group", type=int, default=4)
    parser.add_argument("--mope_new_input_size", type=int, default=224)
    parser.add_argument("--mope_new_pool_mode", choices=("temporal",), default="temporal")
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
    if experiment != "e02c-new":
        raise ValueError(f"unknown MoPE-new experiment: {experiment}")
    inner = model.model
    for parameter in inner._mope_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in inner._mope_projector.parameters():
        parameter.requires_grad_(True)
    counts = {
        "encoder": sum(p.numel() for p in inner._mope_encoder.parameters() if p.requires_grad),
        "projector": sum(p.numel() for p in inner._mope_projector.parameters() if p.requires_grad),
        "other": sum(
            p.numel() for name, p in model.named_parameters()
            if p.requires_grad and "._mope_projector." not in name and "._mope_encoder." not in name
        ),
    }
    if counts["encoder"] != 0 or counts["projector"] == 0 or counts["other"] == 0:
        raise RuntimeError(f"invalid MoPE-new trainability: {counts}")
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
    if (new_args.mope_new_groups, new_args.mope_new_frames_per_group) != (4, 4):
        raise ValueError("final515k requires --mope_new_groups 4 --mope_new_frames_per_group 4")

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
            num_frames=16, groups=new_args.mope_new_groups,
            frames_per_group=new_args.mope_new_frames_per_group,
            input_size=new_args.mope_new_input_size, pool_mode=new_args.mope_new_pool_mode,
        )
        config = model.config
        llm_dim = getattr(config, "hidden_size", None) or config.text_config.hidden_size
        projector = MoPEProjectorCrossAttn(mope_dim=768, llm_dim=llm_dim)
        inner.add_module("_mope_encoder", encoder)
        inner.add_module("_mope_projector", projector)
        base.rank0_print(
            f"[MoPE-final515k] attached experiment={new_args.mope_new_experiment}, "
            f"frames=16, sampling=4x4_uniform_segments_rint, input=224, "
            f"position=3d_sincos, pool=temporal, expected_tokens=8"
        )

    def load_frames(annotation, all_frames):
        if all_frames != 16:
            raise ValueError(f"MoPE-new data wrapper expected 16 frames, got {all_frames}")
        if not annotation.get("mope_video"):
            raise ValueError(
                "final515k E-02c-new requires SPAR annotations with mope_video so "
                "MoPE samples 4x4 from the complete video; regenerate/migrate the "
                "VSI-590K SPAR manifest instead of resampling its eight GUIDE images"
            )
        return load_annotation_for_mope_new(
            annotation, groups=new_args.mope_new_groups,
            frames_per_group=new_args.mope_new_frames_per_group,
            input_size=new_args.mope_new_input_size,
        )

    def set_model_and_verify(model_args, model):
        original_set_model(model_args, model)
        counts = configure_trainability(model, new_args.mope_new_experiment)
        base.rank0_print(f"[MoPE-new] verified trainable parameter counts: {counts}")

    base._attach_mope_to_model = attach
    base.set_model = set_model_and_verify
    data_wrapper._load_mope_frames = load_frames
    data_wrapper._STRICT_MOPE_LOADING = True
    base.train(attn_implementation="flash_attention_2")


if __name__ == "__main__":
    main()
