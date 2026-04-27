"""
Space Sensing Training Entry Point.

Forked from:
  refs/Let_Geometry_GUIDE/qwen-vl-finetune/qwenvl/train/train_qwen.py

Changes vs. upstream GUIDE:
  1. Imports MoPEArguments in addition to the base GUIDE argument dataclasses.
  2. After model loading (and optional LORA setup), attaches MoPEEncoder and
     MoPEProjector to the model when --use_mope is set.
  3. Patches model.model.forward() to inject MoPE embeddings into image/video
     tokens after the GUIDE geometry fusion step.

MoPE Fusion Design:
  - We wrap get_image_features (Qwen3VLModel.get_image_features) to inject a
    per-clip scene-prior bias before GUIDE geometry fusion.
  - mope_bias = mope_projector(mope_encoder(frames)).squeeze(1)  # [1, llm_dim]
  - image_embeds_list = [e + mope_bias for e in image_embeds_list]  # broadcast
  - Mathematically equivalent to post-geometry fusion (addition is commutative):
        (raw_visual + mope_bias) + geo == raw_visual + mope_bias + geo
  - MoPE contributes a per-clip bias that is uniform across all visual tokens,
    analogous to a clip-level scene prior.

Fusion insertion point (wraps get_image_features in model.model.forward):
  File: refs/.../qwenvl/model/modeling_qwen3_vl.py
  Method: Qwen3VLModel.forward()  (line ~1744 calls get_image_features)
  At: self.get_image_features(pixel_values, image_grid_thw)
  Before: torch.cat + _process_geometry_features

  The patch replaces self.get_image_features with a wrapper that calls the
  original, adds mope_bias to each element of image_embeds_list, and returns
  the modified (image_embeds_list, deepstack) unchanged in structure.

  Because modifying refs/ is forbidden, we use runtime forward-method
  monkey-patching after model creation (see src/model/mope_patch.py).
"""

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: ensure GUIDE qwenvl package is importable when this script is
# run from /u/lwu9/Space_sensing/projects/space/src/.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent

# qwenvl lives in src/qwenvl, added via _SRC_ROOT below

# Space Sensing src root (for src.model.mope_encoder / mope_projector)
_SRC_ROOT = _THIS_DIR.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# ---------------------------------------------------------------------------
# Imports from GUIDE (unchanged)
# ---------------------------------------------------------------------------
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: E402
import qwenvl.train.sampler  # noqa: F401, E402

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_processor import make_supervised_data_module
from transformers import AutoProcessor, Trainer, AutoConfig

# ---------------------------------------------------------------------------
# Imports from Space Sensing (new)
# ---------------------------------------------------------------------------
from src.train_framework.argument import (
    ModelArguments,
    MoPEArguments,
    DataArguments,
    TrainingArguments,
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


# ---------------------------------------------------------------------------
# Helpers (verbatim from GUIDE)
# ---------------------------------------------------------------------------

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


# ---------------------------------------------------------------------------
# MoPE integration helpers (new)
# ---------------------------------------------------------------------------

from model.mope_patch import _patch_model_for_mope  # noqa: E402


def _attach_mope_to_model(model, mope_args: MoPEArguments):
    """Instantiate MoPEEncoder + MoPEProjector and attach them to ``model``.

    Encoder is frozen; projector is trainable.

    We attach the modules to ``model.model`` (the inner
    Qwen3VLModel / Qwen2_5_VLModel) so that they are accessible inside the
    patched forward method as ``self._mope_encoder`` / ``self._mope_projector``.
    """
    from model.mope_encoder import MoPEEncoder
    from model.mope_projector import MoPEProjector

    inner_model = model.model  # Qwen3VLModel / Qwen2_5_VLModel

    rank0_print("[Space Sensing] Attaching MoPEEncoder (frozen) ...")
    encoder = MoPEEncoder(
        checkpoint_path=mope_args.mope_checkpoint_path,
        all_frames=mope_args.mope_all_frames,
    )
    # Register as sub-modules so they participate in .to(device), .parameters(), etc.
    inner_model.add_module("_mope_encoder", encoder)

    rank0_print("[Space Sensing] Attaching MoPEProjector (trainable) ...")
    # Qwen3VLConfig stores LLM hidden_size under text_config; Qwen2.5-VL puts it
    # directly on config.  Fall back gracefully so both model families work.
    _cfg = model.config
    _llm_hidden = getattr(_cfg, "hidden_size", None) or getattr(
        getattr(_cfg, "text_config", None), "hidden_size", None
    )
    if _llm_hidden is None:
        raise AttributeError(
            "Cannot determine LLM hidden_size from model.config. "
            "Set --mope_llm_dim explicitly."
        )
    llm_dim = _llm_hidden if mope_args.mope_llm_dim <= 0 else mope_args.mope_llm_dim
    rank0_print(
        f"[Space Sensing] MoPE llm_dim = {llm_dim} "
        f"({'auto from config' if mope_args.mope_llm_dim <= 0 else 'manual'})"
    )
    projector = MoPEProjector(
        mope_dim=768,
        llm_dim=llm_dim,
    )
    inner_model.add_module("_mope_projector", projector)

    # Ensure projector is trainable (encoder is already frozen internally).
    for p in projector.parameters():
        p.requires_grad = True

    rank0_print(
        f"[Space Sensing] MoPE attached: encoder frozen, "
        f"projector dim 768 -> {llm_dim} (trainable)"
    )



# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, MoPEArguments, DataArguments, TrainingArguments)
    )
    model_args, mope_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load config (verbatim from GUIDE)
    # ------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    custom_config_attrs = []

    if hasattr(data_args, "vgllm_resize"):
        setattr(config, "vgllm_resize", data_args.vgllm_resize)
        custom_config_attrs.append("vgllm_resize")
    if hasattr(data_args, "vgllm_resize_mode"):
        setattr(config, "vgllm_resize_mode", data_args.vgllm_resize_mode)
        custom_config_attrs.append("vgllm_resize_mode")
    if hasattr(data_args, "vgllm_target_size"):
        setattr(config, "vgllm_target_size", data_args.vgllm_target_size)
        custom_config_attrs.append("vgllm_target_size")
    if hasattr(data_args, "use_geometry_inputs"):
        setattr(config, "use_geometry_inputs", data_args.use_geometry_inputs)
        custom_config_attrs.append("use_geometry_inputs")
    if hasattr(data_args, "use_patch_size_alin"):
        setattr(config, "use_patch_size_alin", data_args.use_patch_size_alin)
        custom_config_attrs.append("use_patch_size_alin")

    if hasattr(model_args, "use_geometry_encoder"):
        setattr(config, "use_geometry_encoder", model_args.use_geometry_encoder)
        custom_config_attrs.append("use_geometry_encoder")
    if hasattr(model_args, "use_feature_fusion_module"):
        setattr(config, "use_feature_fusion_module", model_args.use_feature_fusion_module)
        custom_config_attrs.append("use_feature_fusion_module")
    if hasattr(model_args, "use_geometry_deepstack_only"):
        setattr(config, "use_geometry_deepstack_only", model_args.use_geometry_deepstack_only)
        custom_config_attrs.append("use_geometry_deepstack_only")
    if hasattr(model_args, "feature_fusion_method"):
        setattr(config, "feature_fusion_method", model_args.feature_fusion_method)
        custom_config_attrs.append("feature_fusion_method")
    if hasattr(model_args, "fusion_num_layers"):
        setattr(config, "fusion_num_layers", model_args.fusion_num_layers)
        custom_config_attrs.append("fusion_num_layers")
    if hasattr(model_args, "use_camera_method") and model_args.use_camera_method is not None:
        setattr(config, "use_camera_method", model_args.use_camera_method)
        custom_config_attrs.append("use_camera_method")
    if hasattr(model_args, "geometry_deepstack_indexes") and model_args.geometry_deepstack_indexes is not None:
        indexes = [int(x.strip()) for x in model_args.geometry_deepstack_indexes.split(",")]
        setattr(config, "geometry_deepstack_indexes", indexes)
        custom_config_attrs.append("geometry_deepstack_indexes")
    if hasattr(model_args, "geometry_deepstack_indexes_pro") and model_args.geometry_deepstack_indexes_pro is not None:
        geo_ds_entries = []
        for pair in model_args.geometry_deepstack_indexes_pro.split(","):
            pair = pair.strip()
            if ":" not in pair:
                raise ValueError(
                    f"geometry_deepstack_indexes_pro must use 'vggt:llm' format, got '{pair}'."
                )
            vggt_part, llm_part = pair.split(":")
            vggt_idx = int(vggt_part.strip())
            llm_layers = [int(x.strip()) for x in llm_part.split("-")]
            geo_ds_entries.append([vggt_idx, llm_layers])
        setattr(config, "geometry_deepstack_indexes_pro", geo_ds_entries)
        custom_config_attrs.append("geometry_deepstack_indexes_pro")
    if hasattr(model_args, "use_deepstack_importance_gate") and model_args.use_deepstack_importance_gate is not None:
        gate_val = model_args.use_deepstack_importance_gate.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_importance_gate", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_importance_gate", gate_layers)
        custom_config_attrs.append(f"use_deepstack_importance_gate={getattr(config, 'use_deepstack_importance_gate')}")
    if hasattr(model_args, "use_deepstack_global_gate") and model_args.use_deepstack_global_gate is not None:
        gate_val = model_args.use_deepstack_global_gate.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_global_gate", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_global_gate", gate_layers)
        custom_config_attrs.append(f"use_deepstack_global_gate={getattr(config, 'use_deepstack_global_gate')}")
    if hasattr(model_args, "use_deepstack_camera_adaln") and model_args.use_deepstack_camera_adaln is not None:
        gate_val = model_args.use_deepstack_camera_adaln.strip()
        if gate_val.lower() == "all":
            setattr(config, "use_deepstack_camera_adaln", "all")
        else:
            gate_layers = [int(x.strip()) for x in gate_val.split(",")]
            setattr(config, "use_deepstack_camera_adaln", gate_layers)
        custom_config_attrs.append(
            f"use_deepstack_camera_adaln={getattr(config, 'use_deepstack_camera_adaln')}"
        )

    if custom_config_attrs:
        rank0_print(f"Setting custom config attributes: {custom_config_attrs}")

    # ------------------------------------------------------------------
    # Load model (verbatim from GUIDE)
    # ------------------------------------------------------------------
    _model_type = getattr(config, 'model_type', '').lower()
    _path_lower = model_args.model_name_or_path.lower()
    _path_name  = Path(model_args.model_name_or_path.rstrip("/")).name.lower()

    _is_qwen3_moe = ('qwen3' in _model_type and 'moe' in _model_type) or \
                    ('qwen3' in _path_lower and 'a' in _path_name)
    _is_qwen3     = ('qwen3' in _model_type) or ('qwen3' in _path_lower)
    _is_qwen25    = ('qwen2.5' in _model_type or 'qwen2_5' in _model_type) or \
                    ('qwen2.5' in _path_lower)

    if _is_qwen3_moe:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif _is_qwen3:
        geometry_encoder_path = model_args.geometry_encoder_path if model_args.use_geometry_encoder else None
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            geometry_encoder_path=geometry_encoder_path,
        )
        data_args.model_type = "qwen3vl"
    elif _is_qwen25:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f"Initialized model: {model_args.model_name_or_path} ({model.__class__.__name__})")

    if data_args.model_type == "qwen3vl":
        from qwenvl.model.processing_qwen3_vl import Qwen3VLProcessor
        processor = Qwen3VLProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # ------------------------------------------------------------------
    # MoPE: attach encoder + projector BEFORE LoRA wrapping / param freeze
    # so that MoPEProjector parameters are visible when we set requires_grad.
    # ------------------------------------------------------------------
    if mope_args.use_mope:
        if not mope_args.mope_checkpoint_path:
            raise ValueError(
                "--use_mope is set but --mope_checkpoint_path is not provided."
            )

        # Override MOPE_ROOT in sys.path if an explicit encoder path was given.
        if mope_args.mope_encoder_path:
            mope_path = os.path.abspath(mope_args.mope_encoder_path)
            if mope_path not in sys.path:
                sys.path.insert(0, mope_path)
                rank0_print(f"[Space Sensing] Added mope_encoder_path to sys.path: {mope_path}")

        _attach_mope_to_model(model, mope_args)

    # ------------------------------------------------------------------
    # Trainability setup (verbatim from GUIDE, with MoPE projector unfreeze)
    # ------------------------------------------------------------------
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Re-enable MoPEProjector after LoRA wrapping froze everything.
        if mope_args.use_mope:
            projector = model.model._mope_projector
            for p in projector.parameters():
                p.requires_grad = True
            rank0_print("[Space Sensing] MoPEProjector re-enabled after LoRA freeze.")
    else:
        set_model(model_args, model)

        # MoPEEncoder is frozen internally; only MoPEProjector is trainable.
        if mope_args.use_mope:
            # Encoder: ensure all params are frozen.
            for p in model.model._mope_encoder.parameters():
                p.requires_grad = False
            # Projector: ensure all params are trainable.
            for p in model.model._mope_projector.parameters():
                p.requires_grad = True
            rank0_print("[Space Sensing] MoPEEncoder frozen, MoPEProjector trainable.")

        is_rank0 = (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_rank() == 0
        )
        if is_rank0:
            if hasattr(model, "visual") and hasattr(model.visual, "print_trainable_parameters"):
                model.visual.print_trainable_parameters()
            lm = getattr(model, "language_model", None) or getattr(model, "model", None)
            if lm is not None and hasattr(lm, "print_trainable_parameters"):
                lm.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Patch forward to inject MoPE (after parameter freeze to avoid
    # accidentally making patch-internal references trainable).
    # ------------------------------------------------------------------
    if mope_args.use_mope:
        _patch_model_for_mope(model)

    # ------------------------------------------------------------------
    # Data + training (verbatim from GUIDE)
    # ------------------------------------------------------------------
    data_module = make_supervised_data_module(processor, data_args=data_args)

    if mope_args.use_mope:
        from train_framework.data.mope_data_wrapper import (
            MoPEDatasetWrapper,
            MoPECollatorWrapper,
        )
        data_module["train_dataset"] = MoPEDatasetWrapper(
            data_module["train_dataset"],
            mope_all_frames=mope_args.mope_all_frames,
        )
        data_module["data_collator"] = MoPECollatorWrapper(data_module["data_collator"])
        rank0_print(
            f"[Space Sensing] MoPE data pipeline: {mope_args.mope_all_frames} frames/sample"
        )
        training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
