# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class
# Import sampler to register custom _get_train_sampler
import qwenvl.train.sampler  # noqa: F401

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer, AutoConfig

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load config first (VG-LLM style: set custom attributes before loading model)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    # Set custom VG-LLM specific attributes to config
    # These attributes will be saved in config.json and can be loaded during evaluation
    custom_config_attrs = []
    
    # VG-LLM image preprocessing settings (from data_args)
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
    
    # Geometry encoder settings (from model_args)
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
                    f"geometry_deepstack_indexes_pro must use 'vggt:llm' format, got '{pair}'. "
                    f"Example: '7:0-1-2-3,14:4-5-6-7' or '7:1,7:2'"
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

    # Load model with the modified config
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        if model_args.use_geometry_encoder:
            geometry_encoder_path = model_args.geometry_encoder_path
        else:
            geometry_encoder_path = None
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
            geometry_encoder_path=geometry_encoder_path,
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
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

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    # Prefer local processor implementation for local development (so edits in qwenvl/ take effect).
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

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        is_rank0 = (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if is_rank0:
            if hasattr(model, "visual") and hasattr(model.visual, "print_trainable_parameters"):
                model.visual.print_trainable_parameters()
            lm = getattr(model, "language_model", None) or getattr(model, "model", None)
            if lm is not None and hasattr(lm, "print_trainable_parameters"):
                lm.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
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
