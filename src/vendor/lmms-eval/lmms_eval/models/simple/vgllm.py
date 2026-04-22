import base64
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

# Add qwen-vl-finetune to path to import local model classes
project_root = Path(__file__).parent.parent.parent.parent.parent
qwen_vl_finetune_path = project_root / "qwen-vl-finetune"
if str(qwen_vl_finetune_path) not in sys.path:
    sys.path.insert(0, str(qwen_vl_finetune_path))

try:
    from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from qwenvl.model.processing_qwen3_vl import Qwen3VLProcessor
    # Import _vgllm_resize_images from data_processor to ensure consistency with training
    from qwenvl.data.data_processor import _vgllm_resize_images
except ImportError as e:
    eval_logger.error(f"Failed to import local Qwen3VL model classes: {e}")
    eval_logger.error(f"Please ensure qwen-vl-finetune is in the path: {qwen_vl_finetune_path}")
    raise

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("vgllm")
class VGLLM(lmms):
    """
    VGLLM Model - Local modified version of Qwen3-VL
    Uses custom model classes from qwen-vl-finetune/qwenvl/model/
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Load config first to read VG-LLM specific settings (aligned with VG-LLM implementation)
        config = AutoConfig.from_pretrained(pretrained)
        
        # Read VG-LLM specific parameters from config (with fallback to defaults)
        # These can be set in model's config.json or passed via model_args
        self.vgllm_resize = getattr(config, "vgllm_resize", False)
        self.vgllm_resize_mode = getattr(config, "vgllm_resize_mode", "crop")
        self.vgllm_target_size = getattr(config, "vgllm_target_size", 512)
        self.use_geometry_inputs = getattr(config, "use_geometry_inputs", False) or getattr(config, "use_geometry_encoder", False)
        
        # Allow override from kwargs (model_args) if provided
        if "vgllm_resize" in kwargs:
            self.vgllm_resize = kwargs.pop("vgllm_resize")
        if "vgllm_resize_mode" in kwargs:
            self.vgllm_resize_mode = kwargs.pop("vgllm_resize_mode")
        if "vgllm_target_size" in kwargs:
            self.vgllm_target_size = kwargs.pop("vgllm_target_size")
        if "use_geometry_inputs" in kwargs:
            self.use_geometry_inputs = kwargs.pop("use_geometry_inputs")
        
        # Validate vgllm_resize parameters
        if self.vgllm_resize and self.vgllm_resize_mode not in ("crop", "pad"):
            raise ValueError(f"vgllm_resize_mode must be either 'crop' or 'pad', got {self.vgllm_resize_mode}")
        
        # Geometry inputs require vgllm_resize to be enabled
        if self.use_geometry_inputs and not self.vgllm_resize:
            eval_logger.warning("use_geometry_inputs requires vgllm_resize=True, enabling vgllm_resize")
            self.vgllm_resize = True

        # Use local Qwen3VLForConditionalGeneration class
        # Note: MoE models are not currently supported in local version
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained, config=config, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        
        # Use local Qwen3VLProcessor class
        self.processor = Qwen3VLProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for VGLLM")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            batch_geometry_inputs = []  # Store geometry inputs for the entire batch
            
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                sample_geometry_inputs = []  # Store geometry inputs for this sample
                
                if visual_list[i] is not None:
                    # Collect images for VG-LLM resize if enabled
                    images_to_resize = []
                    image_indices = []
                    
                    for idx, visual in enumerate(visual_list[i]):
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            if self.vgllm_resize:
                                # Collect images for batch resize
                                images_to_resize.append(visual)
                                image_indices.append(len(processed_visuals))
                                processed_visuals.append({"type": "image", "image": None, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})  # Placeholder
                            else:
                                processed_visuals.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, str):  # Image file path
                            if self.vgllm_resize:
                                # Collect image paths for batch resize
                                images_to_resize.append(visual)
                                image_indices.append(len(processed_visuals))
                                processed_visuals.append({"type": "image", "image": None, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})  # Placeholder
                            else:
                                processed_visuals.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    
                    # Apply VG-LLM resize if enabled
                    if self.vgllm_resize and len(images_to_resize) > 0:
                        try:
                            # Resize images using VG-LLM logic
                            resized_tensors = _vgllm_resize_images(
                                images_to_resize,
                                mode=self.vgllm_resize_mode,
                                target_size=self.vgllm_target_size
                            )
                            
                            # Save geometry_encoder_inputs if requested
                            if self.use_geometry_inputs:
                                sample_geometry_inputs = [t.clone() for t in resized_tensors]
                            
                            # Convert tensors back to PIL Images for processor
                            from torchvision.transforms import ToPILImage
                            to_pil = ToPILImage()
                            resized_pils = [to_pil(t) for t in resized_tensors]
                            
                            # Replace placeholders with resized PIL Images
                            for pil_idx, orig_idx in enumerate(image_indices):
                                if pil_idx < len(resized_pils):
                                    processed_visuals[orig_idx]["image"] = resized_pils[pil_idx]
                        except Exception as e:
                            eval_logger.warning(f"Failed to apply VG-LLM resize for sample {i}: {e}, falling back to default processing")
                            # Fallback: use original images
                            for idx, orig_idx in enumerate(image_indices):
                                if idx < len(images_to_resize):
                                    img = images_to_resize[idx]
                                    if isinstance(img, str):
                                        img = Image.open(img).convert("RGB")
                                    processed_visuals[orig_idx]["image"] = img

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
                # Store geometry inputs for this sample
                if self.use_geometry_inputs and len(sample_geometry_inputs) > 0:
                    batch_geometry_inputs.append(sample_geometry_inputs)
                else:
                    batch_geometry_inputs.append(None)
            
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            # TODO: refactor code to allow return_video_kwargs and return_video_metadata
            image_inputs, video_inputs = process_vision_info(batched_messages, return_video_kwargs=False, image_patch_size=16, return_video_metadata=False)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            # Prepare images_kwargs for processor (aligned with data_processor.py)
            images_kwargs = {}
            if self.vgllm_resize:
                # Skip resize/rescale in image_processor since we already resized
                images_kwargs["do_resize"] = False
                images_kwargs["do_rescale"] = False
            
            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts, 
                    images=image_inputs, 
                    videos=video_inputs, 
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                    do_resize=images_kwargs.get("do_resize", False), 
                    do_rescale=images_kwargs.get("do_rescale", True),
                    padding=True, 
                    padding_side="left", 
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    text=texts, 
                    images=image_inputs, 
                    videos=video_inputs, 
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                    do_resize=images_kwargs.get("do_resize", True), 
                    do_rescale=images_kwargs.get("do_rescale", True),
                    return_tensors="pt"
                )
            
            # Add geometry_encoder_inputs to inputs if available (for model forward pass)
            # Note: geometry_encoder_inputs should be per-sample, matching the batch structure
            if self.use_geometry_inputs and any(ge is not None for ge in batch_geometry_inputs):
                # For now, we'll pass the first sample's geometry inputs if available
                # In a full implementation, this should handle batched geometry inputs properly
                if batch_geometry_inputs[0] is not None and len(batch_geometry_inputs[0]) > 0:
                    try:
                        # Stack geometry inputs for the first sample if they have the same shape
                        ge_inputs = batch_geometry_inputs[0]
                        shapes = [t.shape for t in ge_inputs]
                        if len(shapes) > 0 and all(s == shapes[0] for s in shapes):
                            inputs["geometry_encoder_inputs"] = torch.stack(ge_inputs)
                        else:
                            inputs["geometry_encoder_inputs"] = ge_inputs
                    except Exception as e:
                        eval_logger.warning(f"Failed to process geometry_encoder_inputs: {e}")
                        # Don't add geometry inputs if processing fails
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                # eval_logger.debug(f"Question: {context}")
                # eval_logger.debug(f"Model Raw Response: {ans}")
                # eval_logger.debug(f"Model Clean Response: {clean_ans}")
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")

