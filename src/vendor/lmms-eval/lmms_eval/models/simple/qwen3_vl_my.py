import base64
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from transformers.image_utils import SizeDict
from torchvision import transforms as TF

repo_root = Path(__file__).resolve().parents[4]
qwen_vl_finetune = repo_root / "qwen-vl-finetune"
if qwen_vl_finetune.exists():
    sys.path.insert(0, str(qwen_vl_finetune))
from qwenvl.model.modeling_qwen3_vl import (  # noqa: E402  # type: ignore[import-not-found]
    Qwen3VLForConditionalGeneration,
)
try:  # Optional: local repo may not provide MoE class
    from qwenvl.model.modeling_qwen3_vl import (  # noqa: E402  # type: ignore[import-not-found]
        Qwen3VLMoeForConditionalGeneration,
    )
except Exception:  # pragma: no cover - fallback handled at runtime
    Qwen3VLMoeForConditionalGeneration = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_my")
class Qwen3_VL_MY(lmms):
    """
    Qwen3_VL Model
    "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct"
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
        add_frame_index: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.add_frame_index = add_frame_index
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

        # check whether its an MoE model
        match = re.search(r"A\d+B", pretrained)
        if match:
            if Qwen3VLMoeForConditionalGeneration is None:
                raise ImportError(
                    "检测到 MoE 权重名称，但本地 modeling_qwen3_vl.py 未提供 "
                    "Qwen3VLMoeForConditionalGeneration。请改用非 MoE 权重，"
                    "或补齐本地 MoE 模型实现。"
                )
            model_fn = Qwen3VLMoeForConditionalGeneration
        else:
            model_fn = Qwen3VLForConditionalGeneration
        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

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
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                image_count = 0
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            image_num = len(vr)
                            if image_num <= self.max_num_frames:
                                frame_indices = np.arange(image_num)
                            else:
                                frame_indices = np.linspace(0, image_num - 1, self.max_num_frames).astype(int)
                            for idx in frame_indices:
                                frame = Image.fromarray(vr[int(idx)].asnumpy()).convert("RGB")
                                if self.add_frame_index:
                                    processed_visuals.append({"type": "text", "text": f"Frame-{image_count}: "})
                                processed_visuals.append({"type": "image", "image": frame, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                                image_count += 1
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            if self.add_frame_index:
                                processed_visuals.append({"type": "text", "text": f"Frame-{image_count}: "})
                            processed_visuals.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                            image_count += 1

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
            use_geometry_inputs = getattr(self.model.config, "use_geometry_encoder", False)
            use_patch_size_alin = getattr(self.model.config, "use_patch_size_alin", False)
            geometry_encoder_inputs = None
            if use_geometry_inputs:
                if TF is None:
                    raise RuntimeError("torchvision is required for geometry_encoder_inputs conversion.")
                geometry_encoder_inputs = []
                size_cfg = getattr(self.processor.image_processor, "size", {})
                min_pixels = size_cfg.get("shortest_edge", 56 * 56)
                max_pixels = size_cfg.get("longest_edge", 28 * 28 * 1280)
                from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

                for message in batched_messages:
                    per_sample = []
                    for msg in message:
                        if msg.get("role") != "user":
                            continue
                        for content in msg.get("content", []):
                            if not (isinstance(content, dict) and content.get("type") == "image"):
                                continue
                            img_obj = content.get("image")
                            if img_obj is None:
                                continue
                            if isinstance(img_obj, (str, Path)):
                                pil_img = Image.open(img_obj).convert("RGB")
                            else:
                                pil_img = img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj
                            img_tensor = TF.ToTensor()(pil_img)
                            if (
                                not use_patch_size_alin
                                and isinstance(img_tensor, torch.Tensor)
                                and img_tensor.dim() == 3
                            ):
                                _, height, width = img_tensor.shape
                                resized_height, resized_width = smart_resize(
                                    height, width, factor=14, min_pixels=min_pixels, max_pixels=max_pixels
                                )
                                if (resized_height, resized_width) != (height, width):
                                    img_tensor = self.processor.image_processor.resize(
                                        image=img_tensor,
                                        size=SizeDict(height=resized_height, width=resized_width),
                                        interpolation=self.processor.image_processor.resample,
                                    )
                            per_sample.append(img_tensor)
                    if per_sample:
                        shapes = [t.shape for t in per_sample if isinstance(t, torch.Tensor)]
                        if shapes and all(s == shapes[0] for s in shapes):
                            per_sample = torch.stack(per_sample)
                    geometry_encoder_inputs.append(per_sample)
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
            if self.batch_size > 1:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, padding=True, padding_side="left", return_tensors="pt")
            else:
                inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, do_resize=False, return_tensors="pt")
            if use_geometry_inputs and use_patch_size_alin and geometry_encoder_inputs is not None:
                grid_thw = inputs.get("image_grid_thw")
                if isinstance(grid_thw, torch.Tensor) and grid_thw.dim() == 2:
                    grid_thw_list = grid_thw.tolist()
                else:
                    grid_thw_list = None
                if grid_thw_list:
                    grid_idx = 0
                    resized_geometry_inputs = []
                    for per_sample in geometry_encoder_inputs:
                        if isinstance(per_sample, torch.Tensor):
                            per_sample_list = [img for img in per_sample]
                        else:
                            per_sample_list = per_sample
                        resized_sample = []
                        for img in per_sample_list:
                            if not isinstance(img, torch.Tensor) or img.dim() != 3:
                                resized_sample.append(img)
                                grid_idx += 1
                                continue
                            if grid_idx >= len(grid_thw_list):
                                resized_sample.append(img)
                                continue
                            _, grid_h, grid_w = grid_thw_list[grid_idx]
                            target_h = int(grid_h) * 14
                            target_w = int(grid_w) * 14
                            _, height, width = img.shape
                            if (target_h, target_w) != (height, width):
                                img = self.processor.image_processor.resize(
                                    image=img,
                                    size=SizeDict(height=target_h, width=target_w),
                                    interpolation=self.processor.image_processor.resample,
                                )
                            resized_sample.append(img)
                            grid_idx += 1
                        if isinstance(per_sample, torch.Tensor):
                            resized_sample = torch.stack(resized_sample) if resized_sample else per_sample
                        resized_geometry_inputs.append(resized_sample)
                    geometry_encoder_inputs = resized_geometry_inputs
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)
            if use_geometry_inputs and geometry_encoder_inputs is not None:
                device = inputs["input_ids"].device if "input_ids" in inputs else self.device
                moved_geometry_inputs = []
                for ge in geometry_encoder_inputs:
                    if isinstance(ge, torch.Tensor):
                        moved_geometry_inputs.append(ge.to(device))
                    elif isinstance(ge, list):
                        moved_geometry_inputs.append(
                            [g.to(device) if isinstance(g, torch.Tensor) else g for g in ge]
                        )
                    else:
                        moved_geometry_inputs.append(ge)
                inputs["geometry_encoder_inputs"] = moved_geometry_inputs

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 32768,
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