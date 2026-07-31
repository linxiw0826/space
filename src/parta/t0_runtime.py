"""Authoritative batch-1 processing and model adapter for Part A T0-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torchvision import transforms as TF
from transformers.image_utils import SizeDict

from parta_data_contract import ContractError

from .canonical_data import PartASample

MODEL_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "pixel_values",
        "image_grid_thw",
        "geometry_encoder_inputs",
        "labels",
    }
)


@dataclass(frozen=True)
class ProcessedFixture:
    model_kwargs: Mapping[str, Any]
    sample: PartASample
    images: tuple[Image.Image, ...]
    frame_token_counts: tuple[int, ...]
    frame_ids: tuple[int, ...]
    media_kind: str
    visual_prefix_before_question: bool
    visual_token_mask: torch.Tensor
    question_token_span: tuple[int, int]


class PartAT0Collator:
    """Process exact decoded frames without using the generic resampler."""

    def __init__(self, processor: Any, *, spatial_merge_size: int | None = None):
        self.processor = processor
        inferred = getattr(getattr(processor, "image_processor", None), "merge_size", None)
        self.spatial_merge_size = int(spatial_merge_size or inferred or 0)
        if self.spatial_merge_size <= 0:
            raise ValueError("processor must expose a positive image spatial merge size")
        if not callable(getattr(processor, "apply_chat_template", None)):
            raise TypeError("processor lacks authoritative apply_chat_template API")

    def __call__(
        self,
        sample: PartASample,
        images: Sequence[Image.Image],
        *,
        question_override: str | None = None,
    ) -> ProcessedFixture:
        if len(images) != len(sample.qa["actual_frame_indices"]):
            raise ContractError("decoded image count differs from exact frame binding")
        if any(not isinstance(image, Image.Image) for image in images):
            raise TypeError("T0 collator accepts decoded PIL images only")
        question, answer = _question_answer(sample.qa["conversations"])
        if question_override is not None:
            question = question_override
        question = question.replace("<video>", "").replace("<image>", "").strip()
        content = [
            {"type": "image", "image": image}
            for image in images
        ]
        content.append({"type": "text", "text": question})
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        processed = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if not isinstance(processed, Mapping):
            raise TypeError("processor apply_chat_template must return a mapping")
        unexpected = set(processed) - MODEL_INPUT_KEYS
        # Metadata is never forwarded; processor-native auxiliary keys also
        # fail closed so model API changes cannot silently alter T0.
        if unexpected:
            raise ContractError(f"unsupported processor/model keys: {sorted(unexpected)}")
        if "image_grid_thw" not in processed or "input_ids" not in processed:
            raise ContractError("processor output lacks image_grid_thw/input_ids")
        grid = processed["image_grid_thw"]
        if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or grid.shape[1] != 3:
            raise ContractError("image_grid_thw must be tensor [frames,3]")
        if grid.shape[0] != len(images):
            raise ContractError(
                f"image_grid_thw cardinality {grid.shape[0]} != exact frames {len(images)}"
            )
        denominator = self.spatial_merge_size**2
        products = grid.to(torch.long).prod(-1)
        if (products <= 0).any() or (products % denominator != 0).any():
            raise ContractError("image grid is incompatible with spatial merge size")
        counts = tuple(int(value) for value in (products // denominator).tolist())
        input_ids = processed["input_ids"]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ContractError("input_ids must be a batch-1 tensor")
        visual_token_id = getattr(self.processor, "image_token_id", None)
        if visual_token_id is None:
            visual_token_id = getattr(getattr(self.processor, "tokenizer", None), "image_token_id", None)
        if visual_token_id is None:
            raise ContractError("processor does not expose the authoritative image token id")
        visual_mask = input_ids.eq(int(visual_token_id))
        if int(visual_mask.sum()) != sum(counts):
            raise ContractError("input visual-token mask disagrees with image grids")
        question_ids = _tokenize_text(self.processor, question)
        question_span = _unique_subsequence_span(input_ids[0], question_ids, "question")
        visual_positions = visual_mask[0].nonzero(as_tuple=False).flatten()
        visual_before_question = bool(
            visual_positions.numel() and int(visual_positions.max()) < question_span[0]
        )
        if not visual_before_question:
            raise ContractError("authoritative visual tokens are not before the question")

        labels = _authoritative_answer_labels(self.processor, messages, processed)
        kwargs = {
            key: value
            for key, value in processed.items()
            if key in MODEL_INPUT_KEYS and isinstance(value, torch.Tensor)
        }
        if set(kwargs) != set(processed):
            raise TypeError("every processor model input must be a tensor")
        image_processor = getattr(self.processor, "image_processor", None)
        if not callable(getattr(image_processor, "resize", None)) or not hasattr(image_processor, "resample"):
            raise ContractError("GUIDE image_processor resize/resample API is unavailable")
        to_tensor = TF.ToTensor()
        geometry_inputs = []
        for image, grid_row in zip(images, grid.tolist()):
            tensor = to_tensor(image.convert("RGB"))
            target_height = int(grid_row[1]) * 14
            target_width = int(grid_row[2]) * 14
            if tensor.shape[-2:] != (target_height, target_width):
                tensor = image_processor.resize(
                    image=tensor,
                    size=SizeDict(height=target_height, width=target_width),
                    interpolation=image_processor.resample,
                )
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (3, target_height, target_width):
                raise ContractError("GUIDE image_processor resize returned an invalid geometry tensor")
            geometry_inputs.append(tensor)
        kwargs["geometry_encoder_inputs"] = [torch.stack(geometry_inputs)]
        kwargs["labels"] = labels
        return ProcessedFixture(
            model_kwargs=kwargs,
            sample=sample,
            images=tuple(images),
            frame_token_counts=counts,
            frame_ids=tuple(int(value) for value in sample.qa["actual_frame_indices"]),
            media_kind=str(sample.qa["media_kind"]),
            visual_prefix_before_question=visual_before_question,
            visual_token_mask=visual_mask,
            question_token_span=question_span,
        )

    @staticmethod
    def assert_tap_cardinality(
        fixture: ProcessedFixture,
        visual_state_valid_mask: torch.Tensor,
    ) -> None:
        if visual_state_valid_mask.ndim != 2 or visual_state_valid_mask.shape[0] != 1:
            raise ContractError("T0-A requires batch-1 visual_state_valid_mask")
        actual = int(visual_state_valid_mask[0].sum().item())
        expected = sum(fixture.frame_token_counts)
        if actual != expected:
            raise ContractError(f"tap valid tokens {actual} != grid-derived tokens {expected}")

    def with_question(
        self, fixture: ProcessedFixture, question: str
    ) -> ProcessedFixture:
        """Retokenize text while reusing the exact processed visual tensors."""
        alternate = self(
            fixture.sample, fixture.images, question_override=question
        )
        kwargs = dict(alternate.model_kwargs)
        for key in ("pixel_values", "image_grid_thw", "geometry_encoder_inputs"):
            if key not in fixture.model_kwargs or key not in kwargs:
                raise ContractError(f"question-invariance visual key missing: {key}")
            kwargs[key] = fixture.model_kwargs[key]
        return ProcessedFixture(
            model_kwargs=kwargs,
            sample=fixture.sample,
            images=fixture.images,
            frame_token_counts=fixture.frame_token_counts,
            frame_ids=fixture.frame_ids,
            media_kind=fixture.media_kind,
            visual_prefix_before_question=fixture.visual_prefix_before_question,
            visual_token_mask=alternate.visual_token_mask,
            question_token_span=alternate.question_token_span,
        )


def _tokenize_text(processor: Any, text: str) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    if not callable(tokenizer):
        raise ContractError("processor tokenizer is required to audit question placement")
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        raise ContractError("question tokenization returned no tokens")
    return ids.reshape(-1)


def _unique_subsequence_span(sequence: torch.Tensor, needle: torch.Tensor, name: str) -> tuple[int, int]:
    sequence_values = sequence.tolist()
    needle_values = needle.tolist()
    matches = [
        start
        for start in range(len(sequence_values) - len(needle_values) + 1)
        if sequence_values[start : start + len(needle_values)] == needle_values
    ]
    if len(matches) != 1:
        raise ContractError(f"{name} token span must occur exactly once, found {len(matches)}")
    return matches[0], matches[0] + len(needle_values)


def _authoritative_answer_labels(
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    full: Mapping[str, Any],
) -> torch.Tensor:
    """Mask everything except the assistant completion using the chat template boundary."""
    prompt = processor.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_ids = prompt.get("input_ids") if isinstance(prompt, Mapping) else None
    full_ids = full.get("input_ids")
    if not isinstance(prompt_ids, torch.Tensor) or not isinstance(full_ids, torch.Tensor):
        raise ContractError("chat template did not return prompt/full input_ids")
    prompt_length = prompt_ids.shape[1]
    if prompt_length >= full_ids.shape[1] or not torch.equal(
        prompt_ids[0], full_ids[0, :prompt_length]
    ):
        raise ContractError("assistant generation prompt is not an exact full-chat prefix")
    labels = torch.full_like(full_ids, -100)
    labels[:, prompt_length:] = full_ids[:, prompt_length:]
    if not labels.ne(-100).any():
        raise ContractError("authoritative assistant labels are empty")
    return labels


def _question_answer(conversations: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    question = answer = None
    for row in conversations:
        role = row.get("from", row.get("role"))
        value = row.get("value", row.get("content"))
        if not isinstance(value, str):
            continue
        if role in {"human", "user"} and question is None:
            question = value
        elif role in {"gpt", "assistant"} and answer is None:
            answer = value
    if question is None or answer is None:
        raise ContractError("canonical QA conversations require one user question and assistant answer")
    return question, answer


def forward_visual_tap(model: torch.nn.Module, fixture: ProcessedFixture) -> Any:
    """Forward only reviewed model kwargs plus the explicit tap request."""
    inner = getattr(model, "model", model)
    if getattr(inner, "_mope_encoder", None) is not None or getattr(inner, "_mope_projector", None) is not None:
        raise ContractError("T0-A must remain MoPE-free")
    if getattr(getattr(model, "config", None), "use_geometry_encoder", None) is not True:
        raise ContractError("T0-A requires the reproduced GUIDE geometry encoder")
    if "geometry_encoder_inputs" not in fixture.model_kwargs:
        raise ContractError("T0-A requires visual-derived GUIDE geometry inputs")
    output = model(**fixture.model_kwargs, return_visual_state_tap=True)
    hidden = getattr(output, "visual_state_hidden", None)
    valid = getattr(output, "visual_state_valid_mask", None)
    logits = getattr(output, "logits", None)
    if not isinstance(hidden, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise ContractError("model did not return authoritative visual state tap")
    if not isinstance(logits, torch.Tensor):
        raise ContractError("model did not return QA logits")
    PartAT0Collator.assert_tap_cardinality(fixture, valid)
    return output
