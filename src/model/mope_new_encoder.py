"""Frozen adapter and shared preprocessing for the updated MoPE-JEPA encoder."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "refs" / "mope_new"
DEFAULT_NUM_FRAMES = 16
DEFAULT_SAMPLING_RATE = 4
DEFAULT_INPUT_SIZE = 224
FEATURE_DIM = 768
SPATIAL_TOKENS = 196
TIME_BINS = 8
FULL_TOKENS = TIME_BINS * SPATIAL_TOKENS
POOL_IDS = {"none": 0, "temporal": 1, "mean": 2}


def build_mope_new_transform(input_size: int = DEFAULT_INPUT_SIZE):
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def select_video_indices(total: int, num_frames: int = DEFAULT_NUM_FRAMES,
                         sampling_rate: int = DEFAULT_SAMPLING_RATE) -> np.ndarray:
    if total <= 0:
        raise ValueError("MoPE-new cannot sample an empty video")
    span = num_frames * sampling_rate
    if total <= span:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        center = total // 2
        start = max(0, center - span // 2)
        start = min(start, total - span)
        indices = np.arange(start, start + span, sampling_rate)[:num_frames]
    return np.clip(indices, 0, total - 1)


def select_ordered_images(images: Sequence, num_frames: int = DEFAULT_NUM_FRAMES) -> list:
    if not images:
        raise ValueError("MoPE-new received an empty ordered image list")
    if len(images) >= num_frames:
        indices = np.linspace(0, len(images) - 1, num_frames).astype(int)
        return [images[int(i)] for i in indices]
    return list(images) + [images[-1]] * (num_frames - len(images))


def images_to_mope_new_tensor(images: Sequence[Image.Image], *, input_size: int = 224,
                              num_frames: int = 16) -> torch.Tensor:
    selected = select_ordered_images(images, num_frames)
    transform = build_mope_new_transform(input_size)
    frames = [transform(image.convert("RGB")) for image in selected]
    return torch.stack(frames, dim=1)  # [3,T,H,W]


def load_video_for_mope_new(video_path: str | Path, *, num_frames: int = 16,
                            sampling_rate: int = 4, input_size: int = 224) -> torch.Tensor:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required to read MoPE-new video inputs") from exc
    reader = decord.VideoReader(str(video_path))
    indices = select_video_indices(len(reader), num_frames, sampling_rate)
    batch = reader.get_batch(indices.tolist()).asnumpy()
    images = [Image.fromarray(frame).convert("RGB") for frame in batch]
    return images_to_mope_new_tensor(images, input_size=input_size, num_frames=num_frames)


def load_annotation_for_mope_new(annotation: Mapping, *, num_frames: int = 16,
                                 sampling_rate: int = 4, input_size: int = 224) -> torch.Tensor:
    data_root_raw = annotation.get("data_path") or ""
    data_root = Path(data_root_raw)
    if "image" in annotation:
        paths = annotation["image"]
        if isinstance(paths, str):
            paths = [paths]
        if not paths:
            raise ValueError("MoPE-new annotation contains an empty image list")
        images = []
        for item in paths:
            path = data_root / item if data_root_raw else Path(item)
            try:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(f"cannot read MoPE-new image: {path}") from exc
        return images_to_mope_new_tensor(images, input_size=input_size, num_frames=num_frames)
    if "video" in annotation:
        item = annotation["video"]
        path = data_root / item if data_root_raw else Path(item)
        return load_video_for_mope_new(
            path, num_frames=num_frames, sampling_rate=sampling_rate, input_size=input_size
        )
    raise ValueError("MoPE-new annotation must contain 'image' or 'video'")


def extract_state_dict(checkpoint) -> MutableMapping[str, torch.Tensor]:
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("model", "module", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if not isinstance(state, Mapping):
        raise TypeError(f"MoPE-new checkpoint does not contain a state dict: {type(state)!r}")
    return dict(state)


def clean_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        changed = True
        while changed:
            changed = False
            for prefix in ("_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def encoder_only_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state.items() if not key.startswith("predictor.")}


def validate_and_load_encoder(model: nn.Module, state: Mapping[str, torch.Tensor]) -> list[str]:
    expected = {
        key: value for key, value in model.state_dict().items() if not key.startswith("predictor.")
    }
    supplied = encoder_only_state(clean_state_dict(state))
    missing = sorted(set(expected) - set(supplied))
    mismatched = sorted(
        key for key in expected.keys() & supplied.keys()
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    )
    unexpected = sorted(set(supplied) - set(expected))
    if missing or mismatched or unexpected:
        raise RuntimeError(
            "invalid MoPE-new encoder checkpoint: "
            f"missing={missing[:8]}, shape_mismatch={mismatched[:8]}, unexpected={unexpected[:8]}"
        )
    result = model.load_state_dict(supplied, strict=False)
    non_predictor_missing = [key for key in result.missing_keys if not key.startswith("predictor.")]
    if non_predictor_missing or result.unexpected_keys:
        raise RuntimeError(
            f"MoPE-new strict encoder load failed: missing={non_predictor_missing}, "
            f"unexpected={result.unexpected_keys}"
        )
    return sorted(supplied)


def _import_native_builder(source_root: str | Path):
    root = Path(source_root).expanduser().resolve()
    model_file = root / "models" / "native_mope.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"MoPE-new source is incomplete: {model_file}")
    package = "_space_mope_new_upstream"
    models_package = f"{package}.models"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(root)]
        sys.modules[package] = module
    if models_package not in sys.modules:
        module = types.ModuleType(models_package)
        module.__path__ = [str(root / "models")]
        sys.modules[models_package] = module
    native = importlib.import_module(f"{models_package}.native_mope")
    return native.native_mope_jepa_base_patch16_224


class MoPENewEncoder(nn.Module):
    """Strict, frozen wrapper around checkpoint-50's 8-layer MoPE encoder."""

    def __init__(self, checkpoint_path: str | Path, *, source_root: str | Path = DEFAULT_SOURCE_ROOT,
                 num_frames: int = 16, sampling_rate: int = 4, input_size: int = 224,
                 pool_mode: str = "none", model_factory=None):
        super().__init__()
        if pool_mode not in POOL_IDS:
            raise ValueError(f"invalid MoPE-new pool_mode={pool_mode!r}; choose {tuple(POOL_IDS)}")
        if (num_frames, input_size) != (16, 224):
            raise ValueError("checkpoint-50 contract requires num_frames=16 and input_size=224")
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.input_size = input_size
        self.pool_mode = pool_mode
        factory = model_factory or _import_native_builder(source_root)
        model = factory(
            pretrained=False, all_frames=16, tubelet_size=2, encoder_depth=8,
            dense_layers=4, num_experts=4, top_k=2, num_shared_experts=1,
            router_score_func="sigmoid", router_bias_update_speed=0.0, num_anchors=1,
            pos_embed_type="3d_sincos", predictor_pos_embed_type="3d_sincos",
        )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        loaded = validate_and_load_encoder(model, extract_state_dict(checkpoint))
        model.predictor = nn.Identity()
        self.encoder = model
        self.loaded_encoder_keys = tuple(loaded)
        self.register_buffer(
            "contract", torch.tensor([1, num_frames, sampling_rate, input_size, POOL_IDS[pool_mode]]),
            persistent=True,
        )
        self._freeze()

    def _freeze(self):
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self._freeze()
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        expected = (3, self.num_frames, self.input_size, self.input_size)
        if frames.ndim != 5 or tuple(frames.shape[1:]) != expected:
            raise ValueError(f"MoPE-new expected [B,{','.join(map(str, expected))}], got {tuple(frames.shape)}")
        self._freeze()
        features = self.encoder.encode(frames, token_mask=None, record_stats=False)
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
            raise RuntimeError(f"MoPE-new expected [B,N,{FEATURE_DIM}], got {tuple(features.shape)}")
        if features.shape[1] != FULL_TOKENS or features.shape[1] % SPATIAL_TOKENS:
            raise RuntimeError(
                f"MoPE-new expected {FULL_TOKENS}=8x{SPATIAL_TOKENS} time-major tokens, "
                f"got {features.shape[1]}"
            )
        if self.pool_mode == "temporal":
            features = features.view(features.shape[0], TIME_BINS, SPATIAL_TOKENS, FEATURE_DIM).mean(2)
        elif self.pool_mode == "mean":
            features = features.mean(1, keepdim=True)
        return features


PROJECTOR_KEYS = (
    "norm.weight", "norm.bias", "k_proj.weight", "k_proj.bias",
    "v_proj.weight", "v_proj.bias", "out_proj.weight", "out_proj.bias",
)


def load_hf_component_state(checkpoint_dir: str | Path, component: str) -> dict[str, torch.Tensor]:
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    state = {}
    try:
        from safetensors.torch import load_file
        for shard in sorted(root.glob("*.safetensors")):
            for key, value in load_file(str(shard), device="cpu").items():
                prefix = f"model.{component}."
                if key.startswith(prefix):
                    state[key[len(prefix):]] = value
    except ImportError:
        pass
    if not state:
        for shard in sorted(root.glob("pytorch_model*.bin")):
            values = torch.load(str(shard), map_location="cpu", weights_only=True)
            prefix = f"model.{component}."
            for key, value in values.items():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = value
    return state


def load_projector_warmstart(projector: nn.Module, checkpoint_dir: str | Path) -> dict[str, float]:
    state = load_hf_component_state(checkpoint_dir, "_mope_projector")
    missing = sorted(set(PROJECTOR_KEYS) - set(state))
    unexpected = sorted(set(state) - set(projector.state_dict()))
    mismatched = sorted(
        key for key in set(state) & set(projector.state_dict())
        if tuple(state[key].shape) != tuple(projector.state_dict()[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"invalid E-00b-new projector warm-start: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    projector.load_state_dict(state, strict=True)
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    return {key: float(value.float().norm().item()) for key, value in state.items()}


def load_saved_eval_components(encoder: nn.Module, projector: nn.Module,
                               checkpoint_dir: str | Path,
                               expected_contract: torch.Tensor) -> None:
    encoder_state = load_hf_component_state(checkpoint_dir, "_mope_encoder")
    projector_state = load_hf_component_state(checkpoint_dir, "_mope_projector")
    if not encoder_state:
        raise RuntimeError(f"MoPE-new eval checkpoint has no encoder weights: {checkpoint_dir}")
    if not projector_state:
        raise RuntimeError(f"MoPE-new eval checkpoint has no projector weights: {checkpoint_dir}")
    saved_contract = encoder_state.get("contract")
    if saved_contract is None or not torch.equal(saved_contract.cpu(), expected_contract.cpu()):
        raise RuntimeError(
            f"MoPE-new train/eval contract mismatch: saved={saved_contract}, "
            f"expected={expected_contract.cpu()}"
        )
    encoder.load_state_dict(encoder_state, strict=True)
    missing_projector = sorted(set(PROJECTOR_KEYS) - set(projector_state))
    if missing_projector:
        raise RuntimeError(f"MoPE-new eval projector is incomplete: {missing_projector}")
    projector.load_state_dict(projector_state, strict=True)
