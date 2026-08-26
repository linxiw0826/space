"""
MoPE data pipeline wrappers.

Wraps LazySupervisedDataset and DataCollatorForSupervisedDataset to inject
raw_frames (MoPE input) into each batch without modifying vendored GUIDE code.

mope_frames shape: [B, 3, T, 224, 224] (ImageNet-normalised float32)
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
_STRICT_MOPE_LOADING = False


def _load_mope_frames(ann: dict, all_frames: int) -> torch.Tensor:
    """Load `all_frames` frames from one annotation dict.

    Handles two cases:
      - "image": list of paths (SPAR multi-image format)
      - "video": single path (video format, uses decord)

    Returns: float32 tensor of shape [3, T, H, W] = [3, all_frames, 224, 224]
    """
    data_path = ann.get("data_path", "")
    frames = []

    def _load_pil(path: str) -> torch.Tensor:
        if data_path:
            path = os.path.join(data_path, path)
        pil = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)       # [3, 224, 224]
        return (t - _IMAGENET_MEAN) / _IMAGENET_STD

    if "image" in ann:
        image_list = ann["image"]
        if isinstance(image_list, str):
            image_list = [image_list]
        n = len(image_list)
        if n >= all_frames:
            indices = np.linspace(0, n - 1, all_frames, dtype=int)
        else:
            indices = list(range(n)) + [n - 1] * (all_frames - n)
            indices = np.array(indices, dtype=int)
        for idx in indices:
            frames.append(_load_pil(image_list[int(idx)]))

    elif "video" in ann:
        try:
            import decord
            video_rel = ann["video"]
            video_path = os.path.join(data_path, video_rel) if data_path else video_rel
            vr = decord.VideoReader(video_path)
            total = len(vr)
            indices = np.linspace(0, total - 1, all_frames, dtype=int)
            for idx in indices:
                frame_np = vr[int(idx)].asnumpy()
                pil = Image.fromarray(frame_np).convert("RGB").resize((224, 224), Image.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                frames.append((t - _IMAGENET_MEAN) / _IMAGENET_STD)
        except Exception as exc:
            print(f"[MoPE] WARNING: video frame load failed ({exc}), using zeros.")
            frames = [torch.zeros(3, 224, 224) for _ in range(all_frames)]

    if not frames:
        frames = [torch.zeros(3, 224, 224) for _ in range(all_frames)]

    stacked = torch.stack(frames)           # [T, 3, 224, 224]
    return stacked.permute(1, 0, 2, 3)     # [3, T, 224, 224]


class MoPEDatasetWrapper(Dataset):
    """Wraps LazySupervisedDataset to add `raw_frames` field per item."""

    def __init__(self, base_dataset, mope_all_frames: int = 8):
        self.base = base_dataset
        self.mope_all_frames = mope_all_frames

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i: int):
        item = self.base[i]
        ann = self.base.list_data_dict[i]
        try:
            item["raw_frames"] = _load_mope_frames(ann, self.mope_all_frames)
        except Exception as exc:
            if _STRICT_MOPE_LOADING:
                raise RuntimeError(
                    f"strict MoPE frame loading failed for dataset index {i}"
                ) from exc
            print(f"[MoPE] WARNING: frame load failed for idx {i} ({exc}), using zeros.")
            item["raw_frames"] = torch.zeros(3, self.mope_all_frames, 224, 224)
        return item

    # Forward sampler-required properties to the base dataset
    @property
    def lengths(self):
        return self.base.lengths

    @property
    def modality_lengths(self):
        return self.base.modality_lengths

    @property
    def pre_calculated_length(self):
        return self.base.pre_calculated_length

    @property
    def list_data_dict(self):
        return self.base.list_data_dict


class MoPECollatorWrapper:
    """Wraps DataCollatorForSupervisedDataset to stack raw_frames into mope_frames.

    For E-02b concat fusion (mope_num_tokens > 0): prepends mope_num_tokens
    -100 entries to each labels row so that the loss shape matches the extended
    logit sequence produced by _patch_model_for_mope_concat.
    """

    def __init__(self, base_collator, mope_num_tokens: int = 0):
        self.base = base_collator
        self.mope_num_tokens = mope_num_tokens
        self._diag_calls = 0

    def __call__(self, instances):
        batch = self.base(instances)
        if instances and "raw_frames" in instances[0]:
            # Stack [3, T, 224, 224] tensors → [B, 3, T, 224, 224]
            batch["mope_frames"] = torch.stack(
                [inst["raw_frames"] for inst in instances]
            )
        # [DIAG] Print label statistics for the first 5 batches.
        if self._diag_calls < 5 and "labels" in batch:
            self._diag_calls += 1
            lbl = batch["labels"]
            n_valid = (lbl != -100).sum().item()
            n_total = lbl.numel()
            has_mope = "mope_frames" in batch
            print(
                f"[MoPE COLLATOR DIAG] batch#{self._diag_calls}: "
                f"labels shape={tuple(lbl.shape)}, "
                f"valid(non-(-100))={n_valid}/{n_total}, "
                f"mope_frames_present={has_mope}, "
                f"input_ids shape={tuple(batch.get('input_ids', lbl).shape)}"
            )

        # E-02b: extend labels to match T+N_mope logit sequence length.
        if self.mope_num_tokens > 0 and "labels" in batch and "mope_frames" in batch:
            pad = batch["labels"].new_full(
                (batch["labels"].shape[0], self.mope_num_tokens), -100
            )
            batch["labels"] = torch.cat([pad, batch["labels"]], dim=1)
        return batch
