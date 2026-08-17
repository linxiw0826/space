"""Video dataset for MoPE-JEPA pretraining.

The primary list contains physics/dynamics videos. An optional general-data
list is retained for future binary-gate training. No 17-class labels or soft
label vectors are produced.
"""

import random
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu as decord_cpu
from PIL import Image

from .pretrain_datasets import DataAugmentationForVideoMAEv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _read_video_paths(source):
    source = Path(source).expanduser()
    if source.is_file() and source.suffix.lower() == ".txt":
        with source.open("r", encoding="utf-8") as handle:
            return [Path(line.strip()) for line in handle if line.strip()]
    if source.is_dir():
        return sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    raise FileNotFoundError(f"Video source does not exist: {source}")


class VideoPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_root, transform, num_frames=16,
                 sampling_rate=4, num_sample=1, general_data="",
                 general_max=0):
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.num_sample = num_sample
        self.clips = []

        primary = [path for path in _read_video_paths(datasets_root) if path.is_file()]
        self.clips.extend((str(path), 0) for path in primary)

        general = []
        if general_data:
            general = [path for path in _read_video_paths(general_data) if path.is_file()]
            if general_max > 0:
                general = general[:general_max]
            self.clips.extend((str(path), 1) for path in general)

        print(
            f"[Dataset] TOTAL {len(self.clips)} clips "
            f"(primary={len(primary)}, general={len(general)})"
        )
        if not self.clips:
            raise RuntimeError(f"No valid videos found in {datasets_root}")

    def _sample_frame_ids(self, total):
        if total <= 0:
            return [0] * self.num_frames
        segments = 4
        per_segment = self.num_frames // segments
        edges = np.linspace(0, total, segments + 1).astype(int)
        indices = []
        for index in range(segments):
            start = int(edges[index])
            end = max(int(edges[index + 1]), start + 1)
            indices.extend(
                np.linspace(start, end - 1, per_segment).astype(int).tolist()
            )
        return np.clip(indices, 0, total - 1).astype(int)[:self.num_frames].tolist()

    def _load_frames(self, video_path):
        reader = VideoReader(video_path, num_threads=1, ctx=decord_cpu(0))
        frame_ids = self._sample_frame_ids(len(reader))
        frames = reader.get_batch(frame_ids).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        video_path, binary_label = self.clips[index]
        try:
            images = self._load_frames(video_path)
        except Exception as error:
            print(f"[Dataset] Load failed ({video_path}): {error}; retrying")
            return self.__getitem__(random.randrange(len(self.clips)))

        label = torch.tensor(binary_label, dtype=torch.long)
        if self.num_sample > 1:
            data, enc_masks, dec_masks, labels = [], [], [], []
            for _ in range(self.num_sample):
                sample, enc_mask, dec_mask = self.transform((images, None))
                sample = sample.view(
                    (self.num_frames, 3) + sample.size()[-2:]
                ).transpose(0, 1)
                data.append(sample)
                enc_masks.append(enc_mask)
                dec_masks.append(dec_mask)
                labels.append(label)
            return data, enc_masks, dec_masks, labels

        sample, enc_mask, dec_mask = self.transform((images, None))
        sample = sample.view(
            (self.num_frames, 3) + sample.size()[-2:]
        ).transpose(0, 1)
        return sample, enc_mask, dec_mask, label


def build_video_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAEv2(args)
    dataset = VideoPretrainDataset(
        datasets_root=args.datasets_root,
        transform=transform,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_sample=args.num_sample,
        general_data=getattr(args, "general_data", ""),
        general_max=getattr(args, "general_max", 0),
    )
    print(f"Data Aug = {transform}")
    return dataset


# Backward-compatible import for older launchers.
build_wisa_pretraining_dataset = build_video_pretraining_dataset
