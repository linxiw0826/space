import json
import random
import logging
import re
import time
import itertools
import os
import copy
import pickle
import math
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any, Union
from collections.abc import Sequence
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import cv2

import transformers
from transformers.image_utils import SizeDict

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from .utils import prepare_image_inputs

# Import for VG-LLM resize
try:
    from PIL import ImageOps
    from torchvision import transforms as TF
    from transformers.utils import is_torch_available
    import torch
except Exception:
    ImageOps = None
    TF = None
    is_torch_available = lambda: False
    torch = None

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def unproject_world_coords(intrinsics, poses, depths):
    """
    Unproject depth maps to world coordinates (VG-LLM).
    
    Args:
        intrinsics: (V, 4, 4) or (V, 3, 3) camera intrinsic matrices
        poses: (V, 4, 4) camera-to-world transformation matrices
        depths: (V, H, W) depth maps in millimeters
    
    Returns:
        world_coords: (V, H, W, 3) world coordinates for each pixel
    """
    V, H, W = depths.shape
    
    # Generate pixel coordinate grids
    y = torch.arange(0, H, dtype=torch.float32, device=depths.device)
    x = torch.arange(0, W, dtype=torch.float32, device=depths.device)
    y, x = torch.meshgrid(y, x, indexing='ij')
    
    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    
    # Extract intrinsic parameters
    if intrinsics.shape[-1] == 4:
        fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
        fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
        cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
        cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)
    else:  # 3x3 intrinsic matrix
        fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
        fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
        cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
        cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)
    
    # Convert depth from millimeters to meters
    z = depths.view(V, H*W).float() / 1000.0       # (V, H*W)
    
    # Unproject to camera coordinates
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    z_cam = z
    
    # Stack to form camera coordinates in homogeneous form
    cam_coords = torch.stack([
        x_cam, y_cam, z_cam, torch.ones_like(x_cam)
    ], dim=-1)      # (V, H*W, 4)
    
    # Transform to world coordinates
    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / (world_coords[..., 3:4] + 1e-8)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)
    
    return world_coords


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools: PIL images pass through, string paths become absolute
    image_pool = [
        {"type": "image", "image": img if isinstance(img, Image.Image) else _make_abs_paths(base_path, img)}
        for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages

def preprocess_qwen_visual(
    sources,
    processor,
    images_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    if "images" in source:
        source["image"]=source["images"]
    messages = _build_messages(source, base_path)

    # Extract custom parameters for VG-LLM resize and geometry inputs
    use_geometry_inputs = False
    geometry_encoder_inputs = None

    if images_kwargs:
        use_geometry_inputs = images_kwargs.pop("use_geometry_inputs", False)
        use_patch_size_alin = images_kwargs.pop("use_patch_size_alin", False)
    else:
        use_patch_size_alin = False
    
    apply_kwargs: Dict[str, Any] = {}
    if images_kwargs:
        apply_kwargs["images_kwargs"] = images_kwargs
    
 

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", **apply_kwargs
    )
    
    # Add geometry_encoder_inputs to result if available
    if use_geometry_inputs:
        # Reconstruct geometry_encoder_inputs from pixel_values and image_grid_thw
        # This ensures geometry_encoder_inputs matches the processed image shape
        if "pixel_values" in full_result and "image_grid_thw" in full_result:
            geometry_encoder_inputs = []
            to_tensor = TF.ToTensor() if TF is not None else None
            for message in messages:
                if message.get("role") != "user":
                    continue
                for content in message.get("content", []):
                    if not (isinstance(content, dict) and content.get("type") == "image"):
                        continue
                    img_obj = content.get("image")
                    if img_obj is None:
                        continue
                    if isinstance(img_obj, (str, Path)):
                        pil_img = Image.open(img_obj).convert("RGB")
                    else:
                        pil_img = img_obj.convert("RGB") if isinstance(img_obj, Image.Image) else img_obj
                    if to_tensor is None:
                        raise RuntimeError("torchvision is required for TF.ToTensor() conversion.")
                    img_tensor = to_tensor(pil_img)
                    geometry_encoder_inputs.append(img_tensor)
            
            # Resize geometry inputs to be divisible by VGGT patch size (14)
            if isinstance(geometry_encoder_inputs, list) and len(geometry_encoder_inputs) > 0:
                from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

                size_cfg = getattr(processor.image_processor, "size", {})
                min_pixels = size_cfg.get("shortest_edge", 56 * 56)
                max_pixels = size_cfg.get("longest_edge", 28 * 28 * 1280)

                resized_geometry_inputs = []
            grid_thw = full_result.get("image_grid_thw")
            if isinstance(grid_thw, torch.Tensor) and grid_thw.dim() == 2:
                grid_thw_list = grid_thw.tolist()
            else:
                grid_thw_list = None
            for img_idx, img in enumerate(geometry_encoder_inputs):
                if not isinstance(img, torch.Tensor) or img.dim() != 3:
                    resized_geometry_inputs.append(img)
                    continue
                _, height, width = img.shape
                if use_patch_size_alin and grid_thw_list and img_idx < len(grid_thw_list):
                    _, grid_h, grid_w = grid_thw_list[img_idx]
                    resized_height = int(grid_h) * 14
                    resized_width = int(grid_w) * 14
                else:
                    resized_height, resized_width = smart_resize(
                        height, width, factor=14, min_pixels=min_pixels, max_pixels=max_pixels
                    )
                if (resized_height, resized_width) != (height, width):
                    img = processor.image_processor.resize(
                        image=img,
                        size=SizeDict(height=resized_height, width=resized_width),
                        interpolation=processor.image_processor.resample,
                    )
                resized_geometry_inputs.append(img)
            geometry_encoder_inputs = resized_geometry_inputs

            full_result["geometry_encoder_inputs"] = geometry_encoder_inputs
            

            
        else:
            # Fallback to original geometry_encoder_inputs if pixel_values not available
            if geometry_encoder_inputs is not None:
                full_result["geometry_encoder_inputs"] = geometry_encoder_inputs

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        sub_ann["tag"] = data.get("tag", "2d")
                else:
                    ann["data_path"] = data["data_path"]
                    ann["tag"] = data.get("tag", "2d")
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict
        
        
        self.compute_world_coords = getattr(data_args, "compute_world_coords", False)
        self.embodiedscan_scenes = {}
        if self.compute_world_coords:
            embodiedscan_dir = getattr(data_args, "embodiedscan_dir", "data/embodiedscan")
            if embodiedscan_dir and os.path.exists(embodiedscan_dir):
                try:
                    for split in ["train", "val", "test"]:
                        pkl_path = os.path.join(embodiedscan_dir, f"embodiedscan_infos_{split}.pkl")
                        if os.path.exists(pkl_path):
                            with open(pkl_path, "rb") as f:
                                data = pickle.load(f)["data_list"]
                                for item in data:
                                    if item["sample_idx"].startswith("scannet"):
                                        self.embodiedscan_scenes[item["sample_idx"]] = item
                    rank0_print(f"Loaded {len(self.embodiedscan_scenes)} scenes from embodiedscan metadata")
                except Exception as e:
                    rank0_print(f"Warning: Failed to load embodiedscan metadata: {e}")
                    self.embodiedscan_scenes = {}
        
        self.load_scene_bbox = getattr(data_args, "load_scene_bbox", False)
        self.scan2obj = {}
        if self.load_scene_bbox:
            val_box_type = getattr(data_args, "val_box_type", "pred")
            try:
                for split in ['train', 'val']:
                    box_type = "gt" if split == "train" else val_box_type
                    filename = os.path.join("data", "metadata", f"scannet_{split}_{box_type}_box.json")
                    if os.path.exists(filename):
                        with open(filename) as f:
                            data = json.load(f)
                            self.scan2obj.update(data)
                rank0_print(f"Loaded bbox data for {len(self.scan2obj)} scenes")
            except Exception as e:
                rank0_print(f"Warning: Failed to load scene bbox data: {e}")
                self.scan2obj = {}


        self.use_geometry_inputs = getattr(data_args, "use_geometry_inputs", False)
        
        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num*192
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        """Process image using unified method (VG-LLM)."""
        processor = copy.deepcopy(self.processor.image_processor)
        if isinstance(image_file, str):
            image = Image.open(image_file).convert("RGB")
        elif isinstance(image_file, Image.Image):
            image = image_file
        else:
            raise NotImplementedError(f"Unsupported image type: {type(image_file)}")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):
        """Draw visual marks on images (VG-LLM)."""
        if spar_info is None:
            return
        try:
            info = json.loads(spar_info) if isinstance(spar_info, str) else spar_info
            task_type = info.get("type")
            if task_type:
                # Try to import draw_marker if available
                try:
                    from .draw_marker import DRAW_FUNCTIONS
                    draw_fn = DRAW_FUNCTIONS.get(task_type)
                    if draw_fn:
                        if len(images) == 1:
                            draw_fn(images[0], info)
                        else:
                            draw_fn(images, info)
                except ImportError:
                    rank0_print(f"Warning: draw_marker module not found, skipping visual marks for {task_type}")
        except Exception as e:
            rank0_print(f"Warning: Failed to draw visual marks: {e}")

    def process_video(self, video_file):
        """Process video file (VG-LLM)."""
        if not os.path.exists(video_file):
            rank0_print(f"File not exist: {video_file}")
            raise FileNotFoundError(f"Video file not found: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.processor.image_processor)
        processor.max_pixels = getattr(self.data_args, "video_max_frame_pixels", processor.max_pixels)
        processor.min_pixels = getattr(self.data_args, "video_min_frame_pixels", processor.min_pixels)
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.processor.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    
    def read_video_images(self, source):
        """Read video frames and return paths relative to data_path.

        For directory-based videos, returns sampled frame relative paths.
        For video files (.mp4/.avi/.mov), decodes frames, saves to a cache
        directory next to the video file, and returns cached frame relative paths.
        """
        assert isinstance(source["video"], str), "video should be a string"
        data_path = source["data_path"]
        video_rel = source["video"]
        video_file = os.path.join(data_path, video_rel)
        if not os.path.exists(video_file):
            rank0_print(f"File not exist: {video_file}")
            raise FileNotFoundError(f"Video file not found: {video_file}")

        def get_frame_indices(total_frames, fps=1):
            video_length = total_frames / fps
            interval = getattr(self.data_args, "base_interval", 2)
            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )
            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            return frame_idx

        if os.path.isdir(video_file):
            filenames = sorted(
                f for f in os.listdir(video_file)
                if os.path.isfile(os.path.join(video_file, f))
            )
            frame_idx = get_frame_indices(len(filenames), 1)
            image_paths = [os.path.join(video_rel, filenames[i]) for i in frame_idx]
        elif any(video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]):
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_idx = get_frame_indices(total_frames, avg_fps)

            cache_rel = video_rel + "_frames_cache"
            cache_dir = os.path.join(data_path, cache_rel)
            os.makedirs(cache_dir, exist_ok=True)

            image_paths = []
            for idx in frame_idx:
                frame_name = f"frame_{idx:06d}.jpg"
                cached_abs = os.path.join(cache_dir, frame_name)
                if not os.path.exists(cached_abs):
                    frame = vr[int(idx)].asnumpy()
                    Image.fromarray(frame).convert("RGB").save(cached_abs)
                image_paths.append(os.path.join(cache_rel, frame_name))
        else:
            raise ValueError(f"Unsupported video file format: {video_file}")
        return image_paths

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        # Convert video to images: sample frames and replace <video> with <image>
        if "video" in sources[0]:
            source = copy.copy(sources[0])
            source["conversations"] = copy.deepcopy(source["conversations"])
            image_paths = self.read_video_images(source)
            num_image = len(image_paths)
            source["image"] = image_paths
            source["conversations"][0]["value"] = source["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            source["conversations"][0]["value"] = source["conversations"][0]["value"].replace(
                f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
            )
            del source["video"]
            sources = [source]

        # Draw visual marks for spar data: load images, draw markers, pass PIL directly
        spar_info = sources[0].get("spar_info", None)
        if spar_info is not None:
            source = copy.copy(sources[0])
            image_key = "image" if "image" in source else "images" if "images" in source else None
            if image_key is not None:
                image_files = source[image_key]
                if isinstance(image_files, str):
                    image_files = [image_files]
                data_path = source.get("data_path", "")

                abs_paths = [os.path.join(data_path, f) for f in image_files]
                pil_images = [Image.open(p).convert("RGB") for p in abs_paths]
                self.draw_visual_marks(pil_images, spar_info)
                source[image_key] = pil_images
                sources = [source]

        images_kwargs = None
        if getattr(self.data_args, "use_geometry_inputs", False):
            images_kwargs = {}
            if getattr(self.data_args, "use_geometry_inputs", False):
                images_kwargs["use_geometry_inputs"] = True
                images_kwargs["use_patch_size_alin"] = getattr(
                    self.data_args, "use_patch_size_alin", False
                )
                

        data_dict = preprocess_qwen_visual(sources, self.processor, images_kwargs=images_kwargs)

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        
        # VG-LLM specific: Handle geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = []
            for instance in instances:
                if "geometry_encoder_inputs" in instance:
                    ge_inputs = instance["geometry_encoder_inputs"]
                    # ge_inputs is list[torch.Tensor], each tensor is (C, H, W)
                    # Stack them into a single tensor (N, C, H, W) if all have same shape
                    if isinstance(ge_inputs, list) and len(ge_inputs) > 0:
                        # Check if all tensors have the same shape
                        shapes = [t.shape for t in ge_inputs if isinstance(t, torch.Tensor)]
                        if len(shapes) > 0 and all(s == shapes[0] for s in shapes):
                            geometry_encoder_inputs.append(torch.stack(ge_inputs))
                        else:
                            # If shapes differ, keep as list (will need special handling)
                            geometry_encoder_inputs.append(ge_inputs)
                    else:
                        geometry_encoder_inputs.append(ge_inputs)
                else:
                    # Missing geometry_encoder_inputs, skip this instance
                    break
            
            if len(geometry_encoder_inputs) == len(instances):
                batch["geometry_encoder_inputs"] = geometry_encoder_inputs
                # All data in a batch should have the same tag
                if "tag" in instances[0]:
                    assert len(set([instance.get("tag", "2d") for instance in instances])) == 1, \
                        "all data in a batch should have the same tag"
                    batch["tag"] = instances[0].get("tag", "2d")
        
        # VG-LLM specific: Handle world_coords
        if "world_coords" in instances[0] and instances[0]["world_coords"] is not None:
            world_coords_list = []
            for instance in instances:
                if "world_coords" in instance and instance["world_coords"] is not None:
                    world_coords_list.append(instance["world_coords"])
                else:
                    world_coords_list.append(None)
            # Only add to batch if all instances have world_coords
            if all(wc is not None for wc in world_coords_list):
                batch["world_coords"] = world_coords_list
        
        # VG-LLM specific: Handle objects (bbox)
        if "objects" in instances[0]:
            objects_list = []
            for instance in instances:
                if "objects" in instance and instance["objects"] is not None:
                    objects_list.append(instance["objects"])
                else:
                    objects_list.append(None)
            # Add to batch even if some instances don't have objects
            batch["objects"] = objects_list
        
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
