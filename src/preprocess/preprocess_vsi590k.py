"""
preprocess_vsi590k.py
将 VSI-590K 数据集转换为 VG-LLM 两种训练格式：
  1. LLaVA-Video 格式：<video> token，走 Qwen2.5-VL video 路径（不过 VGGT）
  2. SPAR-style 格式：多个 <image> token，走图像路径（过 VGGT + MoPE）

数据集来源：https://huggingface.co/datasets/nyu-visionx/VSI-590K

用法：
  python preprocess_vsi590k.py \
      --video_dir /path/to/videos \
      --output_dir /path/to/output \
      --num_samples 10000 \
      --num_frames 8 \
      --seed 42

输出：
  {output_dir}/
    vsi590k_video_{N}k.json        # LLaVA-Video 格式
    vsi590k_spar_{N}k.json         # SPAR-style 格式
    frames/{dataset}/{scene}/      # 提取的帧（SPAR 格式用）
"""

import argparse
import json
import os
import random
from pathlib import Path

import cv2
from datasets import load_dataset
from tqdm import tqdm


# ── Task type 分类（来自 VSI-Bench utils.py）─────────────────────────────────
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
ALL_QUESTION_TYPES = MCA_QUESTION_TYPES + NA_QUESTION_TYPES


# ── 问题文本构建 ──────────────────────────────────────────────────────────────

def build_question_text(doc: dict) -> str:
    """根据 task type 构造带后缀的问题文本（与 VSI-Bench 评测格式保持一致）。"""
    question = doc["question"]
    question_type = doc["question_type"]

    if question_type in MCA_QUESTION_TYPES:
        options = doc.get("options", [])
        options_str = "Options:\n" + "\n".join(options) if options else ""
        post_prompt = "Answer with the option's letter from the given choices directly."
        parts = ["These are frames of a video.", question]
        if options_str:
            parts.append(options_str)
        parts.append(post_prompt)
        return "\n".join(parts)
    else:  # NA
        post_prompt = "Please answer the question using a single word or phrase."
        return "\n".join(["These are frames of a video.", question, post_prompt])


# ── 视频帧提取 ────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int, out_dir: str) -> list[str]:
    """
    从 video_path 均匀采样 num_frames 帧，保存为 JPEG，返回帧路径列表。
    若帧已存在则直接复用，不重复提取。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已提取
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= num_frames:
        return [str(p) for p in existing[:num_frames]]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"视频帧数为 0：{video_path}")

    # 均匀采样帧索引
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        indices = [int(step * i + step / 2) for i in range(num_frames)]

    saved_paths = []
    for rank, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = out_dir / f"frame_{rank:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_paths.append(str(frame_path))

    cap.release()
    return saved_paths


# ── 单条样本转换 ──────────────────────────────────────────────────────────────

def to_llava_video_format(doc: dict, video_rel_path: str, sample_id: str) -> dict:
    """
    LLaVA-Video 格式：
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "<video>\n{question}"},
            {"from": "gpt", "value": "{answer}"}
        ],
        "data_source": "vsibench",
        "video": "videos/{dataset}/{scene_name}.mp4",
        "question_type": "...",
        "metadata": {...}
    }
    """
    question_text = build_question_text(doc)
    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"<video>\n{question_text}"},
            {"from": "gpt", "value": str(doc["ground_truth"])},
        ],
        "data_source": "vsibench",
        "video": video_rel_path,
        "question_type": doc["question_type"],
        "metadata": {
            "scene_name": doc.get("scene_name", ""),
            "dataset": doc.get("dataset", ""),
        },
    }


def to_spar_format(
    doc: dict,
    frame_rel_paths: list[str],
    sample_id: str,
) -> dict:
    """
    SPAR-style 格式（多帧图像路径，过 VGGT geometry encoder）：
    {
        "id": "...",
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\n...\n{question}"},
            {"from": "gpt", "value": "{answer}"}
        ],
        "images": ["frames/...", ...],
        "question_type": "...",
        "spar_info": "{...}"
    }
    """
    n = len(frame_rel_paths)
    image_tokens = "\n".join(["<image>"] * n)
    question_text = build_question_text(doc)

    spar_info = json.dumps({
        "question_type": doc["question_type"],
        "scene_name": doc.get("scene_name", ""),
        "dataset": doc.get("dataset", ""),
    })

    return {
        "id": sample_id,
        "conversations": [
            {"from": "human", "value": f"{image_tokens}\n{question_text}"},
            {"from": "gpt", "value": str(doc["ground_truth"])},
        ],
        "images": frame_rel_paths,
        "question_type": doc["question_type"],
        "spar_info": spar_info,
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VSI-590K → VG-LLM 两种训练格式")
    p.add_argument(
        "--hf_dataset",
        default="nyu-visionx/VSI-590K",
        help="HuggingFace 数据集名称或本地路径（默认: nyu-visionx/VSI-590K）",
    )
    p.add_argument(
        "--hf_split",
        default="train",
        help="数据集 split（默认: train）",
    )
    p.add_argument(
        "--video_dir",
        required=True,
        help="本地视频根目录，结构为 {video_dir}/{dataset}/{scene_name}.mp4",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="输出目录（JSON 文件 + 帧目录）",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="采样数量，0 = 使用全量（默认: 0）",
    )
    p.add_argument(
        "--num_frames",
        type=int,
        default=8,
        help="SPAR 格式每个样本提取的帧数（默认: 8）",
    )
    p.add_argument(
        "--question_types",
        nargs="*",
        default=None,
        help="只处理指定 task type（默认: 全部 10 种）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    p.add_argument(
        "--skip_frame_extraction",
        action="store_true",
        help="跳过帧提取（假设帧已存在，仅重新生成 JSON）",
    )
    p.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HuggingFace 缓存目录（可选）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"

    # ── 1. 加载数据集 ─────────────────────────────────────────────────────────
    print(f"加载数据集：{args.hf_dataset} [{args.hf_split}] ...")
    dataset = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        cache_dir=args.hf_cache_dir,
    )

    # ── 2. 过滤 task type ─────────────────────────────────────────────────────
    question_types = args.question_types or ALL_QUESTION_TYPES
    print(f"使用 task types: {question_types}")
    dataset = dataset.filter(
        lambda doc: doc["question_type"] in question_types,
        desc="过滤 task type",
    )

    # ── 3. 采样 ───────────────────────────────────────────────────────────────
    total = len(dataset)
    if args.num_samples > 0 and args.num_samples < total:
        indices = random.sample(range(total), args.num_samples)
        indices.sort()
        dataset = dataset.select(indices)
        print(f"采样 {args.num_samples} / {total} 条")
    else:
        print(f"使用全量数据：{total} 条")

    # ── 4. 确定输出文件名 ──────────────────────────────────────────────────────
    n = len(dataset)
    suffix = f"{n // 1000}k" if n >= 1000 else str(n)
    video_json_path = output_dir / f"vsi590k_video_{suffix}.json"
    spar_json_path = output_dir / f"vsi590k_spar_{suffix}.json"

    # ── 5. 逐条处理 ───────────────────────────────────────────────────────────
    video_records = []
    spar_records = []
    skipped = 0

    for i, doc in enumerate(tqdm(dataset, desc="处理样本")):
        scene_name = doc.get("scene_name", f"scene_{i:06d}")
        dataset_name = doc.get("dataset", "vsibench")
        sample_id = f"vsi_{dataset_name}_{scene_name}_{i}"

        # 视频路径（实际磁盘结构为三层：video_dir/dataset/scene_name/<camera>.mp4）
        _scene_dir = Path(args.video_dir) / dataset_name / scene_name
        _preferred = _scene_dir / "raw_navigation_camera__0.mp4"
        if _preferred.exists():
            video_abs = _preferred
            video_rel = f"videos/{dataset_name}/{scene_name}/raw_navigation_camera__0.mp4"
        elif _scene_dir.is_dir():
            _candidates = sorted(_scene_dir.glob("*.mp4"))
            if _candidates:
                video_abs = _candidates[0]
                video_rel = f"videos/{dataset_name}/{scene_name}/{_candidates[0].name}"
            else:
                skipped += 1
                if skipped <= 5:
                    print(f"  ⚠️  场景目录无 .mp4，跳过：{_scene_dir}")
                continue
        else:
            skipped += 1
            if skipped <= 5:
                print(f"  ⚠️  视频不存在，跳过：{_scene_dir}")
            continue

        # ── LLaVA-Video 格式 ──────────────────────────────────────────────────
        video_records.append(
            to_llava_video_format(doc, video_rel, sample_id + "_video")
        )

        # ── SPAR-style 格式：提取帧 ───────────────────────────────────────────
        frame_out_dir = frames_root / dataset_name / scene_name
        frame_rel_prefix = f"frames/{dataset_name}/{scene_name}"

        if args.skip_frame_extraction:
            frame_paths = sorted(frame_out_dir.glob("frame_*.jpg"))
            frame_rel_paths = [
                f"{frame_rel_prefix}/{p.name}" for p in frame_paths[:args.num_frames]
            ]
        else:
            try:
                abs_paths = extract_frames(
                    str(video_abs), args.num_frames, str(frame_out_dir)
                )
                frame_rel_paths = [
                    f"{frame_rel_prefix}/{Path(p).name}" for p in abs_paths
                ]
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  ⚠️  帧提取失败，跳过：{e}")
                continue

        if not frame_rel_paths:
            skipped += 1
            continue

        spar_records.append(
            to_spar_format(doc, frame_rel_paths, sample_id + "_spar")
        )

    # ── 6. 保存 JSON ──────────────────────────────────────────────────────────
    with open(video_json_path, "w", encoding="utf-8") as f:
        json.dump(video_records, f, ensure_ascii=False, indent=2)

    with open(spar_json_path, "w", encoding="utf-8") as f:
        json.dump(spar_records, f, ensure_ascii=False, indent=2)

    # ── 7. 统计报告 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"✅ 完成！共处理 {len(dataset)} 条，跳过 {skipped} 条")
    print(f"   LLaVA-Video 格式：{len(video_records)} 条 → {video_json_path}")
    print(f"   SPAR-style  格式：{len(spar_records)} 条 → {spar_json_path}")

    # 按 task type 统计
    from collections import Counter
    video_type_counts = Counter(r["question_type"] for r in video_records)
    print("\n按 task type 分布（LLaVA-Video 格式，SPAR 同）：")
    for qt in ALL_QUESTION_TYPES:
        count = video_type_counts.get(qt, 0)
        print(f"  {qt:<40s}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
