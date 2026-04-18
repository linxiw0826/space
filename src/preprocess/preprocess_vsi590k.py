"""
preprocess_vsi590k.py
将 VSI-590K 数据集转换为 VG-LLM 两种训练格式：
  1. LLaVA-Video 格式：<video> token，走 Qwen2.5-VL video 路径（不过 VGGT）
  2. SPAR-style 格式：多个 <image> token，走图像路径（过 VGGT + MoPE）

数据来源：本地 JSONL 标注文件（默认：{video_dir}/vsi_590k.jsonl）

用法：
  python preprocess_vsi590k.py \
      --video_dir /path/to/videos \
      --output_dir /path/to/output \
      --num_samples 10000 \
      --num_frames 8 \
      --seed 42

  # 可选：指定标注文件路径
  python preprocess_vsi590k.py \
      --video_dir /path/to/videos \
      --annotation_file /path/to/vsi_590k.jsonl \
      --output_dir /path/to/output

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

def to_llava_video_format(
    doc: dict,
    video_rel_path: str,
    sample_id: str,
    question_text: str,
) -> dict:
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
            "dataset": doc.get("dataset_name", ""),
        },
    }


def to_spar_format(
    doc: dict,
    frame_rel_paths: list[str],
    sample_id: str,
    question_text: str,
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

    spar_info = json.dumps({
        "question_type": doc["question_type"],
        "scene_name": doc.get("scene_name", ""),
        "dataset": doc.get("dataset_name", ""),
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
        "--annotation_file",
        default=None,
        help="本地 JSONL 标注文件路径（默认: {video_dir}/vsi_590k.jsonl）",
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
        help="只处理指定 task type（默认: 不过滤，使用全部类型）",
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
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = output_dir / "frames"

    # ── annotation_file 默认值：{video_dir}/vsi_590k.jsonl ───────────────────
    annotation_file = args.annotation_file
    if annotation_file is None:
        annotation_file = str(Path(args.video_dir) / "vsi_590k.jsonl")

    # ── 1. 加载本地 JSONL 标注文件 ────────────────────────────────────────────
    print(f"加载标注文件：{annotation_file} ...")
    dataset = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    print(f"共读取 {len(dataset)} 条记录")

    # ── 2. 过滤 task type（仅在 --question_types 指定时过滤）────────────────
    if args.question_types is not None:
        question_types = args.question_types
        print(f"按 task type 过滤: {question_types}")
        dataset = [doc for doc in dataset if doc["question_type"] in question_types]
        print(f"过滤后剩余：{len(dataset)} 条")
    else:
        print("未指定 --question_types，使用全部类型，不过滤")

    # ── 3. 采样 ───────────────────────────────────────────────────────────────
    total = len(dataset)
    if args.num_samples > 0 and args.num_samples < total:
        indices = random.sample(range(total), args.num_samples)
        indices.sort()
        dataset = [dataset[i] for i in indices]
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

    for i, raw_doc in enumerate(tqdm(dataset, desc="处理样本")):
        # 从 video 字段提取 dataset_name 和 scene_name
        # video 格式："{dataset_name}/{scene_name}.mp4"，例如 "scannet/scene0191_00.mp4"
        video_field = raw_doc.get("video", "")
        video_parts = video_field.split("/")
        if len(video_parts) >= 2:
            dataset_name = video_parts[0]
            scene_name = Path(video_parts[-1]).stem  # 去掉 .mp4
        else:
            dataset_name = "vsibench"
            scene_name = Path(video_field).stem if video_field else f"scene_{i:06d}"

        # 构建统一的 doc 结构，补充提取出的字段
        doc = dict(raw_doc)
        doc["dataset_name"] = dataset_name
        doc["scene_name"] = scene_name
        doc["ground_truth"] = raw_doc["conversations"][1]["value"]
        doc["question_type"] = raw_doc["question_type"]

        # 从 conversations[0]["value"] 去掉 "<image>\n" 前缀得到 question_text
        human_value = raw_doc["conversations"][0]["value"]
        if human_value.startswith("<image>\n"):
            question_text = human_value[len("<image>\n"):]
        else:
            question_text = human_value

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
                    print(f"  场景目录无 .mp4，跳过：{_scene_dir}")
                continue
        else:
            skipped += 1
            if skipped <= 5:
                print(f"  视频不存在，跳过：{_scene_dir}")
            continue

        # ── LLaVA-Video 格式 ──────────────────────────────────────────────────
        video_records.append(
            to_llava_video_format(doc, video_rel, sample_id + "_video", question_text)
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
                    print(f"  帧提取失败，跳过：{e}")
                continue

        if not frame_rel_paths:
            skipped += 1
            continue

        spar_records.append(
            to_spar_format(doc, frame_rel_paths, sample_id + "_spar", question_text)
        )

    # ── 6. 保存 JSON ──────────────────────────────────────────────────────────
    with open(video_json_path, "w", encoding="utf-8") as f:
        json.dump(video_records, f, ensure_ascii=False, indent=2)

    with open(spar_json_path, "w", encoding="utf-8") as f:
        json.dump(spar_records, f, ensure_ascii=False, indent=2)

    # ── 7. 统计报告 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"完成！共处理 {len(dataset)} 条，跳过 {skipped} 条")
    print(f"   LLaVA-Video 格式：{len(video_records)} 条 -> {video_json_path}")
    print(f"   SPAR-style  格式：{len(spar_records)} 条 -> {spar_json_path}")

    # 按 task type 统计
    from collections import Counter
    video_type_counts = Counter(r["question_type"] for r in video_records)
    print("\n按 task type 分布（LLaVA-Video 格式，SPAR 同）：")
    # 先打印已知类型
    for qt in ALL_QUESTION_TYPES:
        count = video_type_counts.get(qt, 0)
        if count > 0:
            print(f"  {qt:<40s}: {count}")
    # 打印不在 ALL_QUESTION_TYPES 中的类型（JSONL 可能有新类型名称）
    for qt, count in sorted(video_type_counts.items()):
        if qt not in ALL_QUESTION_TYPES:
            print(f"  {qt:<40s}: {count}  (not in ALL_QUESTION_TYPES)")
    print("=" * 60)


if __name__ == "__main__":
    main()
