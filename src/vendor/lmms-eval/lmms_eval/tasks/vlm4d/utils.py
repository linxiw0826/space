"""
VLM4D task utils — forked from vsibench/utils.py MCA branch.

VLM4D is a 4-way multiple-choice benchmark (HF `shijiezhou/VLM4D`). Scoring is
pure letter matching: fuzzy-extract the first option letter from the model output
and exact-match it against the ground-truth letter. NO LLM judge / API call.

Per-sample return structure mirrors vsibench_process_results (whole-doc copy +
prediction + accuracy) so the verification-① per-question analysis pipeline can
be reused as-is.

PENDING[D-11]: 论文2 评测范围基本定（VSI-Bench 锚点 + VLM4D 动态主场），D-11 形式上仍 OPEN。

字段名/文件结构按 HF README 推断，下载后可能微调。所有依赖确切字段名处用
模块级常量 + TODO[verify-after-download] 注释，集中可改。
"""

import os
from pathlib import Path

import datasets
import pandas as pd
import yaml
from loguru import logger as eval_logger

# -----------------------------------------------------------------------------
# 字段名常量（下载后若 README 推断与实际不符，只改这里）
# TODO[verify-after-download]: 确认 VLM4D 实际字段名。
#   README 推断: id / video / question_type / question / choices(dict A-D) / answer(letter)
# -----------------------------------------------------------------------------
FIELD_VIDEO = "video"            # 视频文件名或相对路径
FIELD_QUESTION = "question"      # 问题文本
FIELD_QUESTION_TYPE = "question_type"  # 运动类别: translational/rotational/action/counting/false-positive
FIELD_CHOICES = "choices"        # dict {"A": .., "B": .., "C": .., "D": ..}
FIELD_ANSWER = "answer"          # ground-truth 选项字母, 如 "A"

# question_type 取值（仅用于日志/兜底；聚合按数据实际出现的类别动态分组）
# TODO[verify-after-download]: 确认 question_type 取值集合。
VLM4D_QUESTION_TYPES = [
    "translational",
    "rotational",
    "action",
    "counting",
    "false-positive",
]

# MCA 打分：纯字母 exact_match（无 judge）
METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
}


# -----------------------------------------------------------------------------
# media_dir 解析：优先 env VLM4D_VIDEO_ROOT，其次 yaml metadata.media_dir，
# 兜底 HF_HOME / cache。参照 vsibench/utils.py 的 media_dir 读取逻辑。
# -----------------------------------------------------------------------------
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vlm4d.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
_safe_yaml = yaml.safe_load("".join(safe_data))

_metadata = _safe_yaml.get("metadata", {})
if isinstance(_metadata, list):
    _metadata = _metadata[0] if _metadata else {}

# env 优先（与 eval 脚本的 VLM4D_VIDEO_ROOT 对齐），yaml metadata 次之
_env_media_dir = os.getenv("VLM4D_VIDEO_ROOT", None)
_yaml_media_dir = _metadata.get("media_dir", None)
_media_dir = _env_media_dir or _yaml_media_dir

if _media_dir and os.path.isdir(_media_dir):
    _video_cache_dir = _media_dir
else:
    _cache_name = _safe_yaml.get("dataset_kwargs", {}).get("cache_dir", "vlm4d")
    _video_cache_dir = os.path.join(base_cache_dir, _cache_name)


def vlm4d_doc_to_visual(doc):
    # 参照 vsibench_doc_to_visual: media_dir + 视频相对路径。
    # TODO[verify-after-download]: 确认 doc[FIELD_VIDEO] 是文件名还是含子目录的相对路径。
    video_path = os.path.join(_video_cache_dir, doc[FIELD_VIDEO])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist.")
    return [video_path]


def _format_choices(choices):
    """把 choices dict {"A":..,"B":..} 展开成 "A. xxx\nB. xxx\n..." 文本。

    VLM4D 的 choices 是 dict（与 VSI 的 options list 不同）。按字母键排序展开，
    保证 A/B/C/D 顺序稳定。兼容 choices 已是 list 的情况（下载后若实际是 list）。
    """
    # TODO[verify-after-download]: 确认 choices 为 dict {"A":..}；若为 list 走兼容分支。
    if isinstance(choices, dict):
        lines = [f"{k}. {choices[k]}" for k in sorted(choices.keys())]
    elif isinstance(choices, (list, tuple)):
        # 兼容: 若每项已含 "A. xxx" 前缀则原样用；否则补字母。
        lines = []
        for idx, c in enumerate(choices):
            c_str = str(c)
            if len(c_str) >= 2 and c_str[0].isalpha() and c_str[1] in ".)":
                lines.append(c_str)
            else:
                lines.append(f"{chr(ord('A') + idx)}. {c_str}")
    else:
        raise ValueError(f"Unexpected choices type: {type(choices)}")
    return "\n".join(lines)


def vlm4d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # 复用 vsibench MCA prompt 风格: pre_prompt + question + Options + post_prompt。
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    question = doc[FIELD_QUESTION]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    options = "Options:\n" + _format_choices(doc[FIELD_CHOICES])
    post_prompt = (
        lmms_eval_specific_kwargs.get("mca_post_prompt", "")
        or "Answer with the option's letter from the given choices directly."
    )
    return "\n".join([pre_prompt, question, options, post_prompt])


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # 同 vsibench: 默认不 shuffle（doc_id = 数据集行号, 便于逐题对齐分析）。
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    # 复用 vsibench: 取首 token、去尾点。提取模型输出的首个选项字母。
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    # 复用 vsibench: 大小写不敏感字母比对。
    return 1.0 if pred.lower() == target.lower() else 0.0


def vlm4d_process_results(doc, results):
    """对齐 vsibench_process_results 的返回结构：
    整条 doc 拷贝 + prediction + accuracy(0/1)，以便复用验证①式逐题分析。
    纯字母匹配，无 judge。
    """
    doc["prediction"] = results[0]
    pred_letter = fuzzy_matching(doc["prediction"])
    target_letter = str(doc[FIELD_ANSWER]).strip()
    for key, value in METRICS_FOR_MCA.items():
        try:
            doc[key] = eval(value)(pred_letter, target_letter)
        except (TypeError, ValueError):
            doc[key] = WORST_CASE_FOR_METRICS[key]
    return {"vlm4d_score": doc}


def vlm4d_aggregate_results(results):
    # 同 vsibench aggregate 风格: 按 question_type 分组算 accuracy, 再算 overall。
    results = pd.DataFrame(results)

    output = {}
    for question_type, question_type_indexes in results.groupby(FIELD_QUESTION_TYPE).groups.items():
        per_question_type = results.iloc[question_type_indexes]
        for metric in METRICS_FOR_MCA.keys():
            output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    # overall = 各 question_type accuracy 的均值（与 vsibench overall 同风格：对子项取平均）。
    output["overall"] = sum([_ for _ in output.values()]) / len(output) if output else 0.0
    eval_logger.info(f"Evaluation results: {output}")
    return output
