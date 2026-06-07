"""
VLM4D task utils — forked from vsibench/utils.py MCA branch.

VLM4D is a 4-way multiple-choice benchmark (HF `shijiezhou/VLM4D`). Scoring is
pure letter matching: fuzzy-extract the first option letter from the model output
and exact-match it against the ground-truth letter. NO LLM judge / API call.

Per-sample return structure mirrors vsibench_process_results (whole-doc copy +
prediction + accuracy), aligned with vsibench so the verification-① per-question
analysis pipeline can read these samples. NOTE, however, that
``src/analysis/oracle_route_headroom.py`` defaults to VSI's wrapper key
(``vsibench_score``) and VSI's question_type families (MCA/NA). To reuse it on
VLM4D, pass ``--score-key vlm4d_score``; VLM4D's single question_type
("multiple-choice") is then handled as its own flat MCA-style family
(see oracle_route_headroom.py改动 2).

PENDING[D-11]: 论文2 评测范围基本定（VSI-Bench 锚点 + VLM4D 动态主场），D-11 形式上仍 OPEN。

字段名/文件结构 VERIFIED 2026-06-06（数据已下载，标注 /data2/wlx/data/VLM4D/QA/real_mc.json
是 JSON 数组，1371 条）。所有依赖确切字段名处用模块级常量集中可改。
"""

import os
from pathlib import Path

import datasets
import pandas as pd
import yaml
from loguru import logger as eval_logger

# -----------------------------------------------------------------------------
# 字段名常量
# VERIFIED 2026-06-06: 真实标注 /data2/wlx/data/VLM4D/QA/real_mc.json（JSON 数组）。
#   每条: id / question / choices(dict A-D) / answer(选项全文) / question_type / video
#   - answer 是正确选项的全文（不是字母），需在 choices 里反查字母作 GT。
#   - video 是完整 HF URL，本地路径 = root + URL 中 resolve/main/ 之后的部分。
#   - question_type 恒为 "multiple-choice"（无运动语义子类），不再用于聚合分组。
# -----------------------------------------------------------------------------
FIELD_VIDEO = "video"            # 完整 HF URL（resolve/main/ 之后为本地相对路径）
FIELD_QUESTION = "question"      # 问题文本
FIELD_QUESTION_TYPE = "question_type"  # 恒为 "multiple-choice"，无语义子类
FIELD_CHOICES = "choices"        # dict {"A": .., "B": .., "C": .., "D": ..}
FIELD_ANSWER = "answer"          # ground-truth 选项全文（需反查 choices 得字母）

# VERIFIED 2026-06-06: question_type 实际只有一个取值 "multiple-choice"，
# 按它分组聚合会退化成单桶，故已废弃按 question_type 的分组（改为按视频来源 source）。

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


def _video_rel_path(url):
    """从完整 HF URL 取本地相对路径。
    VERIFIED 2026-06-06: video 字段是完整 URL，如
      https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/videos_real/davis/aerobatics.mp4
    本地相对路径 = resolve/main/ 之后的部分（videos_real/davis/aerobatics.mp4）。
    若 URL 不含该子串，兜底用 basename。
    """
    url = str(url)
    if "resolve/main/" in url:
        return url.split("resolve/main/")[-1]
    return os.path.basename(url)


def vlm4d_doc_to_visual(doc):
    # 参照 vsibench_doc_to_visual: media_dir + 视频相对路径。
    # VERIFIED 2026-06-06: doc[FIELD_VIDEO] 是完整 HF URL，需先转成本地相对路径。
    rel = _video_rel_path(doc[FIELD_VIDEO])
    video_path = os.path.join(_video_cache_dir, rel)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist.")
    return [video_path]


def _answer_to_letter(doc):
    """把 answer 全文映射为选项字母。
    VERIFIED 2026-06-06: answer 是正确选项的全文，需在 choices(dict) 里反查
    value（去空白）等于 answer（去空白）的那个 key 作为 GT 字母。
    找不到则返回原 answer 字符串（兜底）并 warning。
    """
    choices = doc[FIELD_CHOICES]
    answer = str(doc[FIELD_ANSWER]).strip()
    if isinstance(choices, dict):
        for k, v in choices.items():
            if str(v).strip() == answer:
                return k
    eval_logger.warning(
        f"Could not map answer to a choice letter (answer={doc.get(FIELD_ANSWER)!r}, "
        f"choices={choices!r}); falling back to raw answer string."
    )
    return answer


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


def vlm4d_doc_to_target(doc):
    # 供 yaml 的 doc_to_target 用（!function 引用）。返回 GT 选项字母。
    # VERIFIED 2026-06-06: answer 是全文，需反查 choices 得字母。
    return _answer_to_letter(doc)


def vlm4d_process_results(doc, results):
    """对齐 vsibench_process_results 的返回结构：
    整条 doc 拷贝 + prediction + accuracy(0/1)，以便复用验证①式逐题分析。
    纯字母匹配，无 judge。
    """
    doc["prediction"] = results[0]
    pred_letter = fuzzy_matching(doc["prediction"])
    # VERIFIED 2026-06-06: answer 是全文，反查 choices 得 GT 字母。
    target_letter = _answer_to_letter(doc)
    # 派生 source 字段（视频文件所在目录名：real 区分 davis/ego4d/youtube-vos，
    # synthetic 为 videos_synthetic），供 aggregate 做按来源的 breakdown。
    doc["source"] = os.path.basename(os.path.dirname(_video_rel_path(doc[FIELD_VIDEO])))
    for key, value in METRICS_FOR_MCA.items():
        try:
            doc[key] = eval(value)(pred_letter, target_letter)
        except (TypeError, ValueError):
            doc[key] = WORST_CASE_FOR_METRICS[key]
    return {"vlm4d_score": doc}


def vlm4d_aggregate_results(results):
    # VERIFIED 2026-06-06: question_type 恒为 "multiple-choice"，按它分组会退化。
    # 改为: overall 用全体样本均值（micro），并按视频来源 source 做 breakdown。
    results = pd.DataFrame(results)

    output = {}
    # overall = 全体样本 accuracy 均值（micro）。
    output["overall"] = results["accuracy"].mean()

    # 按视频来源 breakdown。
    for source, source_indexes in results.groupby("source").groups.items():
        per_source = results.iloc[source_indexes]
        output[f"{source}_accuracy"] = per_source["accuracy"].mean()

    eval_logger.info(f"Evaluation results: {output}")
    return output
