import os
import re
import string
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "scanqa.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
_safe_yaml = yaml.safe_load("".join(safe_data))

# media_dir: env SCANQA_VIDEO_ROOT 最优先，其次 yaml metadata，最后 HF 缓存兜底
_metadata = _safe_yaml.get("metadata", {})
if isinstance(_metadata, list):
    _metadata = _metadata[0] if _metadata else {}
_env_media_dir = os.getenv("SCANQA_VIDEO_ROOT", None)
_yaml_media_dir = _metadata.get("media_dir", None)
_media_dir = _env_media_dir or _yaml_media_dir
if _media_dir and os.path.isdir(_media_dir):
    _video_cache_dir = _media_dir
else:
    _cache_name = _safe_yaml.get("dataset_kwargs", {}).get("cache_dir", "scanqa")
    _video_cache_dir = os.path.join(base_cache_dir, _cache_name)


def scanqa_doc_to_visual(doc):
    video_path = os.path.join(_video_cache_dir, doc["scene_id"] + ".mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video path: {video_path} does not exist.")
    return [video_path]


def scanqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video showing an indoor scene."
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") or "Please answer the question using a single word or short phrase."
    return "\n".join([pre_prompt, question, post_prompt])


_ARTICLES = {"a", "an", "the"}


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _ARTICLES]
    return " ".join(tokens)


def scanqa_process_results(doc, results):
    pred = results[0]
    doc["prediction"] = pred
    pred_norm = _normalize_answer(pred)
    answers_norm = [_normalize_answer(a) for a in doc["answers"]]

    doc["exact_match"] = 1.0 if pred_norm in answers_norm else 0.0
    doc["exact_match_refined"] = (
        1.0
        if pred_norm and any(pred_norm in a or a in pred_norm for a in answers_norm if a)
        else 0.0
    )
    return {"scanqa_score": doc}


def scanqa_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {
        "exact_match": results["exact_match"].mean(),
        "exact_match_refined": results["exact_match_refined"].mean(),
    }
    output["overall"] = output["exact_match"]
    eval_logger.info(f"Evaluation results: {output}")
    return output
