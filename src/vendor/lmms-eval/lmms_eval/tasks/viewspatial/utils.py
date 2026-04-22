import os
import re
from pathlib import Path

import yaml
from PIL import Image

with open(Path(__file__).parent / "viewspatial.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
_safe_yaml = yaml.safe_load("".join(safe_data))
_media_dir = _safe_yaml.get("metadata", {}).get("media_dir", "")


def viewspatial_doc_to_text(doc):
    question = doc["question"]
    choices = doc["choices"]

    question_text = f"Question: {question}\n"
    choices_text = f"Choices: {choices}\n"
    post_prompt = "Reply only to the corresponding option.\nAnswer:"

    prompt = question_text + choices_text + post_prompt
    return prompt


def _strip_dataset_prefix(path: str) -> str:
    """Strip the 'ViewSpatial-Bench/' prefix that the upstream JSON uses."""
    prefix = "ViewSpatial-Bench/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def viewspatial_doc_to_visual(doc):
    image_paths = doc["image_path"]
    images = [
        Image.open(
            os.path.join(_media_dir, _strip_dataset_prefix(p))
        ).convert("RGB")
        for p in image_paths
    ]
    return images


def extract_option(text):
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def viewspatial_process_results(doc, results):
    """Processes the model's output for a single viewspatial document."""
    # extract grounded answer
    grounded_output = doc["answer"]
    grounded_option = extract_option(grounded_output)
    # eval_logger.info(f"Grounded answer: {grounded_output}")

    # extract predicted answer
    pred = results[0]
    pred = pred.split("\n")[-1]
    pred_answer = extract_option(pred)
    # eval_logger.info(f"Predicted answer: {pred_answer}")

    score = 1.0 if pred_answer == grounded_option else 0.0
    # eval_logger.info(f"Score: {score}")

    return {"overall_accuracy": {"score": score}}


def viewspatial_aggregate_results(results):
    """Aggregates the 'overall_accuracy' results.

    Calculates the mean score from a list of result dictionaries.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key (e.g., [{'score': 1.0}, ...]).

    Returns:
        float: The average score (mean accuracy).
    """
    # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res["score"]

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score