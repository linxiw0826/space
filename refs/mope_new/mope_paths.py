"""Portable path configuration for code, assets, and datasets."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ASSETS_ROOT = Path(
    os.environ.get("MOPE_ASSETS_ROOT", REPO_ROOT.parent / "mope-jepa-assets")
).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("MOPE_DATA_ROOT", REPO_ROOT.parent)).expanduser().resolve()

JEPA_OUTPUT_ROOT = Path(
    os.environ.get("MOPE_JEPA_OUTPUT_ROOT", ASSETS_ROOT / "jepa_checkpoints")
).expanduser().resolve()
SFT_OUTPUT_ROOT = Path(
    os.environ.get("MOPE_SFT_OUTPUT_ROOT", ASSETS_ROOT / "sft_checkpoints")
).expanduser().resolve()
PRETRAINED_ROOT = Path(
    os.environ.get("MOPE_PRETRAINED_ROOT", ASSETS_ROOT / "pretrained")
).expanduser().resolve()
QWEN_MODEL_ROOT = Path(
    os.environ.get("MOPE_QWEN_MODEL_ROOT", ASSETS_ROOT / "models" / "Qwen3-VL-2B-Instruct")
).expanduser().resolve()


def require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_dir(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
