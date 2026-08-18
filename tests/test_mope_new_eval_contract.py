import ast
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn
from PIL import Image

from model.mope_new_encoder import (
    images_to_mope_new_tensor,
    load_saved_eval_components,
    load_annotation_for_mope_new,
    select_ordered_images,
)


@pytest.mark.parametrize(
    ("script_name", "experiment_name"),
    [
        ("eval_e00b_mope_new_vsibench.sh", "e00b_mope_new_projector_only_4b"),
        ("eval_e02c_mope_new_vsibench.sh", "e02c_mope_new_crossattn_joint_4b"),
        ("eval_e03a_mope_new_vsibench.sh", "e03a_mope_new_crossattn_two_stage_4b"),
    ],
)
def test_eval_dry_run_uses_server_roots_and_timestamped_log(script_name, experiment_name):
    root = Path.cwd()
    env = os.environ.copy()
    env.update(
        {
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": "/contract/output",
            "SPACE_LOG_ROOT": "/contract/logs",
            "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
            "DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval" / script_name)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"checkpoint=/contract/output/train/{experiment_name}" in result.stdout
    assert f"output=/contract/output/eval/vsibench/{experiment_name}" in result.stdout
    assert f"Log=/contract/logs/eval/{experiment_name}_vsibench_" in result.stdout


def test_shared_preprocessing_contract_for_ordered_images(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (320, 240), color=(index * 20, 0, 0)).save(path)
        paths.append(path.name)
    annotation = {"data_path": str(tmp_path), "image": paths}
    actual = load_annotation_for_mope_new(annotation)
    images = [Image.open(tmp_path / item).convert("RGB") for item in paths]
    expected = images_to_mope_new_tensor(images)
    assert actual.shape == (3, 16, 224, 224)
    assert torch.equal(actual, expected)
    assert len(select_ordered_images(images, 16)) == 16


def test_empty_or_unreadable_images_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_annotation_for_mope_new({"image": []})
    with pytest.raises(RuntimeError, match="cannot read"):
        load_annotation_for_mope_new({"data_path": str(tmp_path), "image": ["missing.png"]})


def test_new_eval_module_is_syntactically_importable_contract():
    path = Path("src/eval/models/qwen3_vl_mope_new.py")
    tree = ast.parse(path.read_text())
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    assert "Qwen3VLMoPENewCrossAttn" in classes


class TinyEncoder(nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        self.register_buffer("contract", contract.clone())


class TinyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.out_proj = nn.Linear(2, 2)


def save_eval_state(root, encoder, projector, include_encoder=True, include_projector=True):
    state = {}
    if include_encoder:
        state.update({f"model._mope_encoder.{k}": v for k, v in encoder.state_dict().items()})
    if include_projector:
        state.update({f"model._mope_projector.{k}": v for k, v in projector.state_dict().items()})
    torch.save(state, root / "pytorch_model.bin")


def test_eval_components_load_strictly_and_reject_contract_mismatch(tmp_path, monkeypatch):
    contract = torch.tensor([1, 16, 4, 224, 0])
    encoder, projector = TinyEncoder(contract), TinyProjector()
    monkeypatch.setattr(
        "model.mope_new_encoder.PROJECTOR_KEYS", tuple(projector.state_dict().keys())
    )
    save_eval_state(tmp_path, encoder, projector)
    load_saved_eval_components(encoder, projector, tmp_path, contract)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        load_saved_eval_components(encoder, projector, tmp_path, torch.tensor([1, 16, 4, 224, 1]))


@pytest.mark.parametrize("missing", ["encoder", "projector"])
def test_eval_missing_components_fail_instead_of_guide_fallback(tmp_path, missing, monkeypatch):
    contract = torch.tensor([1, 16, 4, 224, 0])
    encoder, projector = TinyEncoder(contract), TinyProjector()
    monkeypatch.setattr(
        "model.mope_new_encoder.PROJECTOR_KEYS", tuple(projector.state_dict().keys())
    )
    save_eval_state(
        tmp_path, encoder, projector,
        include_encoder=missing != "encoder", include_projector=missing != "projector",
    )
    with pytest.raises(RuntimeError, match=f"no {missing} weights"):
        load_saved_eval_components(encoder, projector, tmp_path, contract)
