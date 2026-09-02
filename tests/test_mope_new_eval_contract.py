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


def test_complete_hf_resolver_rejects_empty_weight_shard(tmp_path):
    root = Path.cwd()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}'
    )
    (checkpoint / "model-00001-of-00001.safetensors").touch()
    helper = root / "scripts/idea1_feature/_mope_new_eval_lib.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; mope_resolve_complete_hf_checkpoint "$2"', "bash", str(helper), str(checkpoint)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "No complete HF checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("script_name", "experiment_name"),
    [
        ("eval_e02c_mope_new_vsibench.sh", "e02c_mope_new_crossattn_joint_4b"),
        ("eval_e04a_mope_new_vsibench.sh", "e04a_mope_new_e01_projector_only_4b"),
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
    assert f"requested_checkpoint=/contract/output/train/{experiment_name}" in result.stdout
    assert f"Effective_checkpoint=/contract/output/train/{experiment_name}" in result.stdout
    assert f"output=/contract/output/eval/vsibench/{experiment_name}" in result.stdout
    assert f"Log=/contract/logs/eval/{experiment_name}_vsibench_" in result.stdout
    assert "--num_processes=4" in result.stdout
    assert "CUDA_VISIBLE_DEVICES" not in result.stderr


def test_vsibench_eval_promotes_exactly_two_flat_artifacts(tmp_path):
    root = Path.cwd()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    accelerate = fake_bin / "accelerate"
    accelerate.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
output=''
while (($#)); do
  if [[ "$1" == '--output_path' ]]; then output="$2"; shift 2; else shift; fi
done
mkdir -p "$output/train__fake"
printf '{"results": {"vsibench": 0.5}}\\n' > "$output/train__fake/20260822_results.json"
: > "$output/train__fake/20260822_samples_vsibench_mope_new.jsonl"
for ((i=0; i<5130; i++)); do
  printf '{"doc_id": %s}\\n' "$i" >> "$output/train__fake/20260822_samples_vsibench_mope_new.jsonl"
done
"""
    )
    accelerate.chmod(0o755)
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    experiment = "e02c_mope_new_crossattn_joint_4b"
    result_dir = output_root / "eval" / "vsibench" / experiment
    # An earlier valid result must remain until the replacement run succeeds.
    result_dir.mkdir(parents=True)
    (result_dir / "old_results.json").write_text("{}")
    checkpoint = output_root / "train" / experiment / "checkpoint-5000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model-00001-of-00001.safetensors").write_text("weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}'
    )
    incomplete_checkpoint = checkpoint.parent / "checkpoint-8000"
    incomplete_checkpoint.mkdir()
    (incomplete_checkpoint / "config.json").write_text("{}")
    (incomplete_checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"missing.safetensors"}}'
    )
    mope_checkpoint = tmp_path / "checkpoint-50.pth"
    mope_checkpoint.write_text("mope")
    annotation = tmp_path / "test.jsonl"
    annotation.write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": str(output_root),
            "SPACE_LOG_ROOT": str(log_root),
            "MOPE_NEW_CKPT": str(mope_checkpoint),
            "VSIBENCH_JSONL": str(annotation),
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval/eval_e02c_mope_new_vsibench.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"Effective_checkpoint={checkpoint}" in completed.stdout
    assert sorted(path.name for path in result_dir.iterdir()) == [
        f"{experiment}_results.json",
        f"{experiment}_samples.jsonl",
    ]
    assert not list((output_root / "eval" / "vsibench").glob(f"{experiment}.work.*"))


def test_failed_vsibench_eval_preserves_prior_results_and_cleans_workdir(tmp_path):
    root = Path.cwd()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    accelerate = fake_bin / "accelerate"
    accelerate.write_text("#!/usr/bin/env bash\nexit 9\n")
    accelerate.chmod(0o755)
    output_root = tmp_path / "output"
    experiment = "e02c_mope_new_crossattn_joint_4b"
    result_dir = output_root / "eval" / "vsibench" / experiment
    result_dir.mkdir(parents=True)
    old_result = result_dir / "old_results.json"
    old_result.write_text('{"overall": 0.4}')
    checkpoint = output_root / "train" / experiment / "checkpoint-7000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_text("weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model.safetensors"}}'
    )
    mope_checkpoint = tmp_path / "checkpoint-50.pth"
    mope_checkpoint.write_text("mope")
    annotation = tmp_path / "test.jsonl"
    annotation.write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": str(output_root),
            "SPACE_LOG_ROOT": str(tmp_path / "logs"),
            "MOPE_NEW_CKPT": str(mope_checkpoint),
            "VSIBENCH_JSONL": str(annotation),
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval/eval_e02c_mope_new_vsibench.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 9
    assert old_result.read_text() == '{"overall": 0.4}'
    assert not list((output_root / "eval" / "vsibench").glob(f"{experiment}.work.*"))


@pytest.mark.parametrize(
    ("script_name", "experiment_name"),
    [
        ("eval_e02c_mope_new_vlm4d.sh", "e02c_mope_new_crossattn_joint_4b"),
        ("eval_e04a_mope_new_vlm4d.sh", "e04a_mope_new_e01_projector_only_4b"),
    ],
)
def test_vlm4d_eval_dry_run_contract(script_name, experiment_name):
    root = Path.cwd()
    env = os.environ.copy()
    env.update(
        {
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": "/contract/output",
            "SPACE_LOG_ROOT": "/contract/logs",
            "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
            "DRY_RUN": "1",
            "CUDA_VISIBLE_DEVICES": "7",
            "NUM_PROCESSES": "1",
            "MAIN_PORT": "29604",
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
    assert f"requested_checkpoint=/contract/output/train/{experiment_name}" in result.stdout
    assert f"Effective_checkpoint=/contract/output/train/{experiment_name}" in result.stdout
    assert f"output=/contract/output/eval/vlm4d/{experiment_name}" in result.stdout
    assert f"Log=/contract/logs/eval/{experiment_name}_vlm4d_" in result.stdout
    assert "--model qwen3_vl_mope_new_crossattn" in result.stdout
    assert "--tasks vlm4d_real_mc_mope_new" in result.stdout
    assert "checkpoint-50.pth" in result.stdout
    assert "mope_all_frames=16" in result.stdout
    assert "mope_groups=4" in result.stdout
    assert "mope_frames_per_group=4" in result.stdout
    assert "mope_input_size=224" in result.stdout
    assert "mope_pool_mode=temporal" in result.stdout
    assert "--num_processes=1" in result.stdout
    assert "--main_process_port=29604" in result.stdout


@pytest.mark.parametrize(
    ("script_name", "benchmark", "port", "extra_output"),
    [
        (
            "smoke_e02c_mope_new_vsibench.sh",
            "vsibench",
            "29527",
            "Smoke coverage=all 10 question types total=10",
        ),
        (
            "smoke_e02c_mope_new_vlm4d.sh",
            "vlm4d",
            "29529",
            "Smoke coverage=all 3 video sources total=4",
        ),
        (
            "smoke_e04a_mope_new_vsibench.sh",
            "vsibench",
            "29527",
            "Smoke coverage=all 10 question types total=10",
        ),
        (
            "smoke_e04a_mope_new_vlm4d.sh",
            "vlm4d",
            "29529",
            "Smoke coverage=all 3 video sources total=4",
        ),
    ],
)
def test_final515k_smoke_dry_run_is_multigpu_and_isolated(
    script_name, benchmark, port, extra_output
):
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
    experiment = (
        "e04a_mope_new_e01_projector_only_4b"
        if "e04a" in script_name
        else "e02c_mope_new_crossattn_joint_4b"
    )
    assert "Mode=smoke" in result.stdout
    assert f"output=/contract/output/eval/smoke/{benchmark}/{experiment}" in result.stdout
    assert f"Log=/contract/logs/eval/smoke/{experiment}_{benchmark}_smoke_" in result.stdout
    assert "--num_processes=4" in result.stdout
    assert f"--main_process_port={port}" in result.stdout
    assert extra_output in result.stdout
    assert (
        f"Data_preflight_report=/contract/output/audit/mope_final515k_eval/"
        f"{benchmark}_data_preflight.json"
    ) in result.stdout
    assert " --limit " not in result.stdout


@pytest.mark.parametrize(
    ("script_name", "benchmark"),
    [
        ("smoke_e02c_mope_new_vsibench.sh", "vsibench"),
        ("smoke_e02c_mope_new_vlm4d.sh", "vlm4d"),
    ],
)
def test_final515k_smoke_refuses_formal_result_directory(script_name, benchmark):
    root = Path.cwd()
    experiment = "e02c_mope_new_crossattn_joint_4b"
    output_root = "/contract/output"
    formal = f"{output_root}/eval/{benchmark}/{experiment}"
    for unsafe_result in (
        formal,
        f"{output_root}/eval/{benchmark}",
        f"{formal}/child",
        f"{output_root}/unrelated-smoke",
    ):
        env = os.environ.copy()
        env.update(
            {
                "SPACE_ROOT": str(root),
                "SPACE_OUTPUT_ROOT": output_root,
                "SPACE_LOG_ROOT": "/contract/logs",
                "RESULTS_DIR": unsafe_result,
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
        )
        assert result.returncode == 2
        assert "Smoke output must be an isolated child" in result.stderr


def test_vlm4d_smoke_refuses_more_ranks_than_samples():
    root = Path.cwd()
    env = os.environ.copy()
    env.update(
        {
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": "/contract/output",
            "SPACE_LOG_ROOT": "/contract/logs",
            "MOPE_NEW_ALLOW_MISSING_ASSETS": "1",
            "DRY_RUN": "1",
            "NUM_PROCESSES": "5",
        }
    )
    result = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval/smoke_e02c_mope_new_vlm4d.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "VLM4D smoke requires NUM_PROCESSES <= 4; got 5" in result.stderr


def test_final515k_wrappers_preflight_source_root():
    for script in (
        "scripts/idea1_feature/_mope_new_eval_common.sh",
        "scripts/idea1_feature/_mope_new_vlm4d_eval_common.sh",
    ):
        source = Path(script).read_text()
        assert '[[ -d "${MOPE_NEW_SOURCE_ROOT}" ]]' in source
        assert "Missing MoPE-new source" in source


def test_eval_artifact_validation_rejects_partial_samples(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"doc_id": 0}\n')
    helper = Path.cwd() / "scripts/idea1_feature/_mope_new_eval_lib.sh"
    result = subprocess.run(
        [
            "bash", "-c",
            'source "$1"; mope_validate_jsonl_count "$2" 3',
            "bash", str(helper), str(samples),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "samples JSONL count mismatch: expected=3, actual=1" in result.stderr


def test_vlm4d_eval_resolves_checkpoint_and_promotes_two_flat_artifacts(tmp_path):
    root = Path.cwd()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    accelerate = fake_bin / "accelerate"
    accelerate.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
output=''
include_path=''
while (($#)); do
  if [[ "$1" == '--output_path' ]]; then
    output="$2"; shift 2
  elif [[ "$1" == '--include_path' ]]; then
    include_path="$2"; shift 2
  else
    shift
  fi
done
test -s "$include_path/utils.py"
test -s "$include_path/vlm4d.yaml"
test -s "$include_path/vlm4d_real_mc.yaml"
mkdir -p "$output/train__fake"
printf '{"results": {"vlm4d": 0.5}}\\n' > "$output/train__fake/20260901_results.json"
: > "$output/train__fake/20260901_samples_vlm4d_real_mc_mope_new.jsonl"
for ((i=0; i<1371; i++)); do
  printf '{"doc_id": %s}\\n' "$i" >> "$output/train__fake/20260901_samples_vlm4d_real_mc_mope_new.jsonl"
done
"""
    )
    accelerate.chmod(0o755)
    output_root = tmp_path / "output"
    log_root = tmp_path / "logs"
    experiment = "e02c_mope_new_crossattn_joint_4b"
    result_dir = output_root / "eval" / "vlm4d" / experiment
    result_dir.mkdir(parents=True)
    (result_dir / "old_results.json").write_text("{}")
    checkpoint = output_root / "train" / experiment / "checkpoint-7000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model-00001-of-00001.safetensors").write_text("weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}'
    )
    mope_checkpoint = tmp_path / "checkpoint-50.pth"
    mope_checkpoint.write_text("mope")
    source_root = tmp_path / "mope-source"
    source_root.mkdir()
    annotation = tmp_path / "real_mc.json"
    annotation.write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": str(output_root),
            "SPACE_LOG_ROOT": str(log_root),
            "MOPE_NEW_CKPT": str(mope_checkpoint),
            "MOPE_NEW_SOURCE_ROOT": str(source_root),
            "VLM4D_JSONL": str(annotation),
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval/eval_e02c_mope_new_vlm4d.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert f"Effective_checkpoint={checkpoint}" in completed.stdout
    assert sorted(path.name for path in result_dir.iterdir()) == [
        f"{experiment}_results.json",
        f"{experiment}_samples.jsonl",
    ]
    assert not list((output_root / "eval" / "vlm4d").glob(f"{experiment}.work.*"))


def test_failed_vlm4d_eval_preserves_prior_results_and_cleans_workdir(tmp_path):
    root = Path.cwd()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    accelerate = fake_bin / "accelerate"
    accelerate.write_text("#!/usr/bin/env bash\nexit 9\n")
    accelerate.chmod(0o755)
    output_root = tmp_path / "output"
    experiment = "e02c_mope_new_crossattn_joint_4b"
    result_dir = output_root / "eval" / "vlm4d" / experiment
    result_dir.mkdir(parents=True)
    old_result = result_dir / "old_results.json"
    old_result.write_text('{"overall": 0.4}')
    checkpoint = output_root / "train" / experiment
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_text("weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        '{"weight_map":{"x":"model.safetensors"}}'
    )
    mope_checkpoint = tmp_path / "checkpoint-50.pth"
    mope_checkpoint.write_text("mope")
    source_root = tmp_path / "mope-source"
    source_root.mkdir()
    annotation = tmp_path / "real_mc.json"
    annotation.write_text("{}\n")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SPACE_ROOT": str(root),
            "SPACE_OUTPUT_ROOT": str(output_root),
            "SPACE_LOG_ROOT": str(tmp_path / "logs"),
            "MOPE_NEW_CKPT": str(mope_checkpoint),
            "MOPE_NEW_SOURCE_ROOT": str(source_root),
            "VLM4D_JSONL": str(annotation),
        }
    )
    completed = subprocess.run(
        ["bash", str(root / "scripts/idea1_feature/eval/eval_e02c_mope_new_vlm4d.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 9
    assert old_result.read_text() == '{"overall": 0.4}'
    assert not list((output_root / "eval" / "vlm4d").glob(f"{experiment}.work.*"))


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
    assert len(select_ordered_images(images)) == 16


def test_mope_video_sidecar_takes_precedence_over_guide_images(monkeypatch):
    seen = {}

    def fake_loader(path, **kwargs):
        seen["path"] = str(path)
        seen.update(kwargs)
        return torch.zeros(3, 16, 224, 224)

    monkeypatch.setattr("model.mope_new_encoder.load_video_for_mope_new", fake_loader)
    result = load_annotation_for_mope_new({
        "data_path": "/dataset", "image": ["guide/frame0.jpg"],
        "mope_video": "videos/scene.mp4",
    })
    assert result.shape == (3, 16, 224, 224)
    assert seen == {
        "path": "/dataset/videos/scene.mp4", "groups": 4,
        "frames_per_group": 4, "input_size": 224,
    }


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
    loader_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_video_for_mope_new"
    ]
    assert len(loader_calls) == 1
    assert {keyword.arg for keyword in loader_calls[0].keywords} == {
        "groups", "frames_per_group", "input_size"
    }


def test_new_eval_uses_accelerate_unwrap_property():
    """The final515k plugin must not dereference attributes on a DDP wrapper."""
    path = Path("src/eval/models/qwen3_vl_mope_new.py")
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLMoPENewCrossAttn"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    source = ast.unparse(init)
    assert "base_model = self.model" in source
    assert "self._model.model" not in source
    assert "self._model.config" not in source
    assert "next(self._model.parameters())" not in source
    assert "inner._mope_encoder.eval()" in source
    assert "inner._mope_projector.eval()" in source


def test_final515k_eval_enables_fail_closed_sidecar_contract():
    new_tree = ast.parse(Path("src/eval/models/qwen3_vl_mope_new.py").read_text())
    strict_assignments = [
        node for node in ast.walk(new_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_mope_eval_fail_closed"
            for target in node.targets
        )
    ]
    assert len(strict_assignments) == 1
    assert isinstance(strict_assignments[0].value, ast.Constant)
    assert strict_assignments[0].value.value is True

    base_tree = ast.parse(Path("src/eval/models/qwen3_vl_mope.py").read_text())
    generate = next(
        node for node in ast.walk(base_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_until"
    )
    source = ast.unparse(generate)
    assert "strict MoPE eval could not extract the source video" in source
    assert "strict MoPE eval produced no MoPE frames" in source
    assert "inner = self.model.model" in source


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
