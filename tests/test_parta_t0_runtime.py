import subprocess
import sys
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parta.canonical_data import PartASample
from parta.state_head import StateHeadConfig
from parta.state_loss import StateLossConfig, StateTargets
from parta.t0_runtime import PartAT0Collator, forward_visual_tap
from parta.training import attach_a1o_state_head, run_a1o_side_branch
from parta_data_contract import CANONICAL_CATEGORIES
from parta_data_contract import ContractError


def load_runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts/parta/run_t0_a.py"
    spec = importlib.util.spec_from_file_location("run_t0_a_test_module", script)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    return script, runner


def test_tensor_diagnostic_is_exact_across_chunks_without_full_float_copy(monkeypatch):
    _, runner = load_runner_module()
    monkeypatch.setattr(runner, "TENSOR_DIAGNOSTIC_CHUNK_ELEMENTS", 100_000)
    values = torch.arange(300_000, dtype=torch.float32) - 123.0
    diagnostic = runner._tensor_diagnostic(values.to(torch.bfloat16))
    reference = values.to(torch.bfloat16).float()
    assert diagnostic["shape"] == [300_000]
    assert diagnostic["finite"] is True
    assert diagnostic["nonfinite_count"] == 0
    assert diagnostic["min"] == float(reference.min())
    assert diagnostic["max"] == float(reference.max())
    assert diagnostic["l2_norm"] == pytest.approx(
        float(reference.double().norm()), rel=1e-12
    )

    bad = torch.tensor([1.0, float("nan"), float("inf")])
    diagnostic = runner._tensor_diagnostic(bad)
    assert diagnostic["finite"] is False
    assert diagnostic["nonfinite_count"] == 2
    assert diagnostic["min"] == diagnostic["max"] == 1.0
    assert diagnostic["l2_norm"] is None


class FakeProcessor:
    def __init__(self, *, visual_after_question=False, corrupt_prompt=False):
        self.resize_calls = []
        self.image_processor = SimpleNamespace(
            merge_size=2, resample="fake-bilinear", resize=self._resize
        )
        self.image_token_id = 99
        self.tokenizer = self
        self.seen_pixel_values = []
        self.visual_after_question = visual_after_question
        self.corrupt_prompt = corrupt_prompt

    def _resize(self, *, image, size, interpolation):
        self.resize_calls.append((tuple(image.shape), size.height, size.width, interpolation))
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(size.height, size.width), mode="bilinear", align_corners=False
        ).squeeze(0)

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert not add_special_tokens and return_tensors == "pt"
        token = 10 if text == "question" else 11
        return {"input_ids": torch.tensor([[token]])}

    def apply_chat_template(self, messages, **kwargs):
        generation_prompt = kwargs.pop("add_generation_prompt", False)
        assert kwargs == {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
        images = [row["image"] for row in messages[0]["content"] if row["type"] == "image"]
        question = next(row["text"] for row in messages[0]["content"] if row["type"] == "text")
        self.seen_pixel_values.append([image.getpixel((0, 0))[0] for image in images])
        question_token = 10 if question == "question" else 11
        ids = ([question_token] + [99] * len(images) if self.visual_after_question else [99] * len(images) + [question_token]) + [20]
        if generation_prompt and self.corrupt_prompt:
            ids[-1] = 21
        if not generation_prompt:
            ids += [30]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "pixel_values": torch.tensor(self.seen_pixel_values[-1], dtype=torch.float32)[:, None],
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class FakeOutput:
    def __init__(self, hidden, valid, logits, loss):
        self.visual_state_hidden = hidden
        self.visual_state_valid_mask = valid
        self.logits = logits
        self.loss = loss


class FakeGuideModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_geometry_encoder=True,
            text_config=SimpleNamespace(hidden_size=8, max_position_embeddings=128),
        )
        self.geometry_encoder = torch.nn.Module()
        self.geometry_encoder.vggt = torch.nn.Linear(1, 8)
        self.geometry_encoder.vggt.requires_grad_(False)
        self.geometry_merger = torch.nn.Linear(1, 8)
        self.feature_fusion_module = torch.nn.Linear(8, 8)
        self.language_model = torch.nn.Linear(8, 8)
        self.received_keys = None
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    def forward(self, **kwargs):
        assert kwargs.pop("return_visual_state_tap") is True
        self.received_keys = set(kwargs)
        geometry = kwargs["geometry_encoder_inputs"]
        assert isinstance(geometry, list) and len(geometry) == 1
        hidden = self.language_model(
            self.feature_fusion_module(self.geometry_merger(kwargs["pixel_values"]))
        ).unsqueeze(0)
        valid = torch.ones(1, hidden.shape[1], dtype=torch.bool)
        logits = hidden.mean(1)
        loss = logits.float().square().mean() if "labels" in kwargs else None
        return FakeOutput(hidden, valid, logits, loss)


class FakeDistributedWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.forward_calls = 0

    def forward(self, **kwargs):
        self.forward_calls += 1
        return self.module(**kwargs)


def sample(media_kind, count):
    source = "adt" if media_kind == "video" else "hypersim"
    qa = {
        "qa_id": f"{source}:q",
        "source_dataset": source,
        "scene_id": "s",
        "media_kind": media_kind,
        "vsi_media": f"{source}/x.{'mp4' if media_kind == 'video' else 'png'}",
        "actual_frame_indices": list(range(count)),
        "actual_frame_keys": [str(index) for index in range(count)],
        "conversations": [
            {"from": "human", "value": f"<{media_kind}> question"},
            {"from": "gpt", "value": "answer"},
        ],
    }
    frames = tuple({"frame_key": str(index), "visible_nodes": []} for index in range(count))
    return PartASample({"nodes": []}, frames, qa)


def test_fake_processor_model_end_to_end_video_and_image_exact_order():
    processor = FakeProcessor()
    collator = PartAT0Collator(processor)
    model = FakeGuideModel()
    attach_a1o_state_head(
        model,
        StateHeadConfig(
            hidden_size=8,
            num_categories=len(CANONICAL_CATEGORIES),
            num_layers=1,
            num_heads=1,
            ffn_dim=16,
        ),
    )
    for media_kind, count in (("video", 16), ("image", 1)):
        images = tuple(Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(count))
        fixture = collator(sample(media_kind, count), images)
        output = forward_visual_tap(model, fixture)
        assert processor.seen_pixel_values[-1] == list(range(count))
        assert fixture.frame_ids == tuple(range(count))
        assert fixture.frame_token_counts == (1,) * count
        assert fixture.visual_prefix_before_question
        assert fixture.model_kwargs["labels"][0, :-1].eq(-100).all()
        assert fixture.model_kwargs["labels"][0, -1].item() == 30
        assert processor.resize_calls[-count:] == [
            ((3, 2, 2), 28, 28, "fake-bilinear")
        ] * count
        assert output.visual_state_valid_mask.sum().item() == count
        assert model.received_keys == {
            "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "geometry_encoder_inputs", "labels"
        }
        assert "qa" not in model.received_keys and "frame_ids" not in model.received_keys
        alternate = collator.with_question(fixture, "different question")
        assert alternate.model_kwargs["pixel_values"] is fixture.model_kwargs["pixel_values"]
        assert alternate.model_kwargs["geometry_encoder_inputs"] is fixture.model_kwargs["geometry_encoder_inputs"]
        empty = StateTargets(
            categories=torch.empty(0, dtype=torch.long),
            centers_world_m=torch.empty(0, 3),
            extents_m=torch.empty(0, 3),
            visibility=torch.empty(0, count),
            category_valid=torch.empty(0, dtype=torch.bool),
            center_valid=torch.empty(0, dtype=torch.bool),
            extent_valid=torch.empty(0, dtype=torch.bool),
            visibility_valid=torch.empty(0, count, dtype=torch.bool),
            scene_scale_m=torch.tensor(1.0),
            source_dataset="adt" if media_kind == "video" else "hypersim",
            scene_id="s",
        )
        branch = run_a1o_side_branch(
            model,
            output.visual_state_hidden,
            output.visual_state_valid_mask,
            [fixture.frame_token_counts],
            [fixture.frame_ids],
            [fixture.media_kind],
            [empty],
            StateLossConfig(),
        )
        branch.losses["loss_state"].backward()
        assert model.geometry_merger.weight.grad is not None


def test_forward_visual_tap_inspects_nested_wrapper_but_forwards_through_outer_wrapper():
    processor = FakeProcessor()
    fixture = PartAT0Collator(processor)(
        sample("video", 2),
        tuple(Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(2)),
    )
    guide = FakeGuideModel()
    inner_wrapper = FakeDistributedWrapper(guide)
    outer_wrapper = FakeDistributedWrapper(inner_wrapper)

    output = forward_visual_tap(outer_wrapper, fixture)

    assert output.visual_state_valid_mask.sum().item() == 2
    assert outer_wrapper.forward_calls == 1
    assert inner_wrapper.forward_calls == 1


def test_forward_visual_tap_does_not_unwrap_ordinary_configured_model():
    processor = FakeProcessor()
    fixture = PartAT0Collator(processor)(
        sample("image", 1), (Image.new("RGB", (2, 2), (0, 0, 0)),)
    )
    guide = FakeGuideModel()
    guide.module = torch.nn.Module()

    output = forward_visual_tap(guide, fixture)

    assert output.visual_state_valid_mask.sum().item() == 1


def test_cli_failure_is_nonzero_and_atomic_report_exists(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/parta/run_t0_a.py"
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--adt-root", str(tmp_path / "missing-adt"),
            "--hypersim-root", str(tmp_path / "missing-hypersim"),
            "--media-root", str(tmp_path / "missing-media"),
            "--model-path", str(tmp_path / "missing-model"),
            "--vggt-path", str(tmp_path / "missing-vggt"),
            "--output", str(output),
            "--device", "cpu",
            "--dtype", "float32",
        ],
        cwd=script.parents[2],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    failure = output / "t0_a_failure.json"
    assert failure.is_file()
    assert '"status": "failed"' in failure.read_text()
    assert '"status": "failed"' in (output / "t0_a_report.json").read_text()
    assert '"status": "failed"' in (output / "provenance.json").read_text()


@pytest.mark.parametrize(
    "processor,match",
    [
        (FakeProcessor(visual_after_question=True), "not before the question"),
        (FakeProcessor(corrupt_prompt=True), "not an exact full-chat prefix"),
    ],
)
def test_collator_attacks_fail_closed(processor, match):
    collator = PartAT0Collator(processor)
    images = tuple(Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(16))
    with pytest.raises(ContractError, match=match):
        collator(sample("video", 16), images)


def test_successful_fake_cli_is_transactional_and_runs_real_gates(tmp_path, monkeypatch):
    """Exercise main(), including checkpoint restores, without claiming a real GPU T0 pass."""
    _, runner = load_runner_module()
    output = tmp_path / "completed"
    roots = {name: tmp_path / name for name in ("adt", "hypersim", "media", "model", "vggt")}
    for root in roots.values():
        root.mkdir()
    (roots["adt"] / "qa_manifest_exact_verified.jsonl").write_text(json.dumps({
        "source_dataset": "adt", "media_kind": "video",
        "actual_frame_indices": list(range(16)), "sampling_base_interval": 1.0,
        "frame_binding_sha256": "a" * 64,
    }) + "\n")
    (roots["hypersim"] / "qa_manifest_exact_verified.jsonl").write_text(json.dumps({
        "source_dataset": "hypersim", "media_kind": "image",
        "actual_frame_indices": [7], "evidence_frame_indices": [7],
        "qa_evidence_scope": "frame_verified", "qa_visual_support_verified": True,
        "sampling_base_interval": None, "frame_binding_sha256": "a" * 64,
    }) + "\n")
    (roots["model"] / "model.bin").write_bytes(b"fake-guide")
    (roots["model"] / "config.json").write_text('{"model":"guide"}\n')
    (roots["vggt"] / "vggt.bin").write_bytes(b"fake-vggt")
    (roots["vggt"] / "config.json").write_text('{"model":"vggt"}\n')
    monkeypatch.setattr(sys, "argv", [
        str(Path(runner.__file__)),
        "--adt-root", str(roots["adt"]),
        "--hypersim-root", str(roots["hypersim"]),
        "--media-root", str(roots["media"]),
        "--model-path", str(roots["model"]),
        "--vggt-path", str(roots["vggt"]),
        "--output", str(output),
        "--dtype", "float32", "--device", "cpu", "--seed", "42",
    ])
    fixtures = []
    for scene_id in runner.FIXTURE_SCENE_IDS:
        kind = "video" if scene_id.startswith("Apartment") else "image"
        item = sample(kind, 16 if kind == "video" else 1)
        item.qa["scene_id"] = scene_id
        item.qa["qa_id"] = f"q:{scene_id}"
        item.qa["frame_binding_sha256"] = "a" * 64
        fixtures.append(item)
    monkeypatch.setattr(runner, "PartACanonicalDataset", lambda roots: SimpleNamespace(samples=fixtures))
    monkeypatch.setattr(runner, "ExactMediaLoader", lambda root: SimpleNamespace(load=lambda item: tuple(
        Image.new("RGB", (2, 2), (index + 1, 0, 0)) for index in item.qa["actual_frame_indices"]
    )))
    runtime_contract = {
        "geometry_encoder_type_effective": "vggt",
        "geometry_encoder_type_source": "model_default_missing_legacy_field",
        "processor_min_pixels": runner.GUIDE_MIN_PIXELS,
        "processor_max_pixels": runner.GUIDE_MAX_PIXELS,
        "processor_pixel_field_sources": {
            "min_pixels": "processor_artifact_explicit",
            "max_pixels": "processor_artifact_explicit",
        },
    }
    monkeypatch.setattr(
        runner, "_load_local",
        lambda *args: (FakeProcessor(), FakeGuideModel(), runtime_contract),
    )
    monkeypatch.setattr(
        runner,
        "_startup_resource_preflight",
        lambda *args: {
            "estimated_smoke_checkpoint_bytes": 1,
            "required_transaction_disk_bytes": 2,
            "disk_free_bytes": 3,
            "cuda_device_index": 0,
            "cuda_capability": [9, 0],
            "cuda_memory_free_bytes": 4,
            "cuda_memory_total_bytes": 5,
            "flash_attn_version": "fake",
        },
    )
    captured = {}
    original_attach = runner.attach_a1o_state_head
    def attach_and_capture(model, config):
        head = original_attach(model, config)
        captured.setdefault("model", model)
        return head
    monkeypatch.setattr(runner, "attach_a1o_state_head", attach_and_capture)
    def empty_target(item):
        count = len(item.qa["actual_frame_indices"])
        target = StateTargets(
            categories=torch.empty(0, dtype=torch.long), centers_world_m=torch.empty(0, 3),
            extents_m=torch.empty(0, 3), visibility=torch.empty(0, count),
            category_valid=torch.empty(0, dtype=torch.bool), center_valid=torch.empty(0, dtype=torch.bool),
            extent_valid=torch.empty(0, dtype=torch.bool), visibility_valid=torch.empty(0, count, dtype=torch.bool),
            scene_scale_m=torch.tensor(1.0), source_dataset=item.qa["source_dataset"], scene_id=item.qa["scene_id"],
        )
        return target, SimpleNamespace(actual_input_visible_object_count=0, truncated_object_ids=())
    monkeypatch.setattr(runner, "build_state_targets", empty_target)
    runner.main()
    report = json.loads((output / "t0_a_report.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    resolved = json.loads((output / "resolved_config.json").read_text())
    assert report["status"] == "complete_passed"
    assert report["checks"]["finite"]["parameter_gradients"]
    assert report["checks"]["question_invariance"]["qa_logits_intentionally_unconstrained"]
    assert report["checks"]["head_free_equivalence"]["backbone_digest_matches"]
    assert json.loads((output / "run_status.json").read_text())["status"] == "complete"
    assert resolved["seed"] == 42 and resolved["base_interval"] == 1.0
    assert resolved["use_cache"] is False
    assert resolved["gradient_checkpointing"] is True
    assert resolved["gradient_checkpointing_use_reentrant"] is False
    assert captured["model"].gradient_checkpointing_enabled is True
    assert captured["model"].input_require_grads_enabled is True
    assert provenance["manifest_provenance"]["adt"]["base_interval"] == 1.0
    assert provenance["manifest_provenance"]["hypersim"]["base_interval"] is None
    assert provenance["manifest_provenance"]["hypersim"]["sampling_contract"] == "single_frame_verified_v1"
    assert provenance["geometry_encoder_type_effective"] == "vggt"
    assert provenance["geometry_encoder_type_source"] == "model_default_missing_legacy_field"
    assert provenance["a1_checkpoint_artifact"]["ordered_shards"][0]["sha256"]
    for key in ("guide_checkpoint_artifact", "vggt_checkpoint_artifact"):
        assert provenance[key]["mode"] == "no_index_explicit_manifest"
        assert provenance[key]["config_files"][0]["name"] == "config.json"
        assert provenance[key]["ordered_shards"][0]["sha256"]
        assert provenance[key]["artifact_sha256"]
    assert provenance["a1_checkpoint_role"] == "initialization_no_optimizer_updates"
    assert provenance["a1_checkpoint_optimizer_steps"] == 0
    assert provenance["parameter_sha256_before_backward"] == provenance["parameter_sha256_after_backward"]
    assert (output / "t0_a1o_smoke_no_update.pt").is_file()
    assert provenance["head_free_restored_backbone_sha256"] == provenance["a1_loaded_shared_state_sha256"]
    assert not list(tmp_path.glob(".completed.running-*"))


def test_runner_source_has_no_stale_external_a1_input_or_undefined_flow():
    script = Path(__file__).resolve().parents[1] / "scripts/parta/run_t0_a.py"
    source = script.read_text()
    compile(source, str(script), "exec")
    assert "--a1-checkpoint" not in source
    assert "args.a1_checkpoint" not in source
    assert source.index('smoke_payload = torch.load(resume_path') < source.index(
        "load_head_free_checkpoint("
    )
    assert source.index("a1_artifact = _checkpoint_artifact_provenance(resume_path)") < source.index(
        '"a1_checkpoint_artifact": a1_artifact'
    )


def test_data_contract_imports_with_real_runner_src_only_path(tmp_path):
    """Prevent pytest's repository-root sys.path from hiding runner import failures."""
    project = Path(__file__).resolve().parents[1]
    code = """
import sys
from pathlib import Path

project = Path(sys.argv[1]).resolve()
assert project not in map(Path, map(str, sys.path))
import parta_data_contract
import adt_gt_supported_clip
assert adt_gt_supported_clip.ContractError is parta_data_contract.ContractError
assert callable(parta_data_contract.validate_guide_sampling_binding)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    result = subprocess.run(
        [sys.executable, "-c", code, str(project)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_prefix,python_paths,cwd_mode",
    [
        ("", ("src",), "tmp"),
        ("src.", ("root",), "tmp"),
        ("", ("root", "src"), "root"),
    ],
)
def test_data_contract_import_namespace_identity(
    tmp_path, module_prefix, python_paths, cwd_mode
):
    project = Path(__file__).resolve().parents[1]
    code = """
import importlib
import sys

prefix = sys.argv[1]
contract = importlib.import_module(prefix + "parta_data_contract")
clip = importlib.import_module(prefix + "adt_gt_supported_clip")
assert clip.ContractError is contract.ContractError
opposite = "parta_data_contract" if prefix else "src.parta_data_contract"
assert opposite not in sys.modules
"""
    resolved_paths = [
        str(project / "src") if item == "src" else str(project)
        for item in python_paths
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(resolved_paths)
    result = subprocess.run(
        [sys.executable, "-c", code, module_prefix],
        cwd=project if cwd_mode == "root" else tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_seed_other_than_42_is_rejected_before_execution(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/parta/run_t0_a.py"
    result = subprocess.run([
        sys.executable, str(script), "--adt-root", "a", "--hypersim-root", "h",
        "--media-root", "m", "--model-path", "model", "--vggt-path", "vggt",
        "--output", str(tmp_path / "out"), "--seed", "41",
    ], cwd=script.parents[2], text=True, capture_output=True)
    assert result.returncode == 2
    assert "freezes T0-A seed exactly to 42" in result.stderr


def test_main_rejects_non_frozen_seed_even_if_parse_is_monkeypatched(tmp_path, monkeypatch):
    _, runner = load_runner_module()
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(
        adt_root=tmp_path / "adt", hypersim_root=tmp_path / "hypersim",
        media_root=tmp_path / "media", model_path=tmp_path / "model",
        vggt_path=tmp_path / "vggt", output=tmp_path / "out",
        dtype="float32", device="cpu", seed=7,
    ))
    with pytest.raises(ValueError, match="freezes T0-A seed exactly to 42"):
        runner.main()


def test_checkpoint_index_and_shard_attacks_fail_or_change_digest(tmp_path):
    _, runner = load_runner_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "a.safetensors").write_bytes(b"a")
    index = checkpoint / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a.key": "a.safetensors", "b.key": "b.safetensors"}}))
    with pytest.raises(RuntimeError, match="index/shard set mismatch"):
        runner._checkpoint_artifact_provenance(checkpoint)
    (checkpoint / "b.safetensors").write_bytes(b"b")
    index.write_text(json.dumps({"weight_map": {"b.key": "b.safetensors", "a.key": "a.safetensors"}}))
    unsorted = runner._checkpoint_artifact_provenance(checkpoint)
    index.write_text(json.dumps({"weight_map": {"a.key": "a.safetensors", "b.key": "b.safetensors"}}))
    original = runner._checkpoint_artifact_provenance(checkpoint)
    assert unsorted["index"]["ordered_weight_map_sha256"] == original["index"]["ordered_weight_map_sha256"]
    (checkpoint / "b.safetensors").write_bytes(b"modified")
    modified_shard = runner._checkpoint_artifact_provenance(checkpoint)
    assert original["artifact_sha256"] != modified_shard["artifact_sha256"]
    index.write_text(json.dumps({
        "metadata": {"attack": True},
        "weight_map": {"a.key": "a.safetensors", "b.key": "b.safetensors"},
    }))
    modified_index = runner._checkpoint_artifact_provenance(checkpoint)
    assert modified_shard["artifact_sha256"] != modified_index["artifact_sha256"]
    index.write_text(json.dumps({"weight_map": {"a.key": "../a.safetensors"}}))
    with pytest.raises(RuntimeError, match="unsafe shard names"):
        runner._checkpoint_artifact_provenance(checkpoint)


def test_no_index_checkpoint_manifest_binds_config_and_rejects_unindexed_shards(tmp_path):
    _, runner = load_runner_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"value":1}\n')
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    original = runner._checkpoint_artifact_provenance(checkpoint)
    assert original["mode"] == "no_index_explicit_manifest"
    assert [item["name"] for item in original["config_files"]] == ["config.json"]
    (checkpoint / "config.json").write_text('{"value":2}\n')
    modified = runner._checkpoint_artifact_provenance(checkpoint)
    assert original["artifact_sha256"] != modified["artifact_sha256"]
    (checkpoint / "second.safetensors").write_bytes(b"second")
    with pytest.raises(RuntimeError, match="sharded checkpoint requires an index JSON"):
        runner._checkpoint_artifact_provenance(checkpoint)


def test_manifest_sampling_is_source_aware_and_fail_closed(tmp_path):
    _, runner = load_runner_module()
    path = tmp_path / "manifest.jsonl"

    hypersim = {
        "source_dataset": "hypersim", "media_kind": "image",
        "actual_frame_indices": [38], "evidence_frame_indices": [38],
        "qa_evidence_scope": "frame_verified", "qa_visual_support_verified": True,
        "sampling_base_interval": None, "frame_binding_sha256": "b" * 64,
    }
    path.write_text(json.dumps(hypersim) + "\n")
    result = runner._validate_manifest_sampling(path, "hypersim", 1.0)
    assert result["base_interval"] is None
    assert result["sampling_contract"] == "single_frame_verified_v1"

    for field, value, match in (
        ("actual_frame_indices", [38, 39], "single-frame"),
        ("qa_evidence_scope", "scene_associated_unlocalized", "frame-verified evidence"),
        ("qa_visual_support_verified", False, "frame-verified evidence"),
        ("evidence_frame_indices", [39], "frame-verified evidence"),
        ("sampling_base_interval", 1.0, "base_interval is inapplicable"),
        ("sampling_parameters", {"base_interval": 1.0}, "base_interval is inapplicable"),
    ):
        attacked = dict(hypersim)
        attacked[field] = value
        path.write_text(json.dumps(attacked) + "\n")
        with pytest.raises(RuntimeError, match=match):
            runner._validate_manifest_sampling(path, "hypersim", 1.0)

    adt = {
        "source_dataset": "adt", "media_kind": "video",
        "actual_frame_indices": list(range(16)), "sampling_base_interval": 1.0,
        "frame_binding_sha256": "c" * 64,
    }
    path.write_text(json.dumps(adt) + "\n")
    assert runner._validate_manifest_sampling(path, "adt", 1.0)["base_interval"] == 1.0
    adt["sampling_base_interval"] = None
    path.write_text(json.dumps(adt) + "\n")
    with pytest.raises(RuntimeError, match="base_interval mismatch"):
        runner._validate_manifest_sampling(path, "adt", 1.0)


def test_geometry_encoder_legacy_default_and_explicit_attack():
    _, runner = load_runner_module()
    missing = runner._geometry_encoder_contract(SimpleNamespace())
    assert missing == {
        "geometry_encoder_type_effective": "vggt",
        "geometry_encoder_type_source": "model_default_missing_legacy_field",
    }
    explicit = runner._geometry_encoder_contract(
        SimpleNamespace(geometry_encoder_type="vggt")
    )
    assert explicit["geometry_encoder_type_source"] == "config_explicit"
    with pytest.raises(RuntimeError, match="requires VGGT"):
        runner._geometry_encoder_contract(
            SimpleNamespace(geometry_encoder_type="pi3")
        )


def test_context_preflight_records_budget_and_rejects_overflow():
    _, runner = load_runner_module()
    fixture = SimpleNamespace(
        model_kwargs={"input_ids": torch.zeros(1, 12, dtype=torch.long)},
        frame_token_counts=(3, 4),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=16)
        )
    )
    result = runner._context_preflight(fixture, model)
    assert result == {
        "frame_visual_tokens": [3, 4],
        "total_visual_tokens": 7,
        "sequence_length": 12,
        "context_limit": 16,
    }
    model.config.text_config.max_position_embeddings = 11
    with pytest.raises(RuntimeError, match="exceeds model context"):
        runner._context_preflight(fixture, model)


def test_e01_gradient_gate_freezes_vggt_and_uses_trainable_guide_path():
    _, runner = load_runner_module()
    model = FakeGuideModel()
    loss = sum(parameter.sum() for parameter in (
        model.geometry_merger.weight,
        model.feature_fusion_module.weight,
        model.language_model.weight,
    ))
    loss.backward()
    audit = runner._e01_trainability_audit(model)
    assert audit["passed"]
    assert audit["frozen_vggt_parameter_count"] > 0
    assert not audit["frozen_vggt_violations"]
    model.geometry_encoder.vggt.weight.requires_grad_(True)
    assert not runner._e01_trainability_audit(model)["passed"]


def test_resource_preflight_records_capacity_and_fails_closed(tmp_path, monkeypatch):
    _, runner = load_runner_module()
    artifact = {"ordered_shards": [{"size_bytes": 100}]}
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**9))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(runner.torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setattr(runner.torch.cuda, "mem_get_info", lambda index: (1000, 2000))
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(__version__="fake"))
    passed = runner._startup_resource_preflight(
        tmp_path, artifact, artifact, "cuda:0"
    )
    assert passed["passed"] and passed["estimated_smoke_checkpoint_bytes"] == 200
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    failed = runner._startup_resource_preflight(
        tmp_path, artifact, artifact, "cuda:0"
    )
    assert not failed["passed"]
    assert any("insufficient disk" in item for item in failed["failures"])
