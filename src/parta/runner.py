"""Shared, fail-closed training core for matched Part A A0 and A1-O.

The core is model-adapter based so CPU tests and the GUIDE runtime use exactly
the same optimizer/checkpoint/logging transaction.  A0 never constructs the
state head; A1-O's head is a training-only side branch.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import numpy as np
from torch import nn

from .checkpoint import ResumeContract, load_training_checkpoint, save_training_checkpoint
from .provenance import stable_sha256
from .state_loss import StateLossConfig, StateTargets
from .training import A1OSideBranchOutput, run_a1o_side_branch
from .training_log import JsonlTrainingLogger, TrainingStepRecord


@dataclass(frozen=True)
class PartATrainConfig:
    arm: str
    seed: int = 42
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lambda_state: float = 0.05
    num_slots: int = 384
    video_min_frames: int = 16
    video_max_frames: int = 32
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    max_steps: int = 1000
    dtype: str = "bfloat16"
    defaults_status: str = "pending_gate_config_after_d62_execution_evidence"

    def validate(self) -> None:
        if self.arm not in {"a0", "a1o"}:
            raise ValueError("arm must be a0 or a1o")
        if self.num_slots != 384 or (self.video_min_frames, self.video_max_frames) != (16, 32):
            raise ValueError("Part A freezes K=384 and dynamic 16-32 frame bounds")
        if self.arm == "a1o" and not 0.0 < self.lambda_state <= 1.0:
            raise ValueError("A1-O lambda_state must be positive")
        if min(self.gradient_accumulation_steps, self.save_steps, self.max_steps) < 1:
            raise ValueError("step counts must be positive")
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "gradient accumulation >1 is not yet a reviewed runner contract; "
                "freeze it after GPU profiling"
            )
        if self.max_grad_norm <= 0 or self.learning_rate <= 0:
            raise ValueError("optimizer values must be positive")


def matched_fairness_payload(
    config: PartATrainConfig,
    *,
    manifest_sha256: str,
    initialization_sha256: str,
    exact_frame_binding_sha256: str,
    trainable_shared_parameter_names: Sequence[str],
    execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Arm-independent identity; only head/loss fields are intentionally absent."""
    config.validate()
    payload = {
        "schema_version": "parta_matched_fairness_v1",
        "manifest_sha256": manifest_sha256,
        "initialization_sha256": initialization_sha256,
        "exact_frame_binding_sha256": exact_frame_binding_sha256,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "video_min_frames": config.video_min_frames,
        "video_max_frames": config.video_max_frames,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "save_steps": config.save_steps,
        "max_steps": config.max_steps,
        "dtype": config.dtype,
        "sampler": "source_balanced_round_robin_v1",
        "shared_trainable_parameters": sorted(trainable_shared_parameter_names),
        "allowed_arm_differences": ["arm", "state_head", "state_loss", "lambda_state"],
    }
    if execution_contract is not None:
        forbidden = {"arm", "lambda_state", "state_head", "state_loss"} & set(execution_contract)
        if forbidden:
            raise ValueError(f"execution contract contains arm-specific fields: {sorted(forbidden)}")
        payload["execution_contract"] = dict(execution_contract)
    return payload


def assert_matched_fairness(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if dict(left) != dict(right):
        differing = sorted(set(left) | set(right))
        differing = [key for key in differing if left.get(key) != right.get(key)]
        raise ValueError(f"A0/A1-O matched fairness mismatch: {differing}")
    return stable_sha256(left)


def validate_single_step_execution_contract(
    *, cli_gradient_accumulation_steps: int, config: PartATrainConfig,
    execution_contract: Mapping[str, Any], world_size: int,
) -> None:
    """Keep the currently reviewed runner fail-closed at one batch per step."""
    values = (
        cli_gradient_accumulation_steps,
        config.gradient_accumulation_steps,
        execution_contract.get("gradient_accumulation_steps"),
    )
    if values != (1, 1, 1):
        raise ValueError(
            "formal Part A runner only supports gradient_accumulation_steps == 1; "
            f"cli/config/execution={values}"
        )
    if execution_contract.get("effective_global_batch_size") != world_size:
        raise ValueError("effective global batch must equal world_size when accumulation is disabled")


def seed_matched_run(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_a1o_head_without_advancing_shared_rng(
    model: nn.Module, config: Any, *, seed: int
) -> nn.Module:
    """Initialize the side head deterministically and restore the shared RNG stream."""
    from .checkpoint import capture_rng_state, restore_rng_state
    from .training import attach_a1o_state_head

    shared_rng = capture_rng_state()
    try:
        seed_matched_run(seed + 10_000_019)
        return attach_a1o_state_head(model, config)
    finally:
        restore_rng_state(shared_rng)


@dataclass
class SourceBalancedCursor:
    """Deterministic epoch-aware cursor with exact resume semantics."""

    rows: Sequence[Mapping[str, Any]]
    seed: int
    epoch: int = 0
    position: int = 0

    def order(self) -> list[int]:
        from .unified_data import iter_source_balanced_indices

        return list(iter_source_balanced_indices(self.rows, seed=self.seed, epoch=self.epoch))

    def next_index(self) -> int:
        order = self.order()
        if not order:
            raise RuntimeError("empty source-balanced sampler")
        if self.position >= len(order):
            self.epoch += self.position // len(order)
            self.position %= len(order)
            order = self.order()
        index = order[self.position]
        self.position += 1
        if self.position == len(order):
            self.epoch += 1
            self.position = 0
        return index

    def next_distributed_index(self, *, rank: int, world_size: int) -> int:
        """Shard one global source-balanced step while keeping cursors identical."""
        if not 0 <= rank < world_size:
            raise ValueError("invalid distributed rank/world size")
        selected = None
        for offset in range(world_size):
            value = self.next_index()
            if offset == rank:
                selected = value
        assert selected is not None
        return selected


@dataclass
class PartATrainBatch:
    model_inputs: Mapping[str, Any]
    targets: Sequence[StateTargets]
    source_datasets: Sequence[str]
    frame_ids: Sequence[Sequence[int]]
    frame_token_counts: Sequence[Sequence[int]]
    media_kinds: Sequence[str]
    expected_frame_binding_sha256: Sequence[str]

    def validate(self) -> None:
        size = len(self.targets)
        fields = (
            self.source_datasets,
            self.frame_ids,
            self.frame_token_counts,
            self.media_kinds,
            self.expected_frame_binding_sha256,
        )
        if size < 1 or any(len(field) != size for field in fields):
            raise ValueError("training batch metadata cardinality mismatch")
        for kind, ids, counts in zip(self.media_kinds, self.frame_ids, self.frame_token_counts):
            if len(ids) != len(counts):
                raise ValueError("frame IDs/counts mismatch")
            if kind == "video" and not 16 <= len(ids) <= 32:
                raise ValueError("video batch violates dynamic 16-32 frame contract")
            if kind == "image" and len(ids) != 1:
                raise ValueError("image batch must contain one exact frame")
        if any(len(value) != 64 for value in self.expected_frame_binding_sha256):
            raise ValueError("batch lacks exact frame-binding SHA256")


@dataclass
class SharedForwardOutput:
    qa_loss: torch.Tensor
    visual_state_hidden: torch.Tensor | None = None
    visual_state_valid_mask: torch.Tensor | None = None
    a1o_side_branch: A1OSideBranchOutput | None = None


ForwardAdapter = Callable[[nn.Module, Mapping[str, Any], bool], SharedForwardOutput]


class PartATrainer:
    """Optimizer-step implementation shared by both matched arms."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: PartATrainConfig,
        forward_adapter: ForwardAdapter,
        logger: JsonlTrainingLogger,
        manifest_sha256: str,
        resolved_config_sha256: str,
        matched_contract_sha256: str,
        loss_config: Any = None,
        is_primary: bool = True,
        transaction_kind: str = "formal",
        promotable: bool = True,
    ) -> None:
        config.validate()
        candidate = model
        while not hasattr(candidate, "parta_state_head"):
            wrapped = getattr(candidate, "module", None)
            if not isinstance(wrapped, nn.Module):
                wrapped = getattr(candidate, "_fsdp_wrapped_module", None)
            if not isinstance(wrapped, nn.Module) or wrapped is candidate:
                break
            candidate = wrapped
        has_head = hasattr(candidate, "parta_state_head")
        if has_head != (config.arm == "a1o"):
            raise ValueError("A0 must be head-free and A1-O must have exactly one state head")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.forward_adapter = forward_adapter
        self.logger = logger
        self.loss_config = loss_config or StateLossConfig()
        self.is_primary = is_primary
        self.resume_contract = ResumeContract(
            arm=config.arm,
            manifest_sha256=manifest_sha256,
            resolved_config_sha256=resolved_config_sha256,
            matched_contract_sha256=matched_contract_sha256,
            transaction_kind=transaction_kind,
            promotable=promotable,
        )
        self.resume_contract.validate()
        self.global_step = 0
        self.epoch = 0
        self.sampler_position = 0

    def resume(self, path: str) -> None:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel
            is_fsdp = isinstance(self.model, FullyShardedDataParallel)
        except ImportError:
            is_fsdp = False
        if is_fsdp:
            from .checkpoint import load_fsdp_training_checkpoint
            counters = load_fsdp_training_checkpoint(
                path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                expected_contract=self.resume_contract,
            )
        else:
            checkpoint_model = getattr(self.model, "module", self.model)
            counters = load_training_checkpoint(
                path, model=checkpoint_model, optimizer=self.optimizer,
                scheduler=self.scheduler, expected_contract=self.resume_contract,
            )
        self.global_step = counters["global_step"]
        self.epoch = counters["epoch"]
        self.sampler_position = counters["sampler_position"]

    def save(self, path: str) -> None:
        model_state = optimizer_state = None
        try:
            from torch.distributed.fsdp import (
                FullOptimStateDictConfig, FullStateDictConfig,
                FullyShardedDataParallel, StateDictType,
            )
            is_fsdp = isinstance(self.model, FullyShardedDataParallel)
        except ImportError:
            is_fsdp = False
        if is_fsdp:
            with FullyShardedDataParallel.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state = self.model.state_dict()
                optimizer_state = FullyShardedDataParallel.optim_state_dict(
                    self.model, self.optimizer
                )
        elif hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        if not self.is_primary:
            return
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            sampler_position=self.sampler_position,
            contract=self.resume_contract,
            model_state=model_state,
            optimizer_state=optimizer_state,
        )

    def train_step(self, batch: PartATrainBatch) -> TrainingStepRecord:
        batch.validate()
        started = time.perf_counter()
        cuda_parameter = next(
            (parameter for parameter in self.model.parameters() if parameter.device.type == "cuda"),
            None,
        )
        cuda_device = cuda_parameter.device if cuda_parameter is not None else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            output = self.forward_adapter(self.model, batch.model_inputs, self.config.arm == "a1o")
            qa_loss = output.qa_loss
            if not isinstance(qa_loss, torch.Tensor) or qa_loss.ndim != 0:
                raise TypeError("forward adapter must return scalar qa_loss")
            losses: dict[str, torch.Tensor] = {
                name: qa_loss * 0.0
                for name in ("existence", "category", "center", "extent", "visibility")
            }
            state_loss = qa_loss * 0.0
            matched_slots = 0
            if self.config.arm == "a1o":
                branch = output.a1o_side_branch
                if branch is None:
                    # Retained only for unwrapped CPU seams/T0-A compatibility.
                    if output.visual_state_hidden is None or output.visual_state_valid_mask is None:
                        raise ValueError("A1-O forward lacks authoritative visual state tap")
                    branch = run_a1o_side_branch(
                        self.model, output.visual_state_hidden, output.visual_state_valid_mask,
                        batch.frame_token_counts, batch.frame_ids, batch.media_kinds,
                        batch.targets, self.loss_config,
                    )
                state_loss = branch.losses["loss_state"]
                for name in losses:
                    losses[name] = branch.losses[f"loss_{name}"]
                matched_slots = sum(len(rows) for rows, _ in branch.losses["assignments"])
            total_loss = qa_loss + (self.config.lambda_state * state_loss if self.config.arm == "a1o" else 0.0)
            if not torch.isfinite(total_loss):
                raise FloatingPointError("non-finite total loss")
            total_loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        named_parameters = [(name, parameter) for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        parameters = [parameter for _, parameter in named_parameters]
        shared_parameters = [parameter for name, parameter in named_parameters if "parta_state_head." not in name]
        head_parameters = [parameter for name, parameter in named_parameters if "parta_state_head." in name]
        shared_grad_norm = _gradient_norm(shared_parameters)
        head_grad_norm = _gradient_norm(head_parameters)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        grad_norm = float(grad_norm_tensor.detach().float())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        self.sampler_position += len(batch.targets)
        elapsed = max(time.perf_counter() - started, 1e-12)
        gt_count = sum(target.num_objects for target in batch.targets)
        frame_count = sum(len(ids) for ids in batch.frame_ids)
        record = TrainingStepRecord(
            step=self.global_step,
            arm=self.config.arm,
            source_dataset="+".join(sorted(set(batch.source_datasets))),
            qa_loss=float(qa_loss.detach().float()),
            state_loss=float(state_loss.detach().float()),
            total_loss=float(total_loss.detach().float()),
            loss_existence=float(losses["existence"].detach().float()),
            loss_category=float(losses["category"].detach().float()),
            loss_center=float(losses["center"].detach().float()),
            loss_extent=float(losses["extent"].detach().float()),
            loss_visibility=float(losses["visibility"].detach().float()),
            grad_norm=grad_norm,
            effective_gt_objects=gt_count,
            matched_slots=matched_slots,
            unmatched_slots=len(batch.targets) * self.config.num_slots - matched_slots,
            actual_frame_count=frame_count,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            step_seconds=elapsed,
            samples_per_second=len(batch.targets) / elapsed,
            frames_per_second=frame_count / elapsed,
            peak_cuda_memory_bytes=(
                int(torch.cuda.max_memory_allocated(cuda_device))
                if cuda_device is not None else None
            ),
            lambda_state=self.config.lambda_state if self.config.arm == "a1o" else 0.0,
            shared_grad_norm=shared_grad_norm,
            state_head_grad_norm=head_grad_norm,
            matching_mean_cost=(
                float(branch.losses["matching_mean_cost"].detach().float())
                if self.config.arm == "a1o" else 0.0
            ),
            matching_valid_pairs=matched_slots,
        )
        self.logger.write(
            record,
            extra={
                "frame_binding_sha256": list(batch.expected_frame_binding_sha256),
                "defaults_status": self.config.defaults_status,
            },
        )
        return record


def config_sha256(config: PartATrainConfig) -> str:
    config.validate()
    return stable_sha256(asdict(config))


def _gradient_norm(parameters: Sequence[torch.Tensor]) -> float:
    grads = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).norm(2))
