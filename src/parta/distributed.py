"""Small torchrun/FSDP lifecycle contract for formal Part A runners."""

from __future__ import annotations

import atexit
import os
from dataclasses import dataclass

import torch


def destroy_distributed() -> None:
    """Release an initialized process group on normal and exceptional exits."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun Part A training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        atexit.register(destroy_distributed)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def synchronize_failure(local_failed: bool, context: DistributedContext) -> bool:
    if context.world_size == 1:
        return local_failed
    flag = torch.tensor([int(local_failed)], device=f"cuda:{context.local_rank}")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        torch.distributed.barrier(device_ids=[context.local_rank])
