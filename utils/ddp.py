"""Distributed (DDP) training helpers, driven by torchrun env vars."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_ddp() -> tuple[bool, int, int, int]:
    """Initialise the NCCL process group if launched via torchrun.

    Returns:
        (is_distributed, rank, local_rank, world_size)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    return True, rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def ddp_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module
