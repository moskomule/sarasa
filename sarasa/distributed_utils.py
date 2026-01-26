import os
from datetime import timedelta
from functools import cache
from typing import Literal

import torch
from loguru import logger
from torch import distributed as dist
from torch import nn


@cache
def world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@cache
def rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@cache
def local_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_local_rank()


def init_distributed(
    init_timeout_seconds: int,
) -> None:
    if "RANK" in os.environ:
        # run with torchrun
        dist.init_process_group(
            backend="nccl",
            device_id=int(os.environ["LOCAL_RANK"]),
            timeout=timedelta(seconds=init_timeout_seconds),
        )
        logger.info("Initialized distributed process group")


def update_timeout(
    timeout_seconds: int,
    device: torch.device,
) -> None:
    logger.info(f"Updating distributed timeout to {timeout_seconds} seconds")
    torch.distributed.barrier(device_ids=[device.index])
    torch.accelerator.synchronize(device)

    # at the moment, default process group is the only one supported (None)
    torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=timeout_seconds), None)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def apply_distributed(
    model: nn.Module,
    mode: Literal["ddp", "fsdp"],
    compile: bool,
    reshard_after_forward: bool,
) -> None:
    mesh = dist.device_mesh.init_device_mesh("cuda", (world_size(),))

    if mode == "ddp":
        from torch.distributed._composable.replicate import replicate

        if compile:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

        replicate(model, device_mesh=mesh, bucket_cap_mb=100)
        logger.info("Applied DDP to the model")

    elif mode == "fsdp":
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)

        logger.info(
            f"Applied FSDP to the model (param_dtype={mp_policy.param_dtype}, "
            f"reduce_dtype={mp_policy.reduce_dtype}, reshard_after_forward={reshard_after_forward})"
        )
