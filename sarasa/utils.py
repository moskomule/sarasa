import contextlib
import gc
import os
import time
from datetime import timedelta
from functools import cache
from typing import Literal

import torch
from loguru import logger
from torch import distributed as dist
from torch import nn


@contextlib.contextmanager
def set_dtype(
    dtype: torch.dtype,
) -> None:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class GarbageCollector:
    def __init__(
        self,
        gc_freq: int | None,
    ) -> None:
        self.gc_freq = gc_freq
        if self.gc_freq > 0:
            gc.disable()

    def collect(
        self,
        step: int,
    ) -> None:
        if self.gc_freq <= 0:
            return

        if step % self.gc_freq == 0:
            begin = time.perf_counter()
            gc.collect(generation=1)
            end = time.perf_counter()
            logger.debug(f"Garbage collection at step {step} took {end - begin:.4f} seconds")


# distributed utils


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
