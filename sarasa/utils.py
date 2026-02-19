from __future__ import annotations

import contextlib
import gc
import os
import sys
import time
import typing
from datetime import timedelta
from functools import cache
from pathlib import Path

import torch
from loguru import logger
from torch import distributed as dist

if typing.TYPE_CHECKING:
    from sarasa.config import Config, Distributed, Profile
    from sarasa.models import BaseModel


IGNORE_INDEX = -100


def setup_profiler(
    config: Profile,
    device: torch.device,
    save_dir: Path,
) -> torch.profiler.profile | contextlib.nullcontext:
    if not config.enabled:

        class DummyProfiler:
            def step(self): ...

        return contextlib.nullcontext(DummyProfiler())

    activities = [torch.profiler.ProfilerActivity.CPU]
    if (pa := getattr(torch.profiler.ProfilerActivity, device.type.upper(), None)) is not None:
        activities.append(pa)

    logger.info(f"Profiler is activated with activities: {[a.name for a in activities]}")

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(save_dir / "profiler")),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def setup_logger(config: Config) -> None:
    logger.remove()
    if config.debug:
        logger_format = f"<blue>RANK={rank()}</blue> | " + (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=logger_format,
            backtrace=True,
            diagnose=True,
            level="DEBUG",
        )
    else:
        # log to stderr only for rank 0
        logger.add(sys.stderr, backtrace=True, diagnose=True, level="INFO" if rank() == 0 else "ERROR")

        if world_size() > 1:
            logger.info(
                "Logger is set up for distributed training. Only rank 0 will log messages. Use --debug for more verbose logging."
            )


@contextlib.contextmanager
def set_dtype(
    dtype: torch.dtype,
) -> typing.Generator[None, None, None]:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class GarbageCollector:
    def __init__(
        self,
        gc_freq: int,
    ) -> None:
        self.gc_freq = gc_freq
        if self.gc_freq > 0:
            # manually manage gc
            gc.disable()

    def collect(
        self,
        step: int,
    ) -> None:
        if self.gc_freq <= 0:
            # auto gc, nothing to do
            return

        if step % self.gc_freq == 0:
            begin = time.perf_counter()
            gc.collect(generation=1)
            end = time.perf_counter()
            logger.info(f"Garbage collection at step {step} took {end - begin:.4f} seconds")


# distributed utils


@cache
def world_size() -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@cache
def rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])

    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def init_distributed(
    backend: str,
    init_timeout_seconds: int,
) -> None:
    if "RANK" in os.environ:
        # run with torchrun
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=init_timeout_seconds),
        )
        logger.info("Initialized distributed process group")
    else:
        logger.info("Skipping distributed initialization")


def update_timeout(
    timeout_seconds: int,
    device: torch.device,
) -> None:
    logger.info(f"Updating distributed timeout to {timeout_seconds} seconds")
    torch.distributed.barrier(device_ids=[torch.accelerator.current_device_index()])
    torch.accelerator.synchronize(device)

    # at the moment, default process group is the only one supported (None)
    torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=timeout_seconds), None)


def apply_distributed(
    config: Distributed,
    model: BaseModel,
    device: torch.device,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    compile: bool,
) -> None:
    mesh = dist.device_mesh.init_device_mesh(device.type, (world_size(),))

    if config.dp_replicate_degree != 1:
        from torch.distributed._composable.replicate import replicate

        if compile:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

        # pyrefly: ignore [invalid-param-spec]
        replicate(model, device_mesh=mesh, bucket_cap_mb=100)
        logger.info("Applied DDP to the model")

    elif config.dp_shard_degree == -1:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)
        fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=config.reshard_after_forward)

        # sarasa scales the loss by valid tokens
        # type: ignore
        model.set_gradient_divide_factor(1.0)

        logger.info(
            f"Applied FSDP to the model (param_dtype={mp_policy.param_dtype}, "
            f"reduce_dtype={mp_policy.reduce_dtype}, reshard_after_forward={config.reshard_after_forward})"
        )
