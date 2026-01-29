import enum
import gc
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
from torch.distributed.checkpoint.stateful import Stateful

from sarasa.config import Config


class AsyncMode(enum.StrEnum):
    none = enum.auto()
    default = enum.auto()
    mem_pinned = enum.auto()


class ModelWrapper(Stateful):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"model": get_model_state_dict(self.model)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("...")


class Checkpointer:
    def __init__(
        self,
        config: Config,
        model: torch.nn.Module,
    ):
        self.config = config
        self.checkpoint_freq = config.checkpoint.save_freq
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.async_mode = AsyncMode(config.checkpoint.async_mode)
        if self.async_mode != AsyncMode.none:
            self.pg = dist.new_group(backend="gloo") if dist.is_initialized() else None

        self.stager = None
        self.save_future = None
        self.stage_future = None

        self.state = ModelWrapper(model)

    @torch.no_grad()
    def save(
        self,
        step: int,
    ) -> None:
        if step % self.checkpoint_freq != 0:
            return

        begin = time.perf_counter()
        checkpoint_id = str(self.checkpoint_dir / f"checkpoint_{step:09d}")

        # todo: save other states
        state_dict = self.state.state_dict()

        if self.async_mode == AsyncMode.default:
            gc.collect(1)
            if self.save_future is not None:
                self.save_future.result()
            self.save_future = dcp.async_save(
                state_dict,
                storage_writer=None,
                checkpoint_id=checkpoint_id,
                process_group=self.pg,
            )
            gc.collect(1)
        elif self.async_mode == AsyncMode.mem_pinned:
            gc.collect(1)
            if self.save_future is not None:
                self.save_future.result()
            if self.stager is None:
                self.stager = DefaultStager(StagingOptions(True, True, True, True))
            ret = dcp.async_save(
                state_dict,
                storage_writer=None,
                checkpoint_id=checkpoint_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
            self.save_future = ret.upload_completion
            self.stage_future = ret.staging_completion
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=None,
                checkpoint_id=checkpoint_id,
            )

        logger.info(f"Finished saving checkpoint at step {step} in {time.perf_counter() - begin:.2f} seconds")

    def wait_for_staging(self) -> None:
        # no-op if not using mem_pinned async mode
        if self.stage_future is not None:
            self.stage_future.result()

    def close(self) -> None:
        if self.stager is not None:
            self.stager.close()
