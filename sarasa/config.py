from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Literal

import torch

"""
Variable configuration dataclasses for model, optimizer, lr scheduler, and data
These classes have `create` methods to instantiate the actual objects

Users can define their own configuration dataclasses and pass them to Config.from_cli to use custom components
"""
from sarasa.data import DataConfig as Data  # noqa
from sarasa.models import ModelConfig as Model  # noqa
from sarasa.optimizers import AdamW  # noqa


@dataclasses.dataclass
class LRScheduler:
    warmup_steps: int = 200
    decay_ratio: float | None = None
    """If set, the ratio of total steps to apply decay after warmup. If None, decay starts immediately after warmup."""

    decay_type: Literal["linear", "cosine", "sqrt"] = "linear"
    min_lr_factor: float = 0.0

    def create(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        assert self.decay_ratio is None or (0 <= self.decay_ratio <= 1), "decay_ratio must be between 0 and 1"
        warmup_steps = self.warmup_steps
        stay_steps = 0 if self.decay_ratio is None else int(total_iters * (1 - self.decay_ratio)) - warmup_steps
        decay_steps = total_iters - warmup_steps - stay_steps
        assert warmup_steps >= 0 and decay_steps >= 0 and stay_steps >= 0, (
            f"Invalid lr scheduler steps configuration: {warmup_steps=}, {decay_steps=}, {stay_steps=}"
        )

        # 1 / max(1, warmup_steps) to avoid division by zero
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, 1 / max(1, warmup_steps), total_iters=warmup_steps)

        stay = torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1.0, total_iters=stay_steps)

        match self.decay_type:
            case "linear":
                decay = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.min_lr_factor,
                    total_iters=decay_steps,
                )
            case "sqrt":
                decay = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: max(
                        self.min_lr_factor,
                        (decay_steps - step) / decay_steps,
                    )
                    ** 0.5,
                )
            case "cosine":
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=decay_steps,
                    eta_min=optimizer.param_groups[0]["lr"] * self.min_lr_factor,
                )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmup, stay, decay],
            [self.warmup_steps, self.warmup_steps + stay_steps],
        )
        return scheduler


"""
Static configuration dataclasses

These classes are not expected to be changed by the user
"""


@dataclasses.dataclass
class Train:
    steps: int = 10_000

    grad_clip: float | None = None

    dtype: Literal["bfloat16", "float32"] = "float32"

    compile: bool = False

    gc_freq: int = 50
    """Garbage collection frequency (in steps). If -1, no periodic GC is performed."""

    local_batch_size: int = 32
    """local (per device) batch size"""

    global_batch_size: int = 256
    """
    global (across all devices) batch size, used to compute 
    grad_accum_steps = global_batch_size // (local_batch_size * num_devices)
    """

    use_fa4: bool = True
    """Whether to use FA4 flash attention if available."""

    val_freq: int = -1
    """Validation frequency (in steps). If -1, no validation is performed."""

    use_sac: bool = False
    """Whether to use selective activation checkpointing."""


@dataclasses.dataclass
class Metrics:
    log_freq: int = 10
    use_tensorboard: bool = False
    all_node: bool = False


@dataclasses.dataclass
class Checkpoint:
    save_freq: int = 1000
    async_mode: Literal["none", "default", "mem_pinned"] = "default"


@dataclasses.dataclass
class Distributed:
    backend: Literal["nccl", "gloo"] = "nccl"

    init_timeout_seconds: int = 300
    """Timeout for initializing the distributed process group."""

    train_timeout_seconds: int = 100
    """Timeout for distributed training operations after the first iteration."""

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()


@dataclasses.dataclass
class DDP(Distributed):
    pass


@dataclasses.dataclass
class FSDP(Distributed):
    reshard_after_forward: bool = False
    """Whether to reshard model parameters after each forward pass (FSDP only)."""


@dataclasses.dataclass
class Config[ModelT, OptimizerT, LRSchedulerT, DataT]:
    # variable components
    model: ModelT
    optim: OptimizerT
    lr_scheduler: LRSchedulerT
    data: DataT

    # static components
    train: Train = dataclasses.field(default_factory=Train)
    metrics: Metrics = dataclasses.field(default_factory=Metrics)
    checkpoint: Checkpoint = dataclasses.field(default_factory=Checkpoint)
    distributed: DDP | FSDP = dataclasses.field(default_factory=DDP)

    seed: int = 0
    debug: bool = False
    """ Enable debug mode with more verbose logging and checks."""

    output_dir: Path | str = Path("./outputs")
    """Directory to save checkpoints and logs."""

    config_file: Path | str | None = None
    """Path to a config file (JSON or TOML) to load configuration from."""

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "seq_len") and self.model.seq_len is None:
            if self.data.seq_len is not None:
                self.model.seq_len = self.data.seq_len
            else:
                raise ValueError("Either model.seq_len or data.seq_len must be set.")

    @classmethod
    def create(
        cls,
        model: ModelT,
        optim: OptimizerT,
        lr_scheduler: LRSchedulerT,
        data: DataT,
        **kwargs,
    ) -> Config:
        return cls(
            model=model,
            optim=optim,
            lr_scheduler=lr_scheduler,
            data=data,
            **kwargs,
        )

    @classmethod
    def from_cli(
        cls,
        *,
        model_type: ModelT = Model,
        optim_type: OptimizerT = AdamW,
        lr_scheduler_type: LRSchedulerT = LRScheduler,
        data_type: DataT = Data,
    ) -> Config:
        """
        initialize JobConfig from command line arguments
        update the values with the following priority: CLI arguments > config file > defaults

        *_type can be used to specify custom dataclass types for each section
        >> config = Config.from_cli(optim_type=CustomOptimizerConfig)
        """

        import importlib.util

        import tyro

        loaded_config = None

        if (under := ("--config_file" in sys.argv)) or ("--config-file" in sys.argv):
            config_file = sys.argv[sys.argv.index("--config_file" if under else "--config-file") + 1]
            config_file = Path(config_file)

            if not config_file.exists():
                raise FileNotFoundError(f"Config file {config_file} does not exist.")

            if config_file.suffix != ".py":
                raise ValueError("Only Python config files are supported in this method.")

            spec = importlib.util.spec_from_file_location("custom_config", config_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            configs = [
                config
                for config in module.__dict__.values()
                if isinstance(config, cls) and not isinstance(config, type)
            ]
            if len(configs) == 0:
                raise ValueError(f"No Config instance found in {config_file}.")
            elif len(configs) > 1:
                raise ValueError(f"Multiple Config instances found in {config_file}. Please keep only one.")
            else:
                loaded_config = configs[0]

        return tyro.cli(
            cls[
                model_type,
                optim_type,
                lr_scheduler_type,
                data_type,
            ],
            default=loaded_config,
        )


__all__ = [
    "Config",
    "Model",
    "AdamW",
    "LRScheduler",
    "Data",
    "Train",
    "Metrics",
    "Checkpoint",
    "DDP",
    "FSDP",
]
