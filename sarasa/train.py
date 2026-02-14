import contextlib
import os
import time
from typing import Iterator

import torch
import torch.distributed as dist
from loguru import logger
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn import functional as F

from sarasa.activation_checkpoint import apply_op_sac
from sarasa.checkpoint import Checkpointer
from sarasa.config import Config
from sarasa.evaluate import Evaluator
from sarasa.metrics import MetricsProcessor
from sarasa.utils import (
    IGNORE_INDEX,
    GarbageCollector,
    apply_distributed,
    init_distributed,
    set_dtype,
    setup_logger,
    setup_profiler,
    update_timeout,
    world_size,
)


class Trainer:
    @record
    def __init__(
        self,
        config: Config,
    ) -> None:
        self.config = config

        # only rank 0 logs
        setup_logger(config)

        logger.info(f"Initializing Trainer with config: {self.config}")

        # set seed
        torch.manual_seed(config.seed)
        os.environ["PYTHONHASHSEED"] = str(config.seed % 2**32)

        # setup device
        torch.accelerator.set_device_index(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")

        self.gc = GarbageCollector(config.train.gc_freq)

        # setup distributed
        init_distributed(config.distributed.backend, config.distributed.init_timeout_seconds)

        # setup data and tokenizer -> use vocab size for model setup
        data = config.data.create(
            batch_size=config.train.local_batch_size,
            val_cfg=config.evaluate,
            use_varlen=config.model.attn_type == "varlen",
        )
        self.data_loader = data["train_loader"]  # setup data loader
        val_loader = data.get("val_loader", None)

        self.config.model.vocab_size = len(data["tokenizer"])

        # setup model, optimizer, lr scheduler
        with torch.device("meta"), set_dtype(getattr(torch, config.train.dtype)):
            self.model = self.config.model.create()
            num_params, flops_per_token = self.model.num_params_flops
            model_size = num_params / 1e9
            model_size, unit = (num_params / 1e6, "M") if model_size < 1 else (model_size, "B")
            logger.info(f"Model created with {model_size:.2f}{unit} parameters")

        # following torchtitan, (S)AC -> compilation -> distributed wrapping
        if config.train.use_sac:
            logger.info("Applying Selective Activation Checkpointing (SAC)")
            for i, block in enumerate(self.model.blocks):
                self.model.blocks[i] = apply_op_sac(block)

        if config.train.compile:
            logger.info("Compiling the model")
            for block in self.model.blocks:
                block.compile(fullgraph=True)
            self.model.compile()
            self.loss_fn = torch.compile(self.loss_fn, fullgraph=True, dynamic=False)

        if world_size() > 1:
            apply_distributed(
                config.distributed,
                self.model,
                device=self.device,
                compile=config.train.compile,
            )

        self.model.to_empty(device=self.device)
        self.model.init_weights()

        self.optimizer = self.config.optim.create(self.model)
        self.lr_scheduler = self.config.lr_scheduler.create(self.optimizer, config.train.steps)

        self.step = 0
        self.grad_accum_steps = config.train.global_batch_size // (config.train.local_batch_size * world_size())
        if self.grad_accum_steps > 1:
            logger.info(f"Gradient accumulation step is set to: {self.grad_accum_steps}")

        self.amp_context: contextlib.AbstractContextManager = contextlib.nullcontext()
        if config.distributed.name != "fsdp":
            self.amp_context = torch.autocast(
                device_type=self.device.type,
                dtype=getattr(torch, config.train.amp_dtype),
            )

        self.profile_context = setup_profiler(self.config.profile, self.device, save_dir=self.config.output_dir)

        # setup metrics, checkpointer, evaluator
        self.metrics_processor = MetricsProcessor(config, self.device, flops_per_token)
        self.checkpointer = Checkpointer(config, self.model) if config.checkpoint.freq > 0 else None
        self.evaluator = (
            Evaluator(config.evaluate, val_loader, self.amp_context, self.metrics_processor, self.loss_fn, self.device)
            if val_loader is not None
            else None
        )

        dev_mem_stats = self.metrics_processor.device_mem_monitor.get_peak_stats()
        logger.info(
            f"{self.device.type.upper()} memory: {dev_mem_stats.max_reserved_gib:.2f} GiB for model initialization"
        )

        if config.train.use_fa4:
            logger.info("Using FA4 flash attention")
            try:
                torch.nn.attention.activate_flash_attention_impl("FA4")
            except Exception as e:
                logger.warning(
                    f"Failed to activate FA4 flash attention: {e}. Install sarasa with `flash_attn` extra for better performance."
                )

    @record
    def train(self):
        try:
            logger.info("Starting training...")

            self.model.train()
            with self.profile_context as profiler:
                data_iter = self.batch_generator(self.data_loader)
                for _ in range(self.config.train.steps):
                    self.step += 1
                    self.gc.collect(self.step)
                    try:
                        self.train_step(data_iter)
                    except StopIteration:
                        logger.warning("Data loader exhausted during training.")
                        break

                    if self.checkpointer is not None and self.checkpointer.trigger(self.step):
                        self.checkpointer.save(self.step)

                    if self.evaluator is not None and self.evaluator.trigger(self.step):
                        logger.info("Starting evaluation...")
                        self.evaluator.evaluate(self.model, self.step)

                    profiler.step()

                    if world_size() > 1 and self.step == 1:
                        update_timeout(self.config.distributed.train_timeout_seconds, self.device)

            logger.info("Training completed.")
        finally:
            logger.info("Cleaning up trainer...")
            self.close()

    def batch_generator(
        self,
        data_iter: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        data_iter = iter(data_iter)
        while True:
            begin = time.perf_counter()
            batch = next(data_iter)
            input_dict, target = batch
            self.metrics_processor.ntokens_since_last_log += target.numel()
            self.metrics_processor.data_load_times.append(time.perf_counter() - begin)
            yield input_dict, target

    def train_step(
        self,
        batch_iter: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> None:
        self.optimizer.zero_grad()

        micro_batches = []
        valid_tokens = torch.tensor(0, dtype=torch.long)
        for _ in range(self.grad_accum_steps):
            input_dict, target = next(batch_iter)
            valid_tokens += (target != IGNORE_INDEX).sum()
            micro_batches.append((input_dict, target))

        valid_tokens = valid_tokens.to(self.device)
        if world_size() > 1:
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)

        losses = []
        for input_dict, target in micro_batches:
            input_dict = {
                k: v.to(self.device, non_blocking=(self.device.type == "cuda")) for k, v in input_dict.items()
            }
            target = target.to(self.device, non_blocking=(self.device.type == "cuda"))

            with self.amp_context:
                pred = self.model(**input_dict)
                loss = self.loss_fn(pred, target) / valid_tokens

            del pred
            loss.backward()
            losses.append(loss.detach())

        if self.config.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.train.grad_clip,
                foreach=self.device.type == "cuda",
            )

        if self.checkpointer is not None:
            self.checkpointer.wait_for_staging()

        self.optimizer.step()
        self.lr_scheduler.step()

        loss = torch.stack(losses).sum()

        if not self.metrics_processor.trigger(self.step):
            return

        if world_size() > 1:
            avg_loss = loss.clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            max_loss = loss.clone()
            dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)
        else:
            avg_loss = max_loss = loss

        with torch.no_grad():
            grad_norm = torch.nn.utils.get_total_norm(self.model.parameters(), foreach=self.device.type == "cuda")

        lr = self.lr_scheduler.get_last_lr()[0]
        self.metrics_processor.log(
            self.step,
            global_avg_loss=avg_loss.item(),
            global_max_loss=max_loss.item(),
            extra_metrics={
                "grad_norm": grad_norm.item() if grad_norm >= 0 else float("nan"),
                "lr": lr,
            },
        )

    def loss_fn(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            pred.flatten(0, 1).float(),
            target.flatten(0, 1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

    def close(self) -> None:
        if self.checkpointer is not None:
            self.checkpointer.close()

        if self.metrics_processor is not None:
            self.metrics_processor.close()

        # cleanup distributed
        if world_size() > 1:
            try:
                dist.destroy_process_group()
            except Exception as e:
                logger.warning(f"Failed to destroy process group: {e}")

    def __del__(self):
        self.close()
