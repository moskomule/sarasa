from contextlib import AbstractContextManager
from typing import Callable, Iterable

import torch
import torch.distributed as dist

from sarasa.metrics import MetricsProcessor
from sarasa.train import IGNORE_INDEX
from sarasa.utils import world_size


class Evaluator:
    def __init__(
        self,
        val_loader: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
        amp_context: AbstractContextManager,
        metrics_processor: MetricsProcessor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
    ):
        self.val_loader = val_loader
        self.amp_context = amp_context
        self.metrics_processor = metrics_processor
        self.loss_fn = loss_fn
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        step: int,
    ) -> None:
        model.eval()

        loss = 0.0
        num_steps = len(self.val_loader)

        for input_dict, target in self.val_loader:
            self.metrics_processor.ntokens_since_last_log += target.numel()
            input_dict = {
                k: v.to(self.device, non_blocking=(self.device.type == "cuda")) for k, v in input_dict.items()
            }
            target = target.to(self.device, non_blocking=(self.device.type == "cuda"))
            valid_tokens = (target != IGNORE_INDEX).sum()
            if world_size() > 1:
                dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
            with self.amp_context:
                pred = model(**input_dict)
                loss += self.loss_fn(pred, target) / valid_tokens / num_steps

        self.metrics_processor.val_log(step=step, val_loss=loss.item())

        model.train()
