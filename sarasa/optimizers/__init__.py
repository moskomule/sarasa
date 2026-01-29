import dataclasses
from typing import Literal

import torch

from sarasa.models import BaseModel
from sarasa.optimizers.utils import GroupedOptimizer


@dataclasses.dataclass
class AdamW:
    """
    Default optimizer
    """

    lr: float = 1e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)

    def create(
        self,
        model: BaseModel,
    ) -> torch.optim.Optimizer:
        param_groups = model.param_groups()
        params = sum(param_groups.values(), [])
        optimizer = torch.optim.AdamW(
            params,
            lr=torch.tensor(self.lr, dtype=torch.float32),
            weight_decay=self.weight_decay,
            betas=self.betas,
            fused=True,
        )
        return optimizer


@dataclasses.dataclass
class AdamH:
    """
    AdamH optimizer
    """

    lr: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    rescale_to_unit_ball: bool = False

    adam_lr: float = lr
    adam_betas: tuple[float, float] = betas
    adam_weight_decay: float = 0

    def create(
        self,
        model: BaseModel,
    ) -> torch.optim.Optimizer:
        from .hyperball_optimizer import AdamH as Optimizer

        param_groups = model.param_groups()
        adamh = Optimizer(
            param_groups["matrix"],
            lr=self.lr,
            betas=self.betas,
            rescale_to_unit_ball=self.rescale_to_unit_ball,
        )
        adam = torch.optim.AdamW(
            sum([param_groups[k] for k in param_groups if k != "matrix"], []),
            lr=self.adam_lr,
            betas=self.adam_betas,
            weight_decay=self.adam_weight_decay,
            fused=True,
        )

        return GroupedOptimizer(adamh, adam)


@dataclasses.dataclass
class Muon:
    """
    Muon optimizer
    """

    lr: float = 1e-4
    weight_decay: float = 0.1
    momentum: float = 0.9

    adam_lr: float = lr
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_weight_decay: float = 0

    adjust_lr_fn: Literal["original", "match_rms_adamw"] = "match_rms_adamw"

    def create(
        self,
        model: BaseModel,
    ) -> torch.optim.Optimizer:
        param_groups = model.param_groups()

        muon = torch.optim.Muon(
            param_groups["matrix"],
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            adjust_lr_fn=self.adjust_lr_fn,
        )

        adam = torch.optim.AdamW(
            sum([param_groups[k] for k in param_groups if k != "matrix"], []),
            lr=self.adam_lr,
            betas=self.adam_betas,
            weight_decay=self.adam_weight_decay,
            fused=True,
        )

        return GroupedOptimizer(muon, adam)


@dataclasses.dataclass
class MuonH:
    """
    MuonH optimizer
    """

    lr: float = 1e-2
    momentum: float = 0.9
    rescale_to_unit_ball: bool = False

    adam_lr: float = lr
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_weight_decay: float = 0

    adjust_lr_fn: Literal["original", "match_rms_adamw"] = "match_rms_adamw"

    def create(
        self,
        model: BaseModel,
    ) -> torch.optim.Optimizer:
        from .hyperball_optimizer import MuonH as Optimizer

        param_groups = model.param_groups()

        muon = Optimizer(
            param_groups["matrix"],
            lr=self.lr,
            momentum=self.momentum,
            rescale_to_unit_ball=self.rescale_to_unit_ball,
            adjust_lr_fn=self.adjust_lr_fn,
        )

        adam = torch.optim.AdamW(
            sum([param_groups[k] for k in param_groups if k != "matrix"], []),
            lr=self.adam_lr,
            betas=self.adam_betas,
            weight_decay=self.adam_weight_decay,
            fused=True,
        )

        return GroupedOptimizer(muon, adam)
