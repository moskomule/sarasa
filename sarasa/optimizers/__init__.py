import dataclasses

import torch

from sarasa.models import BaseModel


@dataclasses.dataclass
class AdamWConfig:
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
