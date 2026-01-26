import abc

from torch import nn


class BaseModel(nn.Module, abc.ABC):
    # Common interface for all models in Sarasa

    blocks: list[nn.Module]  # TF blocks

    @abc.abstractmethod
    def init_weights(self) -> None:
        # Actual initialization of model weights
        pass

    @abc.abstractmethod
    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        # Return parameter groups for optimizer
        pass
