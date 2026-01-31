from torch import nn


def convert_to_float8_(model: nn.Module) -> None:
    """
    Convert the model's parameters and buffers to float8 in-place using torchao
    """

    from torchao.float8 import convert_to_float8_training

    def _filter(m: nn.Module, fqn: str) -> bool:
        if fqn == "1":
            return False
        if isinstance(m, (nn.Linear)):
            if m.in_features % 16 != 0 or m.out_features % 16 != 0:
                return False
        return True

    convert_to_float8_training(model, module_filter_fn=_filter)
