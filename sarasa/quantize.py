import torch

from sarasa.models import BaseModel


def to_float8(
    model: BaseModel,
) -> None:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    # optional: filter modules from being eligible for float8 conversion
    def module_filter_fn(mod: torch.nn.Module, fqn: str):
        # don't convert the last module
        if fqn == "1":
            return False
        # don't convert linear modules with weight dimensions not divisible by 16
        if isinstance(mod, torch.nn.Linear):
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
        return True

    config = Float8LinearConfig.from_recipe_name("tensorwise")

    convert_to_float8_training(model, config=config, module_filter_fn=module_filter_fn)
