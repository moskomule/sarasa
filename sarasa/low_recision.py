import torch
import torch.nn as nn


@torch.no_grad()
def to_te_linear_(
    module: nn.Module,
) -> nn.Module:
    """
    Recursively converts all nn.Linear layers in a module to TE's float-8/4 ready ones.
    """
    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )

            setattr(module, name, new_layer)
        else:
            to_te_linear_(child)

    return module
