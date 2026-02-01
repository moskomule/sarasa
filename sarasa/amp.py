import contextlib
import enum

import torch
import torch.nn as nn


class AMPDtype(enum.StrEnum):
    bfloat16 = enum.auto()
    float16 = enum.auto()
    float32 = enum.auto()
    fp8 = enum.auto()
    mxfp8 = enum.auto()
    nvfp4 = enum.auto()


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


def amp_context(
    dtype: AMPDtype,
    device: torch.device,
    is_fsdp: bool,
) -> contextlib.AbstractContextManager:
    if dtype in {
        AMPDtype.bfloat16,
        AMPDtype.float16,
        AMPDtype.float32,
    }:
        if is_fsdp:
            return contextlib.nullcontext()
        else:
            return torch.autocast(
                device_type=device.type,
                dtype=getattr(torch, dtype.value),
            )

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling, NVFP4BlockScaling

    match dtype:
        case AMPDtype.fp8:
            # E4M3 during forward pass, E5M2 during backward pass
            recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
        case AMPDtype.mxfp8:
            recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
        case AMPDtype.nvfp4:
            recipe = NVFP4BlockScaling()

    return te.autocast(recipe=recipe)
