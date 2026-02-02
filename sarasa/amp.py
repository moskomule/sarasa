import contextlib
import enum
from typing import Callable, Literal

import torch
import torch.nn as nn

from sarasa.models.attention import CausalSelfAttention


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
    backend: Literal["torchao", "transformer_engine"],
) -> nn.Module:
    """
    Recursively converts all nn.Linear layers in a module to TE's float-8/4 ready ones.
    """
    if backend == "torchao":
        from torchao.float8 import convert_to_float8_training

        def fliter(m: nn.Module, fqn: str) -> bool:
            if fqn == "1":
                return False
            if isinstance(m, nn.Linear) and (m.in_features % 16 != 0 or m.out_features % 16 != 0):
                return False
            return True

        convert_to_float8_training(module, module_filter_fn=fliter)
        return module

    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        match child:
            case nn.Linear():
                setattr(
                    module,
                    name,
                    te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                    ),
                )
            case CausalSelfAttention():
                setattr(
                    child,
                    "attn",
                    te.attention.DotProductAttention(
                        num_attention_heads=child.num_heads,
                        kv_channels=child.head_dim,
                        num_gqa_groups=None,
                        qkv_format="bshd",
                    ),
                )
                setattr(child, "ht_transpose", False)
            case _:
                to_te_linear_(child)

    return module


def amp_context(
    dtype: AMPDtype,
    device: torch.device,
    is_fsdp: bool,
    fp8_backend: Literal["torchao", "transformer_engine"],
) -> Callable[[], contextlib.AbstractContextManager]:
    def wrapped():
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

        if fp8_backend == "torchao":
            assert dtype == AMPDtype.fp8, "Only fp8 is supported with torchao backend."

            return contextlib.nullcontext()

        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling, NVFP4BlockScaling

        match dtype:
            case AMPDtype.fp8:
                assert te.is_fp8_available(), "FP8 is not available on this system."
                # E4M3 during forward pass, E5M2 during backward pass
                recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
            case AMPDtype.mxfp8:
                assert te.is_mxfp8_available(), "MXFP8 is not available on this system."
                recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
            case AMPDtype.nvfp4:
                assert te.is_nvfp4_available(), "NVFP4 is not available on this system."
                recipe = NVFP4BlockScaling()

        return te.autocast(recipe=recipe)

    return wrapped
