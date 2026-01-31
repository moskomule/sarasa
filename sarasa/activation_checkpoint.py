from collections import defaultdict

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

# for selective op activation checkpointing
_ops_sac_save = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.inductor_compiled_code,
}


def _op_sac_policy(
    ops_to_save: set,
    mm_recompute_shapes: set | None,
    every_nth_mm: int,
):
    mm_recompute_shapes = mm_recompute_shapes or set()

    def _get_custom_policy(meta: dict):
        def _custom_policy(ctx, func, *args, **kwargs):
            # special case, offload to CPU
            if (
                func == torch.ops.aten._to_copy.default
                and "cuda" in str(args[0].device)
                and str(kwargs.get("device", "")) == "cpu"
            ):
                return CheckpointPolicy.MUST_SAVE

            # track mm ops
            mode = "recompute" if ctx.is_recompute else "forward"
            key = f"{mode}_mm_count"

            if func == torch.ops.aten.mm.default:
                if len(args) > 1 and args[1].shape in mm_recompute_shapes:
                    # moe's router
                    return CheckpointPolicy.PREFER_RECOMPUTE
                meta[key] += 1

            # save ops in save list, except every nth mm op
            must_save = (func in ops_to_save) and not (
                func == torch.ops.aten.mm.default and (meta[key] % every_nth_mm == 0)
            )
            return CheckpointPolicy.MUST_SAVE if must_save else CheckpointPolicy.PREFER_RECOMPUTE

        return _custom_policy

    def selective_checkpointing_context_fn():
        return create_selective_checkpoint_contexts(_get_custom_policy(defaultdict(int)))

    return selective_checkpointing_context_fn


def apply_op_sac(
    model: torch.nn.Module,
    ops_to_save: set | None = None,
    mm_recompute_shapes: set | None = None,
    every_nth_mm: int = 2,
) -> torch.nn.Module:
    """Applies selective op activation checkpointing to the given model.

    Ops like mm is expensive, so we want to store their activations for backward.
    On the other hand, ops like activation functions are cheap, so we prefer to recompute them.

    """
    ops_to_save = ops_to_save or _ops_sac_save
    return checkpoint_wrapper(
        model,
        _op_sac_policy(ops_to_save, mm_recompute_shapes, every_nth_mm),
    )
