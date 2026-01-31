"""MuonH and AdamH optimizers
@online{wen2025hyperball,
  title   = {Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization},
  author  = {Wen, Kaiyue and Dang, Xingyu and Lyu, Kaifeng and Ma, Tengyu and Liang, Percy},
  year    = {2025},
  month   = {12},
  day     = {15},
  url     = {https://tinyurl.com/muonh},
  urldate = {2025-12-15}
}

The base code is adapted from PyTorch 2.10.0's Adam and Muon optimizer implementation
"""

import torch
from loguru import logger
from torch import Tensor
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _get_value,
    _stack_if_compiling,
    _to_scalar,
)


def _hyperball_init(
    optimizer: torch.optim.Optimizer,
    rescale_to_unit_ball: bool,
    eps: float,
) -> None:
    # reusable initialization for Hyperball optimizers

    # check if optimizer is already initialized
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            if "initial_norm" in state:
                return

    # check if optimizer's weight decay is zero
    for group in optimizer.param_groups:
        if group.get("weight_decay", 0.0) != 0.0:
            logger.warning(
                "Hyperball does not support weight decay in the base optimizer. "
                "Please set weight_decay=0.0 in the base optimizer."
            )

    # check if all parameters are 2D matrices
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.ndim != 2:
                raise ValueError("Hyperball only supports 2D matrix parameters")

    # rescale initial parameters to have norm 1 or store their initial norms
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                norm = param.norm(dim=0, keepdim=True).clamp_min(eps)
                state = optimizer.state[param]
                if rescale_to_unit_ball:
                    param.div_(norm)
                    state["initial_norm"] = torch.ones_like(norm)
                else:
                    state["initial_norm"] = norm


class AdamH(torch.optim.Adam):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
        eps: float = 1e-8,
        *,
        rescale_to_unit_ball: bool = False,
    ) -> None:
        # rescale_to_unit_ball: if True, rescale all parameters to lie on the unit hyperball at initialization
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.defaults["rescale_to_unit_ball"] = rescale_to_unit_ball

        for group in self.param_groups:
            group.setdefault("rescale_to_unit_ball", rescale_to_unit_ball)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        initial_norms,
    ):
        super()._init_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )
        _hyperball_init(self, rescale_to_unit_ball=group["rescale_to_unit_ball"], eps=group["eps"])
        for p in params_with_grad:
            state = self.state[p]
            initial_norms.append(state["initial_norm"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            initial_norms: list[Tensor] = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                initial_norms,
            )

            _multi_tensor_adamh(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                eps=group["eps"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                initial_norms=initial_norms,
            )

        return loss


def _multi_tensor_adamh(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    initial_norms: list[Tensor],
    grad_scale: Tensor | None,
    found_inf: Tensor | None,
    *,
    beta1: float | Tensor,
    beta2: float | Tensor,
    lr: float | Tensor,
    eps: float,
) -> None:
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

    if isinstance(beta1, Tensor):
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")

    if isinstance(beta2, Tensor):
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")

    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    lr = _to_scalar(lr)
    beta1 = _to_scalar(beta1)
    beta2 = _to_scalar(beta2)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    # We only shuffle around the beta when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    beta1_dict = (  # type: ignore[attr-defined]
        {beta1.device: beta1} if isinstance(beta1, Tensor) and str(beta1.device) != "cpu" else None
    )

    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_state_steps,
    ), _ in grouped_tensors.values():
        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)  # type: ignore[union-attr, attr-defined]

        device_beta1 = beta1_dict[device] if beta1_dict else beta1

        torch._foreach_add_(device_state_steps, 1)

        torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - device_beta1)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)

        if isinstance(beta2, torch.Tensor):
            scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)  # type: ignore[assignment]
            value = 1.0
        else:
            scaled_device_grads = device_grads  # type: ignore[assignment]
            value = 1 - beta2

        torch._foreach_addcmul_(device_exp_avg_sqs, scaled_device_grads, device_grads, value)

        # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
        del device_grads
        del scaled_device_grads

        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]

        step_size = _stack_if_compiling([(lr / bc) * -1 for bc in bias_correction1])

        bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]  # type: ignore[arg-type]

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
        torch._foreach_add_(exp_avg_sq_sqrt, eps)
        updates = [t.clone() for t in device_exp_avgs]
        # device_exp_avgs / exp_avg_sq_sqrt
        torch._foreach_div_(updates, exp_avg_sq_sqrt)

        # hyperball update

        # normalize updates to lie on the hyperball
        norms = torch._foreach_norm(updates, 2)
        torch._foreach_clamp_min_(norms, eps)
        torch._foreach_div_(updates, norms)
        # scale by initial norms
        torch._foreach_mul_(updates, initial_norms)
        # update parameters
        torch._foreach_mul_(updates, step_size)
        torch._foreach_add_(device_params, updates)

        # normalize parameters to lie on the hyperball
        norms = torch._foreach_norm(device_params, 2)
        torch._foreach_clamp_min_(norms, eps)
        torch._foreach_div_(device_params, norms)
        # scale by initial norms
        torch._foreach_mul_(device_params, initial_norms)


class MuonH(torch.optim.Muon):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        rescale_to_unit_ball: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=0,
            momentum=momentum,
            nesterov=nesterov,
            rescale_to_unit_ball=rescale_to_unit_ball,
            **kwargs,
        )
        for group in self.param_groups:
            group.setdefault("rescale_to_unit_ball", rescale_to_unit_ball)

    def _init_group(
        self,
        group,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
        initial_norms: list[Tensor],
    ) -> None:
        super()._init_group(
            group,
            params_with_grad,
            grads,
            muon_momentum_bufs,
        )
        _hyperball_init(self, rescale_to_unit_ball=group["rescale_to_unit_ball"], eps=group["eps"])
        for p in params_with_grad:
            state = self.state[p]
            initial_norm = state["initial_norm"]
            initial_norms.append(initial_norm)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []
            initial_norms: list[Tensor] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
                initial_norms,
            )

            _single_tensor_muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                initial_norms,
                lr=lr,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
            )
        return loss


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    initial_norms: list[Tensor],
    *,
    lr: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
) -> None:
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        # normalize update to lie on the hyperball
        norm = update.norm(dim=0, keepdim=True).clamp_min(eps)
        update.div_(norm)
        update.mul_(initial_norms[i])

        param.add_(update, alpha=-adjusted_lr)

        # normalize parameters to lie on the hyperball
        norm = param.norm(dim=0, keepdim=True).clamp_min(eps)
        param.div_(norm)
        param.mul_(initial_norms[i])
