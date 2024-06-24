from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    _use_grad_for_differentiable,
    _get_value,
    _dispatch_sqrt,
    _stack_if_compiling,
    _capturable_doc,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _default_to_fused_or_foreach,
    ParamsT,
    _view_as_real,
)
from typing import List, Optional, Tuple, Union
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

__all__ = ["AdamW", "adamw"]

import bittensor as bt
import torch
from torch.optim.adamw import adamw


class VerboseAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            bt.logging.info(f"params_with_grad before: {params_with_grad[-1][:10]}")
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                has_complex=has_complex,
            )
            bt.logging.info(f"params_with_grad after: {params_with_grad[-1][:10]}")

        return loss
