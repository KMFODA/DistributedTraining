from hivemind.optim.state_averager import (
    LRSchedulerBase,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from typing import Optional


class DTStateAverager(TrainingStateAverager):
    def __init__(
        self,
        *,
        num_inner_steps: int,
        inner_optimizer: TorchOptimizer,
        scheduler: Optional[SchedulerFactory] = None,
        **kwargs,
    ):
        self.inner_optimizer = inner_optimizer
        self.num_inner_steps = num_inner_steps

        super().__init__(
            **kwargs
        )  # we specifically don't pass the scheduler here, default TrainingStateAverager would use it with the outer optimizer and we w

        self.scheduler_inner_optimizer = (
            scheduler(self.inner_optimizer) if scheduler is not None else None
        )
        assert isinstance(self.scheduler_inner_optimizer, (LRSchedulerBase, type(None)))

    def _update_scheduler(self):
        """Increase the scheduler state until it becomes synchronized with local epoch"""
        # TODO(sami) handle update scheduler
        # for now assuming that all scheduler are on time
        pass

    def update_main_param_after_outer_step(self):
        """Update the main parameters with the inner optimizer step"""
        opt_parameters = [
            param for group in self.optimizer.param_groups for param in group["params"]
        ]
        for main_param, opt_param in zip(self.main_parameters, opt_parameters):
            main_param.data.copy_(opt_param.data, non_blocking=True)