import asyncio
import logging
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
)

import torch
from hivemind.averaging.allreduce import (
    AveragingMode,
    AllReduceRunner
)
from hivemind.averaging.averager import DecentralizedAverager
from hivemind.averaging.control import StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo
from hivemind.dht import DHT
from hivemind.dht.dht import DHT
from hivemind.optim.state_averager import (
    LRSchedulerBase,
    SchedulerFactory,
    TorchOptimizer,
    TrainingStateAverager,
)
from hivemind.p2p import P2PContext
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import (
    as_aiter,
    azip,
    enter_asynchronously,
)
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.timed_storage import DHTExpiration, get_dht_time

GatheredData = Any

hivemind_logger = get_logger(__name__)
hivemind_logger.setLevel(logging.INFO)


class DTGradAverager(DecentralizedAverager):
    """ "
    DiLoCoGradAverager is meant to be used in pair with DiLoCoStateAverager. Specifically it takes as input the offloaded optimizer of DiLoCoStateAverager, and
    use the grad buffer of the offloaded param as averaged_tensors for the DecentralizedAverager. In other words the DiLoCoGradAverager makes sure that the grad of the offloaded optimizer
    are kept in sync between peers.
    """

    def __init__(
        self,
        main_parameters: List[torch.nn.Parameter],
        offloaded_optimizer: TorchOptimizer,
        *,
        dht: DHT,
        prefix: str,
        warn: bool = True,
        **kwargs,
    ):
        if "client_mode" in kwargs:
            if kwargs["client_mode"] is not None and kwargs["client_mode"]:
                raise KeyError("client_mode is not supported in DiLoCoGradAverager")
            else:
                kwargs.pop("client_mode")

        if "averaged_grads" in kwargs:
            raise KeyError(
                "DiLoCoGradAverager does not support averaged_grads since it use the offloaded optimizer gradients directly"
            )

        if not isinstance(main_parameters, (list, tuple)):
            raise ValueError(
                "main_parameters must be a list or tuple of torch.nn.Parameter and not an iterator otherwise parameters will be consumed"
            )
        self.main_parameters = list(main_parameters)
        self.offloaded_optimizer = offloaded_optimizer

        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0

        self._new_averaged_grads = False

        averaged_grads = tuple(grad for grad in self._grads_from_optimizer())

        super().__init__(
            averaged_tensors=averaged_grads,
            dht=dht,
            prefix=prefix,
            client_mode=False,
            **kwargs,
        )

    def _grads_from_optimizer(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated optimizer"""
        param_groups = self.offloaded_optimizer.param_groups
        for param_group in param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                yield param.grad

    def schedule_step(self, scheduled_time: Optional[DHTExpiration] = None, **kwargs) -> StepControl:
        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        assert kwargs.get("weight") is None, "setting weight in schedule_step is not supported"
        return super().step(scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs)

    def step(
        self,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        **kwargs,
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        """
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)

        self.compute_and_load_pseudo_grad_into_averager()
        control.allow_allreduce()

        return control.result(timeout) if wait else control

    @torch.no_grad()
    def compute_and_load_pseudo_grad_into_averager(self):
        """compute pseudo gradient by subtracting the offloaded optimizer parameters with the main parameters and load them in the averager"""
        opt_parameters = [param for group in self.offloaded_optimizer.param_groups for param in group["params"]]
        with self.get_tensors() as averaged_grads:
            for opt_param, averaged_grad, main_param in zip(opt_parameters, averaged_grads, self.main_parameters):
                # opt_param is the param that will be all_reduce, it is suppose to be on cpu
                # main_param is the param that has been updated by the inner optimizer, it is suppose to be on gpu
                grad = opt_param.data - main_param.detach().to(opt_param.device)
                averaged_grad.copy_(grad, non_blocking=True)

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False



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
    
    async def _aggregate_with_group(
        self,
        group_info: GroupInfo,
        min_vector_size: int,
        peerids_to_uids: Dict,
        **kwargs,
    ) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata
        This wrapper method helps allow return of banned senders for better visibility of global contributions"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(
                *map(self.serializer.loads, group_info.gathered)
            )
            user_gathered = dict(
                zip(
                    group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)
                )
            )
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0
                for thr, mode in zip(bandwidths, modes)
            ]
            # hivemind_logger("Donwloaded bandwidths")
            # hivemind_logger(download_bandwidths)
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None,
                load_balance_peers,
                self.total_size,
                download_bandwidths,
                min_vector_size,
            )
            hivemind_logger(group_info.peer_ids)
            hivemind_logger(peer_fractions)
            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = AllReduceRunner(
                    peerids_to_uids=peerids_to_uids,
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    **kwargs,
                )

                self._running_groups[group_info.group_id].set_result(runner)

                if (
                    runner.modes[group_info.peer_ids.index(self.peer_id)]
                    != AveragingMode.AUX
                ):
                    async for tensor, update in azip(as_aiter(*local_tensors), runner):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in runner:
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return user_gathered, runner.banned_senders, group_info.peer_ids
        except BaseException as e:
            if isinstance(e, Exception):
                hivemind_logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")
            
    async def rpc_download_state_partial(
        self, _request: averaging_pb2.DownloadRequest, _context: P2PContext
    ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (serialized_metadata, tensors)

         - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        hivemind_logger.info("rpc_download_state_partial")
        if not self.allow_state_sharing:
            return  # deny request and direct peer to the next prospective averager
        metadata, tensors, infos = await self._get_current_state_from_host_process()
        hivemind_logger.info(len(tensors))
        if infos is None:
            infos = [
                CompressionInfo.from_tensor(tensor, key=i)
                for i, tensor in enumerate(tensors)
            ]
        assert len(tensors) == len(infos)

        # for tensor, info in zip([tensors[0]], infos):
        for tensor, info in zip([tensors[0]], infos):
            for part in split_for_streaming(
                self.state_compression.compress(tensor, info, allow_inplace=False)
            ):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(
                        tensor_part=part, metadata=metadata
                    )
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)
            break