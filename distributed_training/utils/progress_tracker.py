from dataclasses import dataclass
from distributed_training import __run__

import bittensor as bt
import pandas as pd
from huggingface_hub import list_repo_refs
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint
from tqdm import tqdm

import wandb


@dataclass(frozen=False)
class GlobalTrainingProgress:
    epoch: int
    samples_accumulated: int


class LocalTrainingProgress(BaseModel):
    peer_id: bytes
    epoch: conint(ge=0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    time: StrictFloat
    client_mode: StrictBool
    inner_step: conint(ge=0, strict=True)
    loss: confloat(ge=0.0, strict=True)


def get_global_epoch(self):
    try:
        refs = list_repo_refs(self.config.neuron.global_model_name, repo_type="model")
        global_epoch = (
            max(
                [
                    int(tag.name.split(".")[1])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                    )
                ]
            )
            if refs.tags
            else 0
        )
        return global_epoch
    except Exception as e:
        bt.logging.warning(f"Error in get_global_epoch: {str(e)}")
        return 0


def get_local_epoch(self, repo_id: str = None):
    if repo_id is None:
        repo_id = self.config.neuron.local_model_name

    try:
        refs = list_repo_refs(repo_id, repo_type="model")
        local_epoch = (
            max(
                [
                    int(tag.name.split(".")[1])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                    )
                ]
            )
            if refs.tags
            else None
        )
        return local_epoch
    except Exception as e:
        bt.logging.warning(f"Error in get_local_epoch: {str(e)}")
        return None


def get_local_inner_step(self, repo_id: str = None):
    if repo_id is None:
        repo_id = self.config.neuron.local_model_name

    try:
        refs = list_repo_refs(repo_id, repo_type="model")
        local_steps = (
            max(
                [
                    int(tag.name.split(".")[2])
                    for tag in refs.tags
                    if (
                        (len(tag.name.split(".")) == 3)
                        and (tag.name.split(".")[0] == __run__)
                        and (tag.name.split(".")[1] == str(self.local_progress.epoch))
                    )
                ]
            )
            if refs.tags
            else 0
        )
        return local_steps
    except Exception as e:
        bt.logging.warning(f"Error in get_local_inner_step: {str(e)}")
        return 0


def update_global_tracker_state(self):
    try:
        runs = wandb.Api().runs(
            f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project}"
        )
        global_epoch = get_global_epoch(self)
        global_progress = 0

        for run in tqdm(runs):
            if (
                ("validator" in run.name)  # Filter our any miner runs
                and (run.state == "running")  # Filter out any idle wandb runs
                and (
                    f"UID{self.uid}" not in run.name.split("_")
                )  # Filter out any wandb data from the current neuron's UID
            ):
                history = run.history()
                if (
                    ("local_samples_accumulated" in history.columns)
                    and ("global_samples_accumulated" in history.columns)
                    and (
                        not history.loc[
                            pd.isna(history.loc[:, "local_epoch"]) == False,
                            "local_epoch",
                        ].empty
                    )
                    and (sum(history.loc[:, "local_epoch"] == global_epoch) > 0)
                ):
                    filtered_history = history.loc[
                        (history.loc[:, "local_epoch"] == global_epoch), :
                    ]
                    filtered_history = filtered_history.loc[
                        (
                            pd.isna(
                                filtered_history.loc[:, "local_samples_accumulated"]
                            )
                            == False
                        ),
                        :,
                    ]
                    bt.logging.info(run.name.split("_"))
                    bt.logging.info(
                        max(filtered_history.loc[:, "local_samples_accumulated"])
                    )
                    global_progress += max(
                        filtered_history.loc[:, "local_samples_accumulated"]
                    )

            else:
                continue

        # Update global epoch
        self.global_progress.epoch = global_epoch if global_epoch is not None else 0

        # Add local samples
        if self.global_progress.epoch == self.local_progress.epoch:
            global_progress += self.local_progress.samples_accumulated

        # Update global progress
        self.global_progress.samples_accumulated = global_progress

        # Log new porgress
        bt.logging.info(
            f"Local samples:  {self.local_progress.samples_accumulated} | Local epoch:  {self.local_progress.epoch}"
        )
        bt.logging.info(
            f"Global samples: {self.global_progress.samples_accumulated} | Global epoch: {self.global_progress.epoch}"
        )

    except Exception as e:
        bt.logging.info(f"Failed to update global tracker state due to error {e}")
