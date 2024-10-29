import wandb
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint
from dataclasses import dataclass
import pandas as pd
import bittensor as bt
from huggingface_hub import list_repo_refs
from tqdm import tqdm


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


def update_global_tracker_state(self):
    try:
        runs = wandb.Api().runs(
            f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project}"
        )
        refs = list_repo_refs(self.config.neuron.model_name, repo_type="model")
        global_epoch = max([int(tag.name) for tag in refs.tags]) if refs.tags else None
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

        # Add local samples
        if self.global_progress.epoch == self.local_progress.epoch:
            global_progress += self.local_progress.samples_accumulated

        # Update global progress
        self.global_progress.samples_accumulated = global_progress
        self.global_progress.epoch = global_epoch

        # Log new porgress
        bt.logging.info(
            f"Local samples:  {self.local_progress.samples_accumulated} | Local epoch:  {self.local_progress.epoch}"
        )
        bt.logging.info(
            f"Global samples: {self.global_progress.samples_accumulated} | Global epoch: {self.global_progress.epoch}"
        )

    except Exception as e:
        bt.logging.info(f"Failed to update global tracker state due to error {e}")
