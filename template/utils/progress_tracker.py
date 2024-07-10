import wandb
from pydantic import BaseModel, StrictBool, StrictFloat, confloat, conint
from dataclasses import dataclass
import pandas as pd
import bittensor as bt


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
        global_progress = 0
        global_epoch = 0

        for run in runs:
            if (
                ("validator" in run.name)
                and (run.state == "running")
                and (f"UID{self.uid}" not in run.name.split("_"))
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
                ):
                    max_epoch = max(
                        history.loc[
                            pd.isna(history.loc[:, "local_epoch"]) == False,
                            "local_epoch",
                        ]
                    )
                    filtered_history = history.loc[
                        (history.loc[:, "local_epoch"] == max_epoch), :
                    ]
                    filtered_history = filtered_history.loc[
                        (pd.isna(history.loc[:, "local_samples_accumulated"]) == False),
                        :,
                    ]
                    if max_epoch > global_epoch:
                        global_epoch = max(global_epoch, max_epoch)
                        global_progress = 0
                    elif max_epoch < global_epoch:
                        continue

                    global_progress += max(
                        filtered_history.loc[:, "local_samples_accumulated"]
                    )

            else:
                continue

        # Add local samples
        global_progress += self.local_progress.samples_accumulated
        if self.__class__.__name__.lower() == "validator":
            global_epoch = max(global_epoch, self.local_progress.epoch)

        self.global_progress.samples_accumulated = global_progress
        self.global_progress.epoch = global_epoch
        bt.logging.info(
            f"Local samples:  {self.local_progress.samples_accumulated} | Local epoch:  {self.local_progress.epoch}"
        )
        bt.logging.info(
            f"Global samples: {self.global_progress.samples_accumulated} | Global epoch: {self.global_progress.epoch}"
        )

    except Exception as e:
        bt.logging.info(f"Failed to update global tracker state due to error {e}")
