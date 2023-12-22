import torch
import asyncio
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from hivemind.utils.timed_storage import get_dht_time
# Otherwise just use time.time, as per here: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/utils/timed_storage.py#L12

import random 
import json
import time
from time import sleep
import bittensor as bt

class DatasetStateSingelton:
    '''
    This class shares the amount of indicies in an existing dataset for distribution among miners.
    Indices that have been used during an epoch are removed. 
    (There should be a mechanism added on failure to allow for repeating)
    If the indices run out then a new epoch is calculated and the number of available indices is reset to full.
    The following 
    '''
    _instance = None

    def __new__(cls, dht_state, dataset_indices, run_id, default_expiration_time = 600, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatasetStateSingelton, cls).__new__(cls, *args, **kwargs)
            cls._instance.dht_state = dht_state
            assert run_id, "run_id isn't specified when run_id can't be empty/zero/null" 
            cls._instance.run_id = run_id
            cls._instance.dataset_indices_original = dataset_indices
            cls._instance.dataset_indices = None
            cls._instance.loss = None
            cls._instance.default_expiration_time = default_expiration_time
        return cls._instance
    
    @classmethod
    async def initialize_async(cls):
        if cls._instance.dataset_indices is None:
            cls._instance.dataset_indices = await cls._instance.get_dht("dataset_indices")
        if cls._instance.loss is None:
            cls._instance.loss = await cls._instance.get_dht("loss")

    @staticmethod
    def _serialize_to_string(data_str):
        """
        Serializes the dataset indices to a string.
        """
        # Assuming dataset_indices is a list or a similar serializable structure
        return json.dumps(data_str)

    @staticmethod
    def _deserialize_from_string(data_str):
        """
        Deserializes the string back to dataset indices.
        """
        # Assuming the data_str is in JSON format
        return json.loads(data_str.value)
    
    async def get_dht(self, name):
        await asyncio.sleep(2)
        loop = asyncio.get_running_loop()
        stored_data = await loop.run_in_executor(None, self.dht_state.get, f"{self.run_id}_{name}")
        return self._deserialize_from_string(stored_data) if stored_data else None

    async def set_dht(self, name, value):
        await asyncio.sleep(2)
        serialized_value = self._serialize_to_string(value)
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, self.dht_state.store, f"{self.run_id}_{name}", serialized_value, get_dht_time() + self.default_expiration_time)
        return status
    
    @classmethod
    async def get_dataset_indices(cls, groups_count, items_per_group):
        """
        Selects m groups of n consecutive indices from a list in indices_dict[key].
        Each group of n indices is removed from the original list to ensure no replacement.

        :param indices_dict: Dictionary containing lists of indices.
        :param key: Key in the dictionary to access the list of indices.
        :param groups_count: Number of groups to select.
        :param items_per_group: Number of consecutive indices in each group.
        :return: List of selected groups, each group is a list of n indices.
        """
        indices = await cls._instance.get_dht("dataset_indices")
        no_value_flag = False
        try:
            no_value_flag = len(indices) < (groups_count * items_per_group)
        except:
            no_value_flag = True

        if no_value_flag:
            bt.logging.info("Ran out of dataset indices. Reloading")
            # Not enough indices to select the required number of groups"
            # Restore all the values. Then resample.
            await cls._instance.set_dht("dataset_indices", cls._instance.dataset_indices_original)
            try:
                cls.epoch += 1
            except:
                cls.epoch = 1

            return await cls.get_dataset_indices(groups_count, items_per_group)

        selected_groups = []
        for _ in range(groups_count):
            start = random.choice(range(len(indices) - items_per_group + 1))
            group = indices[start:start + items_per_group]
            selected_groups.append(group)

            # Remove selected indices
            indices = indices[:start] + indices[start + items_per_group:]

        # Update the original list in the dictionary
        bt.logging.info("Removing selected indices from the DHT")
        await cls._instance.set_dht("dataset_indices",indices)

        return selected_groups
        
class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, model_name, device):
        if cls._instance is None:
            cls._instance = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            
        return cls._instance


def upload_checkpoint(commit_message, state_averager, model, repo_path,repo_url):
    bt.logging.info("Saving optimizer")
    torch.save(state_averager.optimizer.state_dict(), f"{repo_path}/optimizer_state.pt")
    timestamp_at_upload = time.time()
    bt.logging.info("Started uploading to Model Hub")
    model.push_to_hub(
        repo_name=repo_path,
        repo_url=repo_url,
        commit_message=commit_message,
    )
    bt.logging.info("Finished uploading to Model Hub")