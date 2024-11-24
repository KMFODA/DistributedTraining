import copy

import bittensor as bt
from bitsandbytes.optim import LAMB8bit
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM

from distributed_training.utils.progress_tracker import get_global_epoch


def load_state_from_peer(self, epoch=None, keep_recent=5):
    state_loaded = False
    if epoch == None:
        self.global_progress.epoch = get_global_epoch(self)
        epoch = self.global_progress.epoch

    bt.logging.info("Model Weights Before Loading State")
    current_model_weights_sample = copy.copy(
        [layer for layer in self.model.parameters()][-2][-10:].tolist()
    )
    bt.logging.info(current_model_weights_sample)

    bt.logging.info(f"Old Model Tag: {self.local_progress.epoch}")
    # if (self.global_progress.epoch is not None) and (tag_name >= epoch):
    if self.global_progress.epoch is not None:
        bt.logging.info(
            f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
        )
        attempt = 0
        while True:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.neuron.model_name,
                    revision=str(self.global_progress.epoch),
                    trust_remote_code=True,
                )
                self.model.to(self.device)
                break
            except:
                attempt += 1
                bt.logging.warning(f"Failed to fetch data, retrying. Attempt {attempt}")
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.opt = LAMB8bit(
            optim_groups, lr=self.learning_rate_maximum, betas=(0.9, 0.95), eps=1e-8
        )
        self.grad_averager.parameters = tuple(self.model.parameters())
        # Reset gradient buffers
        self.grad_averager.reset_accumulated_grads_()
        state_loaded = True

        bt.logging.info("Model Weights After Loading State")
        new_model_weights_sample = copy.copy(
            [layer for layer in self.model.parameters()][-2][-10:].tolist()
        )
        bt.logging.info(new_model_weights_sample)

        self.local_progress.epoch = self.global_progress.epoch
        self.local_progress.samples_accumulated = 0
        bt.logging.info(f"New Model Tag: {self.global_progress.epoch}")

        # Delete one model from the chace to maintain disk space
        current_revision = self.model.config._commit_hash
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == self.config.neuron.model_name:
                    revisions = sorted(
                        repo.revisions, key=lambda r: r.last_modified, reverse=True
                    )
                    current_index = next(
                        (
                            i
                            for i, r in enumerate(revisions)
                            if r.commit_hash == current_revision
                        ),
                        None,
                    )
                    if current_index is not None:
                        for revision in revisions[
                            max(current_index + 1, keep_recent) :
                        ]:
                            cache_info.delete_revisions(revision.commit_hash).execute()
                    break
        except:
            bt.logging.warning(
                "Failed to delete previous model version from cache. This might lead to 100% disk space utlisation in the future."
            )

    else:
        bt.logging.info(f"Model With Tag: {epoch} Does Not Exist")

    return state_loaded
