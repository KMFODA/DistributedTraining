from enum import Enum

import bittensor as bt
import torch
import transformers
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer import Trainer
from template.utils.hivemind import load_state_from_peer


class AveragingStage(Enum):
    AWAITING_TRIGGER = 1  # waiting for user to set the trigger that allows running allreduce
    RUNNING_ALLREDUCE = 2  # exchanging tensors with groupmates
    FINISHED = 3  # either done or failed with exception

class CustomValidationCallback(TrainerCallback):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        trainer: Trainer,
        #local_public_key: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.trainer = trainer
        #self.local_public_key = local_public_key
        #self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1
        self.samples = 0
        self.steps = 0
        self.loss = 0
        self.total_samples_processed = 0
        #self.backup_every_steps = backup_every_steps
        self.latest_backup = self.backup_state()
        
    def on_train_begin(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        bt.logging.info("Loading state from peers")
        load_state_from_peer()
    
    def on_step_end(
        self, args: TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs
    ):
        control.should_log = True
        if not self.params_are_finite():
            self.restore_from_backup(self.latest_backup)
            return control
        
        if self.stage == AveragingStage.RUNNING_ALLREDUCE:
            
            bt.logging.info("Received All Reduce Call")
        
            custom_group = self.group

            # Perform AllReduce step with queried miners to get averaged gradients
            bt.logging.info("Performing Gradient Averaging")
            self.gradient_averaging_step = self.grad_averager.step(group=custom_group, wait=False) # TODO Should we await this?
            
            # Reset flag
            self.stage = AveragingStage.FINISHED
        
        return control
    
    @torch.no_grad()
    def params_are_finite(self):
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param)):
                return False
        return True

class CustomTrainer(Trainer):
    def __init__(self, *args, grad_accumulator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_accumulator = grad_accumulator
    
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.cuda.amp.autocast(self.use_amp):
            outputs = model(**inputs)
            loss = outputs.loss / self.args.gradient_accumulation_steps

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if self.grad_accumulator:
            for param in model.parameters():
                if param.grad is not None:
                    # Accumulate Gradients
                    self.grad_averager.accumulate_grads_(batch_size=len(inputs))
        
        return loss.detach()