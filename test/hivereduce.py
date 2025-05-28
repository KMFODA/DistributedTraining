import argparse
import torch
import logging
import os
import copy
import time
import json
import math
import pickle
from contextlib import nullcontext
from functools import partial
import numpy as np

import hivemind
from hivemind.utils.logging import use_hivemind_log_handler # Optional: for detailed hivemind logs

from distributed_training.averaging.averagers import DTStateAverager, DTGradAverager

from model import GPTConfig, GPT
from profiler import Profiler, ProfilerCollection # Optional, can be removed if not profiling

def update_main_param_after_outer_step(model_to_update, averager_with_state):
    """
    Pulls the new averaged state from the state_averager to the local model (raw_model).
    """
    local_model_parameters = list(model_to_update.parameters())
    averaged_parameters = averager_with_state.main_parameters
    
    with torch.no_grad():
        for local_param, avg_param in zip(local_model_parameters, averaged_parameters):
            # Copy from averaged_parameters (source) to local_model_parameters (destination)
            local_param.data.copy_(avg_param.data, non_blocking=True)


# --- Helper functions from sync_diloco.py ---
def get_batch(split, config, device_type, device):
    # data_dir = os.path.join('data', config["dataset"])
    data_dir = r"/workspace/pccl/python/examples/nanogpt_diloco/data/openwebtext/"
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    data_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, ctx, config, get_batch_fn, device_type, device):
    eval_iters = config["eval_iters"]
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch_fn(split, config, device_type, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it, config):
    if not config["decay_lr"]:
        return config["learning_rate"]
    learning_rate = config["learning_rate"]
    warmup_iters = config["warmup_iters"] // config["inner_steps"] 
    lr_decay_iters = config["lr_decay_iters"] // config["inner_steps"]
    min_lr = config["min_lr"]
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    decay_ratio = max(0, min(1, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

EXPORT_PROFILER_VIDEO = False # Set to True to enable video profiling

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="default_config_hivemind.json", help="Path to JSON config.")
    parser.add_argument("--prefix", type=str, required=True, help="DHT prefix for Hivemind.")
    parser.add_argument("--initial_peers", type=str, nargs="*", help="Optional initial DHT peers.")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if (config["dtype"] == "float16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config["dtype"] = "bfloat16"

    # Hivemind DHT Setup
    version = "4"
    address = os.environ.get('PUBLIC_IPADDR')
    port = os.environ.get('VAST_TCP_PORT_70000')

    announce_maddrs = [f"/ip4/{address}/tcp/{port}"]
    print(f"DHT Announcing Multiaddresses: {announce_maddrs}")

    dht_params = {
        "host_maddrs": [
            f"/ip4/0.0.0.0/tcp/{port}",
            f"/ip4/0.0.0.0/udp/{port}/quic",
        ],
        "announce_maddrs": announce_maddrs,
        "start": True,
    }
    
    # Conditionally add initial_peers if provided
    if args.initial_peers:
        dht_params["initial_peers"] = [str(peer) for peer in args.initial_peers]

    print("Initializing DHT...")
    dht = hivemind.DHT(**dht_params)
    print(f"DHT initialized. Visible addresses: {dht.get_visible_maddrs()}")
    
    if not args.initial_peers: 
        time.sleep(15)
        
    # General Setup (from sync_diloco.py)
    is_dht_master_peer = config.get("is_dht_master_peer", True) # Peer responsible for saving checkpoints etc.
    device = config["device"]

    if is_dht_master_peer:
        os.makedirs(config["out_dir"], exist_ok=True)
    
    # Use a unique seed per peer for data loading, but model init should be same if "scratch"
    torch.manual_seed(config.get("seed", 1337) + os.getpid()) # Add pid for some variance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config["dtype"]]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Model Setup (from sync_diloco.py)
    data_dir = os.path.join('data', config["dataset"])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f: meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
    model_args = dict(n_layer=config["n_layer"], n_head=config["n_head"], n_embd=config["n_embd"],
                      block_size=config["block_size"], bias=config["bias"], vocab_size=None, dropout=config["dropout"])
    
    init_from = config["init_from"]
    checkpoint = None
    start_iter_num = 0 # This is the outer step number

    if init_from == 'scratch':
        print("Initializing a new model from scratch.")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        # ... (load model state_dict as in sync_diloco.py) ...
        model_args_ckpt = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = model_args_ckpt[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        start_iter_num = checkpoint.get('iter_num', 0) # outer_iter_num
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=config["dropout"]))
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    else: raise ValueError(f"Unknown init_from: {init_from}")
    
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args['block_size'] = config["block_size"]
    model.to(device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizers
    inner_optimizer = model.configure_optimizers(config["weight_decay"], config["learning_rate"],
                                               (config["beta1"], config["beta2"]), device_type)
    if init_from == 'resume' and checkpoint and 'optimizer' in checkpoint: # Inner optimizer state
        inner_optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint # Free memory
        
    outer_optimizer_partial = partial(torch.optim.SGD, lr=config["outer_learning_rate"], momentum=0.9, nesterov=True)

    if config["compile"]:
        print("Compiling the model...")
        model = torch.compile(model)
    
    raw_model = model # No DDP wrapper

    # Hivemind Averagers
    print("Setting up Hivemind StateAverager...")
    state_averager = DTStateAverager(
        dht=dht, 
        prefix=f"{args.prefix}", 
        optimizer=outer_optimizer_partial,
        params=raw_model.parameters(), 
        initialize_optimizer=True, 
        offload_optimizer=True, # As per old script
        start=True, 
        request_timeout=config.get("hivemind_timeout_request", 10.0),
        next_chunk_timeout=config.get("hivemind_timeout_chunk", 45.0)
    )
    print("Setting up Hivemind GradAverager...")
    grad_averager = DTGradAverager(
        list(raw_model.parameters()), 
        offloaded_optimizer=state_averager.optimizer, # Crucial link
        dht=dht, 
        prefix=f"{args.prefix}",
        start=True,
        #compression=hivemind.Uniform8BitQuantization(), # Example compression
        target_group_size=config.get("hivemind_target_group_size", 2),
        min_group_size=config.get("hivemind_min_group_size", 2),
    )
    
    # Hivemind Progress Tracker
    # Samples per outer step = inner_steps * batch_size * gradient_accumulation_steps
    samples_per_outer_step = config["inner_steps"] * config["batch_size"] * config["gradient_accumulation_steps"]
    tracker = hivemind.optim.progress_tracker.ProgressTracker(
        dht=dht, 
        prefix=f"{args.prefix}_tracker", 
        target_batch_size=samples_per_outer_step, 
        start=True
    )

    # Training Loop
    profiler_collection = ProfilerCollection() # Optional
    local_samples_processed_in_outer_step = 0
    
    # outer_iter_num is equivalent to 'iter_num' in sync_diloco for the outer loop
    for outer_iter_num in range(start_iter_num, config["max_iters"] // config["inner_steps"]):
        t0 = time.time()
        profiler = Profiler() # Optional

        # 1. Update learning rate for inner optimizer
        lr = get_lr(outer_iter_num, config)
        for param_group in inner_optimizer.param_groups:
            param_group['lr'] = lr

        # 2. Perform local (inner) steps
        raw_model.train()
        total_inner_loss = 0.0
        with profiler.session("inner_steps_loop"): # Optional profiling
            for _ in range(config["inner_steps"]):
                inner_optimizer.zero_grad() # Zero grads for each inner step's accumulation
                for micro_step in range(config["gradient_accumulation_steps"]):
                    x, y = get_batch('train', config, device_type, device)
                    with ctx:
                        logits, loss = raw_model(x, y)
                        loss = loss / config["gradient_accumulation_steps"]
                    loss.backward() # Accumulates gradients in raw_model.parameters().grad
                    total_inner_loss += loss.item() * config["gradient_accumulation_steps"]
                
                # Clip grads for the accumulated micro-steps
                if config["grad_clip"] != 0.0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config["grad_clip"])
                
                inner_optimizer.step() # Apply accumulated grads for this inner step
                
                # Report progress for samples processed in this inner optimizer step
                local_samples_processed_in_outer_step += config["batch_size"] * config["gradient_accumulation_steps"]
        
        avg_inner_loss = total_inner_loss / (config["inner_steps"] * config["gradient_accumulation_steps"])
        print(f"Outer iter {outer_iter_num}, Avg Inner Loss: {avg_inner_loss:.4f}")

        # 3. Outer Step with Hivemind
        with profiler.session("hivemind_outer_step"): # Optional profiling
            # Gradients from the *last inner step* are now in raw_model.parameters().grad
            # These are the gradients we want to average for the "outer" update.
            print(f"Outer iter {outer_iter_num}: Averaging gradients with GradAverager...")
            with profiler.session("grad_averager.step"):
                grad_averager.step() # Averages raw_model.parameters().grad with peers.
                                     # Crucially, it then applies these to state_averager.optimizer's parameters' gradients.
                grad_averager.notify_used_averaged_gradients()

            print(f"Outer iter {outer_iter_num}: Applying averaged gradients with StateAverager's optimizer...")
            with profiler.session("state_averager.optimizer_step"):
                # This step uses the offloaded SGD optimizer within state_averager
                # to take a step on its internal (averaged) parameters, using the
                # gradients that grad_averager just prepared.
                # It also implicitly averages the resulting parameters if configured.
                state_averager.step(increment_epoch=False, # We manage epoch with tracker
                                    optimizer_step=True)   # Perform the optimizer step

            print(f"Outer iter {outer_iter_num}: Updating local model with averaged state...")
            with profiler.session("state_averager.update_local_model"):
                update_main_param_after_outer_step(raw_model, state_averager) # Pulls new averaged state to raw_model

        # # Report progress to tracker AFTER the outer step is complete
        # tracker.report_local_progress(samples_processed=local_samples_processed_in_outer_step)
        # local_samples_processed_in_outer_step = 0 # Reset for next outer step

        # if tracker.global_progress.samples_accumulated >= samples_per_outer_step :
        #     print(f"Global target for outer step {outer_iter_num} complete. Advancing global epoch.")
        #     tracker.update_epoch(tracker.global_epoch + 1)
        #     # local_samples_processed_in_outer_step is already reset

        # # Logging and Checkpointing
        # dt = time.time() - t0
        # if outer_iter_num % config["log_interval"] == 0 :
        #     if is_dht_master_peer : # Only one peer should do this
        #         # MFU calculation (from sync_diloco.py)
        #         # ... (copy MFU logic if needed) ...
        #         val_loss = estimate_loss(raw_model, ctx, config, get_batch, device_type, device)['val']
        #         print(f"Outer iter {outer_iter_num}: Val Loss {val_loss:.4f}, Time {dt*1000:.2f}ms")
        #         if config["wandb_log"]: wandb.log({"iter": outer_iter_num, "val_loss": val_loss, "lr": lr, "time_ms": dt*1000})

        #         if val_loss < config.get("best_val_loss", 1e9): # Store best_val_loss in config or track locally
        #             config["best_val_loss"] = val_loss
        #             if outer_iter_num > 0: # Don't save at step 0
        #                 checkpoint = {'model': raw_model.state_dict(),
        #                               'optimizer': inner_optimizer.state_dict(), # Save inner optimizer
        #                               'model_args': model_args, 'iter_num': outer_iter_num, 'config': config}
        #                 print(f"Saving checkpoint to {config['out_dir']}")
        #                 torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
        #     else: # Non-master peers can also log their local loss if desired
        #         print(f"Outer iter {outer_iter_num} (peer): Avg Inner Loss {avg_inner_loss:.4f}, Time {dt*1000:.2f}ms")


        # if config["eval_only"] and outer_iter_num == start_iter_num: break
        # if outer_iter_num >= config["max_iters"] // config["inner_steps"]: break
        
        profiler.print_report() # Optional
        if EXPORT_PROFILER_VIDEO: # Optional
            profiler_collection.add_profiler(profiler, f"OuterStep {outer_iter_num}")
            if outer_iter_num % 100 == 0 and is_dht_master_peer:
                profiler_collection.render_as_video(f"timeline_diloco_hivemind_{outer_iter_num}.mp4")

    # Cleanup
    print("Shutting down Hivemind components...")
    if tracker and hasattr(tracker, 'stop'): tracker.stop()
    if grad_averager and hasattr(grad_averager, 'stop'): grad_averager.stop()
    if state_averager and hasattr(state_averager, 'stop'): state_averager.stop()
    if dht and hasattr(dht, 'shutdown'): dht.shutdown(wait=True)
    print("Training finished.")

if __name__ == "__main__":
    # Optional: Setup detailed Hivemind logging if desired
    # use_hivemind_log_handler("debug") # or "info"
    # logging.basicConfig(level=logging.INFO) # Basic logging for other parts
    main()