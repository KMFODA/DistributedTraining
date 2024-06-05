<<<<<<< HEAD
import copy
=======
from template.validator.validator_core import (
    DatasetStateSingelton,
    ModelSingleton,
    upload_checkpoint,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
>>>>>>> origin/DTAllReduce
from functools import partial

import hivemind
import torch
from hivemind.optim.state_averager import TrainingStateAverager
from transformers import AutoModelForCausalLM, AutoTokenizer

from template.validator.validator_core import (DatasetStateSingelton,
                                               ModelSingleton,
                                               upload_checkpoint)

model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Init DHT
version = "4"
address = "38.147.83.16"
dht_port = 10610
run_id = "s25_run_v1"
announce_maddrs = [f"/ip{version}/{address}/tcp/{dht_port}"]
lr = 0.00001
dht = hivemind.DHT(
    host_maddrs=[f"/ip4/0.0.0.0/tcp/{dht_port}", f"/ip4/0.0.0.0/udp/{dht_port}/quic"],
    initial_peers=[
        "/ip4/161.97.156.125/tcp/8001/p2p/12D3KooWRe4RHd5NxRhfn5rMuCk6hA9UBNnK8V3Xy3ejcFApGkRx",
        "/ip4/38.79.71.1/tcp/10263/p2p/12D3KooWMfiDM67PW6GerfahQPPdc4Bt3tkiHo8vZXieZL5mVTsc",
    ],
    announce_maddrs=announce_maddrs,
    start=True,
    # dameon=True
)

state_averager = TrainingStateAverager(
    dht=dht,
    optimizer=partial(torch.optim.AdamW, lr=lr),
    scheduler=partial(
        torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)
    ),
    params=model.parameters(),
    allow_state_sharing=False,
    start=True,
    prefix=f"{run_id}_state_averager",
    # state_compression=hivemind.Float16Compression(),
    # bandwidth=optimizer_args.bandwidth,
    # client_mode=optimizer_args.client_mode,
    # **asdict(averager_args),
)

model2 = copy.deepcopy(model)
state_averager.load_state_from_peers()
assert model != model2
