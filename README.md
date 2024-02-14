
# ⚠️ Deprecated: This project's base repository has been updated. Please use the new [distributed training repository](https://github.com/bit-current/DistributedTraining/) instead.

<div align="center">

# **Distributed Training Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>

---

![Subnet25](assets/Subnet25.jpg)

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.

1. Install this repository
```bash
git clone https://github.com/KMFODA/DistributedTraining
cd DistributedTraining
pip install -e . && python post_install.py
```

2. Log in to wandb:
```bash
wandb login <your_wandb_api_key>
```

3. Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.

**On Linux**:
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
``` 
**On Mac OS**
```bash
brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
```
---

Once you have installed this repo you can run the miner and validator with **auto updates enabled** using the following commands.
```bash
# To run the miner
chmod +x run_miner.sh
pm2 start run_miner.sh --name distributed_training_miner_auto_update --
    --netuid <your netuid>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url>  # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>

# To run the validator
chmod +x run_validator.sh
pm2 start run_validator.sh --name distributed_training_auto_update --
    --netuid <your netuid> # Must be attained by following the instructions in the docs/running_on_*.md files
    --subtensor.chain_endpoint <your chain url> # Must be attained by following the instructions in the docs/running_on_*.md files
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port <an open port to serve the bt axon on>
    --dht.port <another open port to serve the dht axon on>
    --dht.announce_ip <your device ip address>
```

</div>

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
