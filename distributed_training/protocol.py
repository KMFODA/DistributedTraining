# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Karim Foda

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

from typing import Any, List, Optional

import bittensor as bt
import numpy as np
import pydantic


class IsAlive(bt.Synapse):
    answer: Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
        "This attribute is mutable and can be updated.",
    )
    epoch: Optional[int] = None


class Train(bt.Synapse):
    """
    A simple Train protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    """

    # List of indices trained on
    dataset_seed: list = None

    # Gradient Value of a randomly chosen index
    gradients: float = None

    # Gradient Index to be evaluated
    gradient_test_index: int = None

    # Model Name
    model_name: str = "distributed/gpt2-250m"

    # Model Loss
    loss: float = 0.0


class AllReduce(bt.Synapse):
    answer: Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
        "This attribute is mutable and can be updated.",
    )
    # Learning Rate
    learning_rate: float = None
