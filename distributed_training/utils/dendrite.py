import aiohttp

from bittensor.core.axon import Axon
from bittensor.core.dendrite import DendriteMixin
from bittensor.utils.registration import torch, use_torch
from typing import Optional, Union, List
from bittensor_wallet import Keypair, Wallet


class DTDendriteMixin(DendriteMixin):
    def __init__(self, wallet, connection_limit=100):
        self._connection_limit = connection_limit
        super().__init__(wallet)

    @property
    async def session(self) -> aiohttp.ClientSession:
        """
        An asynchronous property that provides access to the internal `aiohttp <https://github.com/aio-libs/aiohttp>`_
        client session.

        This property ensures the management of HTTP connections in an efficient way. It lazily
        initializes the `aiohttp.ClientSession <https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientSession>`_
        on its first use. The session is then reused for subsequent HTTP requests, offering performance benefits by
        reusing underlying connections.

        This is used internally by the dendrite when querying axons, and should not be used directly
        unless absolutely necessary for your application.

        Returns:
            aiohttp.ClientSession: The active `aiohttp <https://github.com/aio-libs/aiohttp>`_ client session instance.
            If no session exists, a new one is created and returned. This session is used for asynchronous HTTP requests
            within the dendrite, adhering to the async nature of the network interactions in the Bittensor framework.

        Example usage::

            import bittensor                                # Import bittensor
            wallet = bittensor.Wallet( ... )                # Initialize a wallet
            dendrite = bittensor.Dendrite(wallet=wallet)   # Initialize a dendrite instance with the wallet

            async with (await dendrite.session).post(       # Use the session to make an HTTP POST request
                url,                                        # URL to send the request to
                headers={...},                              # Headers dict to be sent with the request
                json={...},                                 # JSON body data to be sent with the request
                timeout=10,                                 # Timeout duration in seconds
            ) as response:
                json_response = await response.json()       # Extract the JSON response from the server

        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=self._connection_limit)
            )
        return self._session


# For back-compatibility with torch
BaseModel: Union["torch.nn.Module", object] = torch.nn.Module if use_torch() else object


class DTDendrite(DTDendriteMixin, BaseModel):  # type: ignore
    def __init__(
        self,
        wallet: Optional[Union["Wallet", "Keypair"]] = None,
        connection_limit: int = 100,
    ):
        if use_torch():
            torch.nn.Module.__init__(self)
        DTDendriteMixin.__init__(self, wallet, connection_limit)


if not use_torch():

    async def call(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)

    DTDendrite.__call__ = call


async def async_dendrite_forward(
    wallet: "Wallet" = None,
    axons: List["Axon"] = [],
    synapse=None,
    connection_limit: int = 100,
    timeout: float = 30.0,
):
    async with DTDendrite(wallet, connection_limit=connection_limit) as d:
        await d(axons, synapse=synapse, timeout=timeout)
