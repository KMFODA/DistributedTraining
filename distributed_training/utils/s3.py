import torch
import bittensor as bt
from dotenv import dotenv_values
import os
from aiobotocore.session import get_session
import botocore.config
import tempfile
from aiobotocore.session import get_session
from typing import List, Dict
import numpy as np
import hashlib
import time
import asyncio

# Load environment variables
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_config.get("AWS_SECRET_ACCESS_KEY")

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)


async def add_slice_for_window_to_buffer(
    self,
    dataset_index: int,
    model: torch.nn.Module,
    window: int,
    seed: str,
    compression: int,
    key: str = "gradient",
):
    model_state_dict = model.state_dict()
    indices = await get_indices_for_window(model, seed, compression)

    # Apply the slice to the model parameters
    for name, param in model.named_parameters():
        model_state_dict[name] = param.grad.view(-1)[
            indices[name].to(model.device)
        ].cpu()

    self.grad_buff_queue.put(
        {
            "dataset_index": dataset_index,
            "gradient_slice": model_state_dict,
            "window": window,
        }
    )


async def upload_slice_for_window(
    bucket: str,
    model: torch.nn.Module,
    window: int,
    seed: str,
    wallet: "bt.wallet",
    compression: int,
    key: str = "slice",
):
    """
    Uploads a compressed slice of a PyTorch model to an S3 bucket.

    Args:
        bucket (str): Name of the S3 bucket.
        model (torch.nn.Module): The PyTorch model to be sliceed and uploaded.
        window (int): The window identifier.
        wallet (bt.wallet): The wallet object containing the hotkey.
        compression (int): The compression factor.
    """
    filename = f"{key}-{window}-{wallet.hotkey.ss58_address}.pt"
    bt.logging.info(f"Uploading slice to S3: {filename}")

    model_state_dict = model.state_dict()
    indices = await get_indices_for_window(model, seed, compression)

    # Apply the slice to the model parameters
    for name, param in model.named_parameters():
        model_state_dict[name] = param.data.view(-1)[
            indices[name].to(model.device)
        ].cpu()

    # Create a temporary file and write the sliceed model state dictionary to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        torch.save(model_state_dict, temp_file)
        temp_file_name = temp_file.name  # Store the temporary file name

    # Upload the file to S3
    session = get_session()
    async with session.create_client(
        "s3",
        region_name="us-east-1",
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    ) as s3_client:
        try:
            with open(temp_file_name, "rb") as f:
                await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
            # Set the object ACL to public-read
            await s3_client.put_object_acl(
                Bucket=bucket, Key=filename, ACL="public-read"
            )
            bt.logging.info(f"Successfully uploaded slice to S3: {filename}")
        except Exception:
            bt.logging.infotion(f"Failed to upload slice {filename} to S3")
        finally:
            # Clean up the temporary file
            os.remove(temp_file_name)
            bt.logging.info(f"Temporary file {temp_file_name} removed")


async def get_indices_for_window(
    model: torch.nn.Module, seed: str, compression: int
) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given window and compression factor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        seed (str): The window seed identifier.
        compression (int): The compression factor.

    Returns:
        Dict[str, torch.LongTensor]: A dictionary mapping parameter names to index tensors.
    """
    bt.logging.info(
        f"Computing indices for window seed {seed} with compression {compression}"
    )
    result = {}
    # Seed the random number generator with the seed
    seed = int(hashlib.md5(str(seed).encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    for name, param in model.named_parameters():
        # Randomly select indices based on the compression factor
        num_indices = max(1, int(param.numel() // compression))
        indices = rng.choice(param.numel(), size=num_indices, replace=False)
        result[name] = torch.from_numpy(indices).long().cpu()
    return result


async def upload_gradient_buffers_to_s3(
    self, bucket: str, wallet: "bt.wallet", key: str
):
    while True:
        if self.grad_buff_queue.qsize() != 0:
            # TODO: Clear from the queue appropriately if succesfull
            # TODO: Delete from S3 after a certain period of time
            # TODO: Block if grad_all_reducing
            grad_dict = self.grad_buff_queue.get()
            window = grad_dict["window"]
            model_state_dict = grad_dict["gradient_slice"]

            filename = f"{key}-{window}-{wallet.hotkey.ss58_address}.pt"
            bt.logging.info(f"Uploading slice to S3: {filename}")

            # Create a temporary file and write the sliceed model state dictionary to it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(model_state_dict, temp_file)
                temp_file_name = temp_file.name  # Store the temporary file name

            # Upload the file to S3
            session = get_session()
            async with session.create_client(
                "s3",
                region_name="us-east-1",
                config=client_config,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            ) as s3_client:
                try:
                    with open(temp_file_name, "rb") as f:
                        await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
                    # Set the object ACL to public-read
                    await s3_client.put_object_acl(
                        Bucket=bucket, Key=filename, ACL="public-read"
                    )
                    bt.logging.info(f"Successfully uploaded slice to S3: {filename}")
                except Exception as e:
                    bt.logging.info(
                        f"Failed to upload slice {filename} to S3 with error {e}"
                    )
                finally:
                    # Clean up the temporary file
                    os.remove(temp_file_name)
                    bt.logging.info(f"Temporary file {temp_file_name} removed")
        else:
            time.sleep(5)
