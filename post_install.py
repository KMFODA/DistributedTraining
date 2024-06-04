import os

import pkg_resources


def remove_nest_asyncio_import():
    try:
        # Locate the installation directory of bittensor
        distribution = pkg_resources.get_distribution("bittensor")
        bittensor_path = distribution.location

        # Path to bittensor __init__.py
        init_path = os.path.join(bittensor_path, "bittensor", "__init__.py")

        if not os.path.exists(init_path):
            print(
                "bittensor __init__.py not found. Ensure bittensor is correctly installed."
            )
            return

        # Read the file
        with open(init_path, "r") as file:
            lines = file.readlines()

        # Remove lines that import nest_asyncio
        with open(init_path, "w") as file:
            for line in lines:
                if "nest_asyncio" not in line:
                    file.write(line)

        # Uninstall nest_asyncio
        os.system("pip uninstall -y nest_asyncio")

    except pkg_resources.DistributionNotFound:
        print("bittensor package is not installed. Please install it first.")


if __name__ == "__main__":
    remove_nest_asyncio_import()
