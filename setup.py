# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

import codecs
import os
import glob


import re
from io import open
from os import path
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py


def custom_proto_compile(output_path):
    print("Compiling custom proto files...")
    import grpc_tools.protoc
    
    proto_path = os.path.join(os.path.dirname(__file__), "template", "custom_proto")
    proto_files = glob.glob(os.path.join(proto_path, "*.proto"))
    
    if not proto_files:
        print("No .proto files found in", proto_path)
        return
    
    cli_args = [
        "grpc_tools.protoc",
        f"--proto_path={proto_path}",
        f"--python_out={output_path}",
        f"--grpc_python_out={output_path}",
    ] + proto_files
    
    print(f"Custom proto files found: {proto_files}")
    
    code = grpc_tools.protoc.main(cli_args)
    if code:
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")
    
    # Make pb2 imports in generated scripts relative
    for script in glob.iglob(os.path.join(output_path, "*_pb2*.py")):
        with open(script, "r+") as file:
            code = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "from . \\1", code))
            file.truncate()
    
    print("Custom proto compilation completed.")


                
def read_requirements(path):
    with open(path, "r") as f:
        requirements = f.read().splitlines()
        processed_requirements = []

        for req in requirements:
            # For git or other VCS links
            if req.startswith("git+") or "@" in req:
                pkg_name = re.search(r"(#egg=)([\w\-_]+)", req)
                if pkg_name:
                    processed_requirements.append(pkg_name.group(2))
                else:
                    # You may decide to raise an exception here,
                    # if you want to ensure every VCS link has an #egg=<package_name> at the end
                    continue
            else:
                processed_requirements.append(req)
        return processed_requirements


requirements = read_requirements("requirements.txt")
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# loading version from setup.py
with codecs.open(
    os.path.join(here, "template/__init__.py"), encoding="utf-8"
) as init_file:
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M
    )
    version_string = version_match.group(1)

class CustomInstallCommand(install):
    def run(self):
        print("Running CustomInstallCommand")
        self.run_command('egg_info')
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])
        subprocess.check_call(['pip', 'install', 'grpcio-tools'])
        print("About to call custom_proto_compile")
        custom_proto_compile("template/custom_proto")
        print("Finished custom_proto_compile")
        install.run(self)
        
class CustomBuildCommand(build_py):
    def run(self):
        print("Running CustomBuildCommand")
        custom_proto_compile("template/custom_proto")
        build_py.run(self)

setup(
    name="bittensor_subnet_template",  # TODO(developer): Change this value to your module subnet name.
    version=version_string,
    description="bittensor_subnet_template",  # TODO(developer): Change this value to your module subnet description.
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opentensor/bittensor-subnet-template",  # TODO(developer): Change this url to your module subnet github url.
    author="bittensor.com",  # TODO(developer): Change this value to your module subnet author name.
    packages=find_packages(),
    include_package_data=True,
    author_email="",  # TODO(developer): Change this value to your module subnet author email.
    license="MIT",
    python_requires=">=3.8",
    install_requires=requirements + ['grpcio-tools'],  # Add grpcio-tools to requirements
    cmdclass={
        'install': CustomInstallCommand,
        'build_py': CustomBuildCommand,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
