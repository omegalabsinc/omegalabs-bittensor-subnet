FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to add repositories
RUN apt-get -y update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && apt-get install -y \
    python3.10 python3.10-distutils python3.10-venv python3.10-dev \
    git libsndfile1 build-essential ffmpeg libpq-dev \
    pkg-config libmysqlclient-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update the symbolic link for python to point to python3.10
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app/

# Install python requirements
COPY ./requirements.txt ./requirements.txt
COPY ./requirements_api.txt ./requirements_api.txt

RUN python -m ensurepip && python -m pip install --upgrade pip setuptools wheel uv
RUN python -m uv pip install -r requirements_api.txt --prerelease=allow --no-cache-dir

COPY . .
RUN python -m pip install -e . --no-cache-dir

ENTRYPOINT bash
# Run the miner launch command like this inside the container:
# python -m miner.miner --netuid 1 --wallet.name miner --wallet.hotkey default --logging.debug
