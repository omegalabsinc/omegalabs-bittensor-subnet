FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to add repositories
RUN apt-get -y update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && apt-get install -y \
    python3.10 python3.10-distutils python3.10-venv python3.10-dev \
    git libsndfile1 build-essential ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update the symbolic link for python to point to python3.10
RUN rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Install python requirements
COPY gpu_requirements.txt requirements.txt
RUN python -m ensurepip && python -m pip install --upgrade pip setuptools wheel
# install imagebind first so that pytorch 1.13 (requested by imagebind) gets overwritten by pytorch2 in requirements.txt
RUN python -m pip install --ignore-installed --no-cache-dir git+https://github.com/facebookresearch/ImageBind.git@c6a47d6dc2b53eced51d398c181d57049ca59286
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m pip install -e .

WORKDIR /validator-api

ENTRYPOINT python -m validator_api.app
