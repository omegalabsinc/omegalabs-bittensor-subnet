#!/bin/bash

MINER_SUFFIX=$1

docker build -f Dockerfile -t omega-bittensor:latest .

docker run --rm \
    --network="host" \
    --gpus all \
    --detach \
    -v $(pwd):/app \
    -v $(pwd)/cache:/root/.cache \
    -v /home/tom/.bittensor:/root/.bittensor \
    -it \
    --name miner-$MINER_SUFFIX \
    omega-bittensor:latest

docker attach miner-$MINER_SUFFIX
