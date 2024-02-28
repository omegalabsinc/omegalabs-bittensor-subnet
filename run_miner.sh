#!/bin/bash

MINER_SUFFIX=$1

docker build -f miner/Dockerfile -t miner:latest .

docker run --rm \
    --network="host" \
    --gpus all \
    --detach \
    -v $(pwd)/miner:/miner \
    -v $(pwd)/cache:/root/.cache \
    -v $(pwd)/miner/.checkpoints:/miner/.checkpoints \
    -v /home/tom/.bittensor:/root/.bittensor \
    -it \
    --name miner-$MINER_SUFFIX \
    miner:latest
docker attach miner-$MINER_SUFFIX
