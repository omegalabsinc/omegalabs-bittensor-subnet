#!/bin/bash

docker build -f Dockerfile -t omega-bittensor:latest .

docker run --rm \
    --network="host" \
    --gpus all \
    --detach \
    -v $(pwd):/app \
    -v $(pwd)/cache:/root/.cache \
    --env-file validator-api/.env \
    -it \
    --entrypoint bash \
    --name validator-api \
    omega-bittensor:latest \
    -c 'python validator-api/app.py'

docker attach validator-api
