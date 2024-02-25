docker build -f validator-api/Dockerfile -t validator-api:latest .

docker run --rm \
    --network="host" \
    --gpus all \
    --detach \
    -v $(pwd)/validator-api:/validator-api \
    -v $(pwd)/cache:/root/.cache \
    -v $(pwd)/validator-api/.checkpoints:/validator-api/.checkpoints \
    --env-file validator-api/.env \
    -it \
    --name validator-api \
    validator-api:latest
docker attach validator-api

# docker run --rm \
#     --network="host" \
#     --gpus all \
#     --detach \
#     -v $(pwd)/validator-api:/validator-api \
#     -v $(pwd)/cache:/root/.cache \
#     -v $(pwd)/validator-api/.checkpoints:/validator-api/.checkpoints \
#     --env-file validator-api/.env \
#     --entrypoint bash \
#     -it \
#     --name validator-api-sh \
#     validator-api:latest
# docker attach validator-api-sh
