import asyncio
import os
from datetime import datetime
import time
from typing import Annotated, List
import random
from pydantic import BaseModel
from traceback import print_exception

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair

from communex.client import CommuneClient
from communex.module.client import ModuleClient
from communex.module.module import Module
from communex.types import Ss58Address

from omega.protocol import Videos, VideoMetadata
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async

from validator_api import score
from validator_api.config import TOPICS_LIST, PROXY_LIST, IS_PROD
from validator_api.dataset_upload import dataset_uploader

NETWORK = os.environ["NETWORK"]
NETUID = int(os.environ["NETUID"])

ENABLE_COMMUNE = bool(os.environ["ENABLE_COMMUNE"])
COMMUNE_NETWORK = os.environ["COMMUNE_NETWORK"]
COMMUNE_NETUID = int(os.environ["COMMUNE_NETUID"])

commune_client = None
if ENABLE_COMMUNE:
    commune_client = CommuneClient()

security = HTTPBasic()
imagebind = ImageBind()

class VideoMetadataUpload(BaseModel):
    metadata: List[VideoMetadata]
    description_relevance_scores: List[float]
    query_relevance_scores: List[float]
    topic_query: str

def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )

def get_commune_addresses(client: CommuneClient, netuid: int) -> dict[int, str]:
    """
    Retrieve all module addresses from the subnet.

    Args:
        client: The CommuneClient instance used to query the subnet.
        netuid: The unique identifier of the subnet.

    Returns:
        A dictionary mapping module IDs to their addresses.
    """

    # Makes a blockchain query for the miner addresses
    module_addreses = client.query_map_address(netuid)
    return module_addreses

def check_commune_validator_hotkey(hotkey: str):
    #keypair = Keypair(ss58_address=credentials.username)

    #modules_addresses = get_commune_addresses(commune_client, COMMUNE_NETUID)
    modules_keys = commune_client.query_map_key(COMMUNE_NETUID)
    val_ss58 = hotkey
    print("hotkey:", hotkey)
    print("modules_keys", modules_keys)
    if val_ss58 not in modules_keys.values():
        raise RuntimeError(f"validator key {val_ss58} is not registered in subnet")

async def main():
    app = FastAPI()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)
            
            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during metagraph sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(90)

    @app.on_event("shutdown")
    async def shutdown_event():
        dataset_uploader.submit()

    @app.post("/api/get_pinecone_novelty")
    async def get_pinecone_novelty(
        metadata: List[VideoMetadata],
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> List[float]:
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required",
            )

        uid = metagraph.hotkeys.index(hotkey)
        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )
        if metagraph.S[uid] < 1000 and NETWORK != "test":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator requires 1000+ staked TAO",
            )
        
        start_time = time.time()
        # query the pinecone index to get novelty scores
        novelty_scores = await score.get_pinecone_novelty(metadata)
        print(f"Returning novelty scores={novelty_scores} for validator={uid} in {time.time() - start_time:.2f}s")
        return novelty_scores

    @app.post("/api/upload_video_metadata")
    async def upload_video_metadata(
        upload_data: VideoMetadataUpload,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> bool:
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required",
            )

        uid = metagraph.hotkeys.index(hotkey)
        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )
        if metagraph.S[uid] < 1000 and NETWORK != "test":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator requires 1000+ staked TAO",
            )

        metadata = upload_data.metadata
        description_relevance_scores = upload_data.description_relevance_scores
        query_relevance_scores = upload_data.query_relevance_scores
        topic_query = upload_data.topic_query

        start_time = time.time()
        video_ids = await score.upload_video_metadata(metadata, description_relevance_scores, query_relevance_scores, topic_query, imagebind)
        print(f"Uploaded {len(video_ids)} video metadata from validator={uid} in {time.time() - start_time:.2f}s")
        return True

    @app.post("/api/get_proxy")
    async def get_proxy(
        hotkey: Annotated[str, Depends(get_hotkey)]
    ) -> str:
        check_commune_validator_hotkey(hotkey)
        
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required",
            )
        
        uid = metagraph.hotkeys.index(hotkey)
        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )
        if metagraph.S[uid] < 1000 and NETWORK != "test":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator requires 1000+ staked TAO",
            )
        
        return random.choice(PROXY_LIST)

    """ TO BE DEPRECATED """
    @app.post("/api/validate")
    async def validate(
        videos: Videos,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> float:
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required",
            )

        uid = metagraph.hotkeys.index(hotkey)

        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )

        start_time = time.time()
        computed_score = await score.score_and_upload_videos(videos, imagebind)
        print(f"Returning score={computed_score} for validator={uid} in {time.time() - start_time:.2f}s")
        return computed_score

    if not IS_PROD:
        @app.get("/api/count_unique")
        async def count_unique(
            videos: Videos,
        ) -> str:
            nunique = await score.get_num_unique_videos(videos)
            return f"{nunique} out of {len(videos.video_metadata)} submitted videos are unique"

        @app.get("/api/check_score")
        async def check_score(
            videos: Videos,
        ) -> dict:
            detailed_score = await score.score_videos_for_testing(videos, imagebind)
            return detailed_score

    @app.get("/api/topic")
    async def get_topic() -> str:
        return random.choice(TOPICS_LIST)
    
    @app.get("/api/topics")
    async def get_topics() -> List[str]:
        return TOPICS_LIST

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    await asyncio.gather(
        resync_metagraph(),
        asyncio.to_thread(uvicorn.run, app, host="0.0.0.0", port=8001)
    )


if __name__ == "__main__":
    asyncio.run(main())
