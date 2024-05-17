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

from validator_api.communex.client import CommuneClient
from validator_api.communex.types import Ss58Address
from validator_api.communex._common import get_node_url

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

# This is only for testing. We'll remove for production
COMMUNE_SN17_KEYS = [
    '5DfXHcZPUJKbiwymhBh7GAjQQ5hCpWSBgHW4z22PRavSkw45',
    'THISISDEFINITELYACOMMUNESUBNETSEVENTEENHOTKEYYO'
    '5FN9ywbHh7J7YT47n2JAQ3Xk9LPYnnqWEdvirBLmANxHzcav',
    '5CLgUYNeN2MEYwcLDbUQ9TX1z2bMPo6QNgKYtniLJ1dZMkfn',
    '5FKeHZxLtUPwWizVZfwQ7tQfaM7d1zyQBRP1Gyz1cEnEeA2J',
    '5FYpYMgYNSegkWnmQ23YTP7c7qFuWqUwsy2Y6Y9oCjKBjLvL'
]

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

def check_commune_validator_hotkey(hotkey: str, modules_keys):
    if hotkey not in modules_keys.values():
        print("Commune validator key not found")
        return False
    
    return True

def authenticate_with_bittensor(hotkey, metagraph):
    if hotkey not in metagraph.hotkeys:
        return False

    uid = metagraph.hotkeys.index(hotkey)
    if not metagraph.validator_permit[uid]:
        print("Bittensor validator permit required")
        return False
    
    if metagraph.S[uid] < 1000 and NETWORK != "test":
        print("Bittensor validator requires 1000+ staked TAO")
        return False
    
    return True

def authenticate_with_commune(hotkey, commune_keys):
    if ENABLE_COMMUNE and not check_commune_validator_hotkey(hotkey, commune_keys):
        return False
    return True

async def main():
    app = FastAPI()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    commune_client = None
    commune_keys = None
    if ENABLE_COMMUNE:
        commune_client = CommuneClient(get_node_url(use_testnet=True if COMMUNE_NETWORK == "test" else False))
        commune_keys = commune_client.query_map_key(COMMUNE_NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)

                # Sync latest commune keys
                if ENABLE_COMMUNE:
                    commune_keys = commune_client.query_map_key(COMMUNE_NETUID)
                    print("commune keys synced")
            
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
        
        if not authenticate_with_bittensor(hotkey, metagraph) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        
        uid = None
        if ENABLE_COMMUNE and hotkey in commune_keys.values():
            # get uid of commune validator
            for key_uid, key_hotkey in commune_keys.items():
                if key_hotkey == hotkey:
                    uid = key_uid
                    break
            validator_chain = "commune"
        elif uid is None and hotkey in metagraph.hotkeys:
            # get uid of bittensor validator
            uid = metagraph.hotkeys.index(hotkey)
            validator_chain = "bittensor"
        
        start_time = time.time()
        # query the pinecone index to get novelty scores
        novelty_scores = await score.get_pinecone_novelty(metadata)
        print(f"Returning novelty scores={novelty_scores} for {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")
        return novelty_scores

    @app.post("/api/upload_video_metadata")
    async def upload_video_metadata(
        upload_data: VideoMetadataUpload,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> bool:
        
        if not authenticate_with_bittensor(hotkey, metagraph) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        
        uid = None
        if ENABLE_COMMUNE and hotkey in commune_keys.values():
            # get uid of commune validator
            for key_uid, key_hotkey in commune_keys.items():
                if key_hotkey == hotkey:
                    uid = key_uid
                    break
            validator_chain = "commune"
        elif uid is None and hotkey in metagraph.hotkeys:
            # get uid of bittensor validator
            uid = metagraph.hotkeys.index(hotkey)
            validator_chain = "bittensor"

        metadata = upload_data.metadata
        description_relevance_scores = upload_data.description_relevance_scores
        query_relevance_scores = upload_data.query_relevance_scores
        topic_query = upload_data.topic_query

        start_time = time.time()
        video_ids = await score.upload_video_metadata(metadata, description_relevance_scores, query_relevance_scores, topic_query, imagebind)
        print(f"Uploaded {len(video_ids)} video metadata from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")
        return True

    @app.post("/api/get_proxy")
    async def get_proxy(
        hotkey: Annotated[str, Depends(get_hotkey)]
    ) -> str:
        
        #hotkey = random.choice(COMMUNE_SN17_KEYS) # for testing purposes
        if not authenticate_with_bittensor(hotkey, metagraph) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
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
