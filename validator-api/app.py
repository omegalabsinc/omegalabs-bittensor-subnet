import asyncio
import os
from datetime import datetime
import time
from typing import Annotated, List, Optional
import random
from pydantic import BaseModel

from traceback import print_exception

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette import status
from substrateinterface import Keypair

from validator_api.communex.client import CommuneClient
from validator_api.communex.types import Ss58Address
from validator_api.communex._common import get_node_url

from omega.protocol import Videos, VideoMetadata
from omega.imagebind_wrapper import ImageBind

from validator_api import score
from validator_api.config import (
    NETWORK, NETUID, 
    ENABLE_COMMUNE, COMMUNE_NETWORK, COMMUNE_NETUID,
    API_KEY_NAME, API_KEYS, DB_CONFIG,
    TOPICS_LIST, PROXY_LIST, IS_PROD
)
from validator_api.dataset_upload import dataset_uploader

import mysql.connector

def connect_to_db():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print("Error in connect_to_db while creating MySQL database connection:", err)

# define the APIKeyHeader for API authorization to our multi-modal endpoints
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

security = HTTPBasic()
imagebind = ImageBind()

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

class VideoMetadataUpload(BaseModel):
    metadata: List[VideoMetadata]
    description_relevance_scores: List[float]
    query_relevance_scores: List[float]
    topic_query: str
    novelty_score: Optional[float] = None
    total_score: Optional[float] = None
    miner_hotkey: Optional[str] = None

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
    if not metagraph.validator_permit[uid] and NETWORK != "test":
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
    # Mount the static directory to serve static files
    app.mount("/static", StaticFiles(directory="validator-api/static"), name="static")

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
        print("Shutdown event fired, attempting dataset upload of current batch.")
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
        is_bittensor = 0
        is_commune = 0
        if ENABLE_COMMUNE and hotkey in commune_keys.values():
            # get uid of commune validator
            for key_uid, key_hotkey in commune_keys.items():
                if key_hotkey == hotkey:
                    uid = key_uid
                    break
            validator_chain = "commune"
            is_commune = 1
        elif uid is None and hotkey in metagraph.hotkeys:
            # get uid of bittensor validator
            uid = metagraph.hotkeys.index(hotkey)
            validator_chain = "bittensor"
            is_bittensor = 1

        metadata = upload_data.metadata
        description_relevance_scores = upload_data.description_relevance_scores
        query_relevance_scores = upload_data.query_relevance_scores
        topic_query = upload_data.topic_query

        start_time = time.time()
        video_ids = await score.upload_video_metadata(metadata, description_relevance_scores, query_relevance_scores, topic_query, imagebind)
        print(f"Uploaded {len(video_ids)} video metadata from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")
        
        if upload_data.miner_hotkey is not None:
            # Calculate and upsert leaderboard data
            datapoints = len(video_ids)
            avg_desc_relevance = sum(description_relevance_scores) / len(description_relevance_scores)
            avg_query_relevance = sum(query_relevance_scores) / len(query_relevance_scores)
            novelty_score = upload_data.novelty_score
            total_score = upload_data.total_score
            miner_hotkey = upload_data.miner_hotkey
            
            try:
                start_time = time.time()
                connection = connect_to_db()

                leaderboard_table_name = "miner_leaderboard"
                if not IS_PROD:
                    leaderboard_table_name += "_test"
                query = f"""
                INSERT INTO {leaderboard_table_name} (
                    hotkey,
                    is_bittensor,
                    is_commune,
                    datapoints,
                    avg_desc_relevance,
                    avg_query_relevance,
                    avg_novelty,
                    avg_score,
                    last_updated
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                ) ON DUPLICATE KEY UPDATE
                    datapoints = datapoints + VALUES(datapoints),
                    avg_desc_relevance = ((avg_desc_relevance * (datapoints - VALUES(datapoints))) + (VALUES(avg_desc_relevance) * VALUES(datapoints))) / datapoints,
                    avg_query_relevance = ((avg_query_relevance * (datapoints - VALUES(datapoints))) + (VALUES(avg_query_relevance) * VALUES(datapoints))) / datapoints,
                    avg_novelty = ((avg_novelty * (datapoints - VALUES(datapoints))) + (VALUES(avg_novelty) * VALUES(datapoints))) / datapoints,
                    avg_score = ((avg_score * (datapoints - VALUES(datapoints))) + (VALUES(avg_score) * VALUES(datapoints))) / datapoints,
                    last_updated = NOW();
                """
                cursor = connection.cursor()
                cursor.execute(query, (
                    miner_hotkey,
                    is_bittensor,
                    is_commune,
                    datapoints,
                    avg_desc_relevance,
                    avg_query_relevance,
                    novelty_score,
                    total_score
                ))
                connection.commit()
                print(f"Upserted leaderboard data for {miner_hotkey} from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")
                
            except mysql.connector.Error as err:
                raise HTTPException(status_code=500, detail=f"Error fetching data from MySQL database: {err}")
            finally:
                if connection:
                    connection.close()
        else:
            print("Skipping leaderboard update because either non-production environment or vali running outdated code.")
        
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

    ################ START MULTI-MODAL API / OPENTENSOR CONNECTOR ################
    @app.get("/api/mm/topics")
    async def get_mm_topics(api_key: str = Security(get_api_key)):
        try:
            connection = connect_to_db()
            query = f"SELECT DISTINCT query FROM omega_multimodal"
            cursor = connection.cursor()
            cursor.execute(query)
            data = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            connection.close()
            return data
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching data from MySQL database: {err}")
        

    @app.get("/api/mm/topic_video_count")
    async def get_mm_topic_video_count(api_key: str = Security(get_api_key)):
        try:
            connection = connect_to_db()
            query = f"SELECT query, COUNT(*) AS num_videos FROM omega_multimodal GROUP BY query"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            return data        
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching data from MySQL database: {err}")


    @app.get("/api/mm/topic_relevant/{topic}")
    async def get_mm_topic_relevant(api_key: str = Security(get_api_key), topic: str = Path(...)):
        try:
            connection = connect_to_db()
            query = f"SELECT video_id, youtube_id, description, start_time, end_time FROM omega_multimodal where query = '{topic}' ORDER BY query_relevance_score DESC LIMIT 100"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            return data        
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching data from MySQL database: {err}")
    ################ END MULTI-MODAL API / OPENTENSOR CONNECTOR ################

    ################ START LEADERBOARD ################
    @app.get("/api/leaderboard")
    async def get_leaderboard_data(hotkey: Optional[str] = None, sort_by: Optional[str] = None, sort_order: Optional[str] = None):
        try:
            leaderboard_table_name = "miner_leaderboard"
            if not IS_PROD:
                leaderboard_table_name += "_test"
            connection = connect_to_db()
            query = f"SELECT * FROM {leaderboard_table_name}"
            params = []

            # Filter by hotkey if provided
            if hotkey:
                query += " WHERE hotkey = %s"
                params.append(hotkey)

            # Sort by the specified column if provided, default to 'datapoints'
            sort_column = "datapoints"  # Default sort column
            sort_order = "DESC"  # Default sort order
            if sort_by:
                # Validate and map sort_by to actual column names if necessary
                valid_sort_columns = {
                    "datapoints": "datapoints",
                    "avg_desc_relevance": "avg_desc_relevance",
                    "avg_query_relevance": "avg_query_relevance",
                    "avg_novelty": "avg_novelty",
                    "avg_score": "avg_score",
                    "last_updated": "last_updated"
                }
                sort_column = valid_sort_columns.get(sort_by, sort_column)
            if sort_order:
                # Validate and map sort_order to actual values if necessary
                valid_sort_orders = {
                    "asc": "ASC",
                    "desc": "DESC"
                }
                sort_order = valid_sort_orders.get(sort_order.lower(), sort_order)
            
            query += f" ORDER BY {sort_column} {sort_order}"

            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            return data        
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching data from MySQL database: {err}")
    
    @app.get("/leaderboard")
    def leaderboard():
        return FileResponse('validator-api/static/leaderboard.html')
    ################ END LEADERBOARD ################

    async def run_server():
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8001)
        server = uvicorn.Server(config)
        await server.serve()
    
    server_task = asyncio.create_task(run_server())
    try:
        await asyncio.gather(
            resync_metagraph(),
            server_task,
        )
    except asyncio.CancelledError:
        server_task.cancel()
        await server_task

if __name__ == "__main__":
    asyncio.run(main())
