import asyncio
import requests
import os
import json
from datetime import datetime
import time
from typing import Annotated, List, Optional, Dict, Any
import random
import json
from pydantic import BaseModel
import traceback

from tempfile import TemporaryDirectory
import huggingface_hub
from datasets import load_dataset
import ulid

from traceback import print_exception

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, Path, Security, BackgroundTasks
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette import status
from substrateinterface import Keypair

from sqlalchemy.orm import Session
from validator_api.database import get_db, get_db_context
from validator_api.database.crud.focusvideo import (
    get_all_available_focus, check_availability, get_purchased_list, check_video_metadata, 
    get_pending_focus, get_video_owner_coldkey, already_purchased_max_focus_tao, get_miner_purchase_stats, MinerPurchaseStats, set_focus_video_score, mark_video_rejected
)
from validator_api.utils.marketplace import get_max_focus_tao
from validator_api.cron.confirm_purchase import confirm_transfer, confirm_video_purchased
from validator_api.services.scoring_service import FocusScoringService

from validator_api.communex.client import CommuneClient
from validator_api.communex.types import Ss58Address
from validator_api.communex._common import get_node_url

from omega.protocol import Videos, VideoMetadata, FocusVideoMetadata
from omega.imagebind_wrapper import ImageBind, IMAGEBIND_VERSION

from validator_api import score
from validator_api.config import (
    NETWORK, NETUID, 
    ENABLE_COMMUNE, COMMUNE_NETWORK, COMMUNE_NETUID,
    API_KEY_NAME, API_KEYS, DB_CONFIG,
    TOPICS_LIST, PROXY_LIST, IS_PROD, 
    FOCUS_REWARDS_PERCENT, FOCUS_API_KEYS
)
from validator_api.dataset_upload import dataset_uploader

### Constants for OMEGA Metadata Dashboard ###
HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MAX_FILES = 1
CACHE_FILE = "desc_embeddings_recent.json"
MIN_AGE = 60 * 60 * 48  # 2 days in seconds

import mysql.connector

def connect_to_db():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print("Error in connect_to_db while creating MySQL database connection:", err)

# define the APIKeyHeader for API authorization to our multi-modal endpoints
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
focus_api_key_header = APIKeyHeader(name="FOCUS_API_KEY", auto_error=False)

security = HTTPBasic()
# imagebind_v2 = ImageBind(disable_lora=False) ## Commented segment to support Imagebind v1 and Imagebind v2
imagebind_v1 = ImageBind(disable_lora=True)

focus_scoring_service = FocusScoringService()

### Utility functions for OMEGA Metadata Dashboard ###
def get_timestamp_from_filename(filename: str):
    return ulid.from_str(os.path.splitext(filename.split("/")[-1])[0]).timestamp().timestamp

def pull_and_cache_dataset() -> List[str]:
    # Get the list of files in the dataset repository
    omega_ds_files = huggingface_hub.repo_info(repo_id=HF_DATASET, repo_type="dataset").siblings
    
    # Filter files that match the DATA_FILES_PREFIX
    recent_files = [
        f.rfilename
        for f in omega_ds_files if
        f.rfilename.startswith(DATA_FILES_PREFIX) and 
        time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]
    
    # Randomly sample up to MAX_FILES from the matching files
    sampled_files = random.sample(recent_files, min(MAX_FILES, len(recent_files)))
    
    # Load the dataset using the sampled files
    video_metadata = []
    with TemporaryDirectory() as temp_dir:
        omega_dataset = load_dataset(HF_DATASET, data_files=sampled_files, cache_dir=temp_dir)["train"]
        for i, entry in enumerate(omega_dataset):
            metadata = []
            if "description" in entry and "description_embed" in entry:
                metadata.append(entry["video_id"])
                metadata.append(entry["youtube_id"])
                metadata.append(entry["start_time"])
                metadata.append(entry["end_time"])
                metadata.append(entry["description"])
                metadata.append(entry["description_relevance_score"])
                metadata.append(entry["query_relevance_score"])
                metadata.append(entry["query"])
                metadata.append(entry["submitted_at"])
                video_metadata.append(metadata)
    
    # Cache the descriptions to a local file
    with open(CACHE_FILE, "w") as f:
        json.dump(video_metadata, f)
    
    return True
### End Utility functions for OMEGA Metadata Dashboard ###

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    
async def get_focus_api_key(focus_api_key_header: str = Security(focus_api_key_header)):
    if focus_api_key_header in FOCUS_API_KEYS:
        return focus_api_key_header
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
    
class FocusMetadataUpload(BaseModel):
    metadata: List[FocusVideoMetadata]
    total_score: Optional[float] = None
    miner_hotkey: Optional[str] = None

class FocusScoreResponse(BaseModel):
    video_id: str
    video_score: float
    video_details: dict

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

def update_commune_keys(commune_client, commune_keys):
    try:
        return commune_client.query_map_key(COMMUNE_NETUID)
    except Exception as err:
        print("Error during commune keys update", str(err))
        return commune_keys

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
        commune_keys = update_commune_keys(commune_client, commune_keys)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                metagraph.sync(subtensor=subtensor)

                # Sync latest commune keys
                if ENABLE_COMMUNE:
                    commune_keys = update_commune_keys(commune_client, commune_keys)
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
        video_ids = await score.upload_video_metadata(metadata, description_relevance_scores, query_relevance_scores, topic_query, imagebind_v1)
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
        
        if not authenticate_with_bittensor(hotkey, metagraph) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        
        return random.choice(PROXY_LIST)
    
    ################ START OMEGA FOCUS ENDPOINTS ################
    async def run_focus_scoring(
        video_id: Annotated[str, Body()],
        focusing_task: Annotated[str, Body()],
        focusing_description: Annotated[str, Body()]
    ) -> Dict[str, Any]:
        try:
            response = await focus_scoring_service.score_video(video_id, focusing_task, focusing_description)
            print(f"Score for focus video <{video_id}>: {response.combined_score}")
            minimum_score = 0.1
            # get the db after scoring the video so it's not open for too long
            with get_db_context() as db:
                if response.combined_score < minimum_score:
                    rejection_reason = f"This video got a score of {response.combined_score * 100:.2f}%, which is lower than the minimum score of {minimum_score * 100}%."
                    mark_video_rejected(db, video_id, rejection_reason=rejection_reason)
                else:
                    set_focus_video_score(db, video_id, response)
            return { "success": True }
        except Exception as e:
            error_string = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Error scoring focus video <{video_id}>: {error_string}")
            mark_video_rejected(db, video_id, rejection_reason=error_string)
            return { "success": False, "error": error_string }

    @app.post("/api/focus/get_focus_score")
    async def get_focus_score(
        api_key: str = Security(get_focus_api_key),
        video_id: Annotated[str, Body()] = None,
        focusing_task: Annotated[str, Body()] = None,
        focusing_description: Annotated[str, Body()] = None,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> Dict[str, bool]:
        # await run_focus_scoring(video_id, focusing_task, focusing_description, db)
        background_tasks.add_task(run_focus_scoring, video_id, focusing_task, focusing_description)
        return { "success": True }

    @app.get("/api/focus/get_list")
    async def _get_available_focus_video_list(
        db: Session=Depends(get_db)
    ):
        """
        Return all available focus videos
        """
        return get_all_available_focus(db)

    # FV TODO: let's do proper miner auth here instead, and then from the retrieved hotkey, we can also
    # retrieve the coldkey and use that to confirm the transfer
    @app.post("/api/focus/purchase")
    async def purchase_video(
        background_tasks: BackgroundTasks,
        video_id: Annotated[str, Body()],
        miner_hotkey: Annotated[str, Body()],
        db: Session=Depends(get_db),
    ):
        if await already_purchased_max_focus_tao(db):
            print("Purchases in the last 24 hours have reached the max focus tao limit.")
            raise HTTPException(400, "Purchases in the last 24 hours have reached the max focus tao limit, please try again later.")

        availability = await check_availability(db, video_id, miner_hotkey)
        print('availability', availability)
        if availability['status'] == 'success':
            amount = availability['price']
            video_owner_coldkey = get_video_owner_coldkey(db, video_id)
            background_tasks.add_task(confirm_video_purchased, video_id)
            return {
                'status': 'success',
                'address': video_owner_coldkey,
                'amount': amount,
            }
        else:
            return availability
        
    @app.post("/api/focus/verify-purchase")
    async def verify_purchase(
        miner_hotkey: Annotated[str, Body()],
        video_id: Annotated[str, Body()],
        block_hash: Annotated[str, Body()],
        db: Session=Depends(get_db),
    ):
        video_owner_coldkey = get_video_owner_coldkey(db, video_id)
        result = await confirm_transfer(db, video_owner_coldkey, video_id, miner_hotkey, block_hash)
        if result:
            return {
                'status': 'success',
                'message': 'Video purchase verification was successful'
            }
        else:
            return {
                'status': 'error',
                'message': f'Video purchase verification failed for video_id {video_id} on block_hash {block_hash} by miner_hotkey {miner_hotkey}'
            }

    @app.get('/api/focus/miner_purchase_score/{miner_hotkey}')
    async def miner_purchase_score(
        miner_hotkey: str,
        db: Session = Depends(get_db)
    ) -> MinerPurchaseStats:
        return get_miner_purchase_stats(db, miner_hotkey)

    @app.get('/api/focus/miner_purchase_scores/{miner_hotkey_list}')
    async def miner_purchase_scores(
        miner_hotkey_list: str,
        db: Session = Depends(get_db)
    ) -> Dict[str, MinerPurchaseStats]:
        return {
            hotkey: get_miner_purchase_stats(db, hotkey)
            for hotkey in miner_hotkey_list.split(',')
        }
    
    @app.get('/api/focus/get_rewards_percent')
    async def get_rewards_percent():
        return FOCUS_REWARDS_PERCENT
    
    @app.get('/api/focus/get_max_focus_tao')
    async def _get_max_focus_tao():
        return await get_max_focus_tao()
    
    async def cache_max_focus_tao():
        while True:
            """Re-caches the value of max_focus_tao."""
            print("cache_max_focus_tao()")

            max_attempts = 3
            attempt = 0

            while attempt < max_attempts:
                try:
                    max_focus_tao = await get_max_focus_tao()
                    break  # Exit the loop if the function succeeds

                # In case of unforeseen errors, the api will log the error and continue operations.
                except Exception as err:
                    attempt += 1
                    print(f"Error during recaching of max_focus_tao (Attempt {attempt}/{max_attempts}):", str(err))

                    if attempt >= max_attempts:
                        print("Max attempts reached. Skipping this caching this cycle.")
                        break

            # Sleep in seconds
            await asyncio.sleep(1800) # 30 minutes
    ################ END OMEGA FOCUS ENDPOINTS ################
    
    """ TO BE DEPRECATED """
    @app.post("/api/validate")
    async def validate(
        videos: Videos,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> float:
        if not authenticate_with_bittensor(hotkey, metagraph):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Valid hotkey required.",
            )
        uid = metagraph.hotkeys.index(hotkey)
        
        start_time = time.time()

        if not hasattr(videos, 'imagebind_version'):
            print("`videos` object does not have `imagebind_version` attribute, using original model")
        elif videos.imagebind_version is None:
            print("imagebind_version is None, using original model")
        elif videos.imagebind_version != IMAGEBIND_VERSION:
            print(f"imagebind_version is {videos.imagebind_version}, using original model")
        else:
            print(f"imagebind_version is {IMAGEBIND_VERSION}, using new model")
        
        '''
        ## Commented segment to support Imagebind v1 and Imagebind v2
        
        # handle youtube video metadata
        if videos.imagebind_version is not None and videos.imagebind_version == IMAGEBIND_VERSION:
            youtube_rewards = await score.score_and_upload_videos(videos, imagebind_v1)
        else:
            youtube_rewards = await score.score_and_upload_videos(videos, imagebind_v2)
        '''
        youtube_rewards = await score.score_and_upload_videos(videos, imagebind_v1)

        if youtube_rewards is None:
            print("YouTube rewards are empty, returning None")
            return None
        
        total_rewards: float = youtube_rewards
        
        print(f"Total Rewards: {total_rewards}")
        print(f"Returning score={total_rewards} for validator={uid} in {time.time() - start_time:.2f}s")

        return total_rewards

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
            detailed_score = await score.score_videos_for_testing(videos, imagebind_v1)
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
        return FileResponse('./validator-api/static/leaderboard.html')
    
    @app.get("/api/leaderboard-dataset-data")
    async def get_leaderboard_dataset_data():
        try:
            connection = connect_to_db()
            query = "SELECT * FROM hf_dataset_snapshots ORDER BY snapshot_date ASC"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            return data        
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching leaderboard dataset data from MySQL database: {err}")
        
    @app.get("/api/leaderboard-miner-data")
    async def get_leaderboard_miner_data(hotkey: Optional[str] = None):
        try:
            connection = connect_to_db()
            params = []

            query = "SELECT * FROM miner_leaderboard_snapshots wHERE 1=1"

             # Filter by hotkey if provided
            if hotkey:
                query += " AND hotkey = %s"
                params.append(hotkey)

            query += " ORDER BY snapshot_date ASC"

            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            return data        
        except mysql.connector.Error as err:
            raise HTTPException(status_code=500, detail=f"Error fetching leaderboard miner data from MySQL database: {err}")

    ################ END LEADERBOARD ################

    ################ START DASHBOARD ################
    async def resync_dataset():
        while True:
            """Resyncs the dataset by updating our JSON data source from the huggingface dataset."""
            print("resync_dataset()")

            max_attempts = 3
            attempt = 0

            while attempt < max_attempts:
                try:
                    pull_and_cache_dataset()
                    break  # Exit the loop if the function succeeds

                # In case of unforeseen errors, the api will log the error and continue operations.
                except Exception as err:
                    attempt += 1
                    print(f"Error during dataset sync (Attempt {attempt}/{max_attempts}):", str(err))
                    #print_exception(type(err), err, err.__traceback__)

                    if attempt >= max_attempts:
                        print("Max attempts reached. Skipping this sync cycle.")
                        break

            # Sleep in seconds
            await asyncio.sleep(1800) # 30 minutes

    @app.get("/dashboard/get-video-metadata")
    async def get_video_metadata(
        sort_by: Optional[str] = "submitted_at",
        sort_order: Optional[str] = "desc",
        page: Optional[int] = 1,
        items_per_page: Optional[int] = 50
    ):
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                descriptions = json.load(f)
            
            # Define a mapping from sort_by parameter to the index in the metadata list
            sort_index_mapping = {
                "video_id": 0,
                "youtube_id": 1,
                "start_time": 2,
                "end_time": 3,
                "description": 4,
                "description_relevance_score": 5,
                "query_relevance_score": 6,
                "query": 7,
                "submitted_at": 8
            }
            
            if sort_by and sort_by in sort_index_mapping:
                index = sort_index_mapping[sort_by]
                reverse = sort_order == "desc"
                descriptions.sort(key=lambda x: x[index], reverse=reverse)
            
            # Pagination logic
            total_items = len(descriptions)
            start = (page - 1) * items_per_page
            end = start + items_per_page
            paginated_descriptions = descriptions[start:end]

            for video in paginated_descriptions:
                video[0] = ".." + str(video[0])[:6]
                video[5] = round(video[5], 4)  # Round description_relevance_score
                video[6] = round(video[6], 4)  # Round query_relevance_score
                date_time = datetime.fromtimestamp(video[8])
                video[8] = date_time.strftime('%Y-%m-%d %H:%M:%S')  # Format submitted_at
            
            return {
                "total_items": total_items,
                "page": page,
                "items_per_page": items_per_page,
                "data": paginated_descriptions
            }
        else:
            return {"error": "Cache file not found"}
    
    @app.get("/dashboard")
    def dashboard():
        return FileResponse('validator-api/static/dashboard.html')
    ################ END DASHBOARD ################

    async def run_server():
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8001)
        server = uvicorn.Server(config)
        await server.serve()
    
    server_task = asyncio.create_task(run_server())
    try:
        # Wait for the server to start
        await asyncio.gather(
            server_task,
            resync_metagraph(),
            cache_max_focus_tao(),
            resync_dataset(),
        )
    except asyncio.CancelledError:
        server_task.cancel()
        await server_task

if __name__ == "__main__":
    asyncio.run(main())
