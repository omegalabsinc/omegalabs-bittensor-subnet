import asyncio
import json
import psutil
import os
import random
import time
import traceback
from datetime import datetime
from tempfile import TemporaryDirectory
from traceback import print_exception
from typing import Annotated, Any, Dict, List, Optional

import aiohttp
import bittensor
import huggingface_hub
import mysql.connector
import sentry_sdk
import ulid
import uvicorn
from datasets import load_dataset
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Request,
    Security,
)
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from substrateinterface import Keypair
from validator_api.check_blocking import detect_blocking
from validator_api.communex._common import get_node_url
from validator_api.communex.client import CommuneClient
from validator_api.config import (
    API_KEY_NAME,
    API_KEYS,
    COMMUNE_NETUID,
    COMMUNE_NETWORK,
    DB_CONFIG,
    ENABLE_COMMUNE,
    FIXED_ALPHA_TAO_ESTIMATE,
    FOCUS_API_KEYS,
    FOCUS_API_URL,
    FOCUS_REWARDS_PERCENT,
    IMPORT_SCORE,
    IS_PROD,
    NETUID,
    NETWORK,
    PORT,
    PROXY_LIST,
    SENTRY_DSN,
)
from validator_api.cron.confirm_purchase import (
    confirm_transfer,
    confirm_video_purchased,
)
from validator_api.database import get_db, get_db_context
from validator_api.database.crud.focusvideo import (
    MinerPurchaseStats,
    TaskType,
    FocusVideoCache,
    check_availability,
    get_video_owner_coldkey,
    mark_video_rejected,
    set_focus_video_score,
)
from validator_api.database.models.focus_video_record import (
    FocusVideoRecord,
    FocusVideoStateExternal,
)
from validator_api.dataset_upload import audio_dataset_uploader, video_dataset_uploader
from validator_api.limiter import limiter
from validator_api.scoring.scoring_service import (
    FocusScoringService,
    LegitimacyCheckError,
    VideoTooLongError,
    VideoTooShortError,
    VideoUniquenessError,
)
from validator_api.utils.marketplace import (
    TASK_TYPE_MAP,
    get_max_focus_alpha_per_day,
    get_variable_reward_pool_alpha,
    get_fixed_reward_pool_alpha,
)
from validator_api.database.models.miner_bans import miner_banned_until

from omega.protocol import VideoMetadata
from sqlalchemy import select, update

print("IMPORT_SCORE:", IMPORT_SCORE)

if IMPORT_SCORE is not False:
    from validator_api import score
else:
    # remove cuda error on mac
    score = None


### Constants for OMEGA Metadata Dashboard ###
HF_DATASET = "omegalabsinc/omega-multimodal"
DATA_FILES_PREFIX = "default/train/"
MAX_FILES = 1
CACHE_FILE = "desc_embeddings_recent.json"
MIN_AGE = 60 * 60 * 48  # 2 days in seconds


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

focus_scoring_service = FocusScoringService()

print("SENTRY_DSN:", SENTRY_DSN)
sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)

# region Utility functions for OMEGA Metadata Dashboard


def get_timestamp_from_filename(filename: str):
    return (
        ulid.from_str(os.path.splitext(filename.split("/")[-1])[0])
        .timestamp()
        .timestamp
    )


def pull_and_cache_dataset() -> List[str]:
    # Get the list of files in the dataset repository
    omega_ds_files = huggingface_hub.repo_info(
        repo_id=HF_DATASET, repo_type="dataset"
    ).siblings

    # Filter files that match the DATA_FILES_PREFIX
    recent_files = [
        f.rfilename
        for f in omega_ds_files
        if f.rfilename.startswith(DATA_FILES_PREFIX)
        and time.time() - get_timestamp_from_filename(f.rfilename) < MIN_AGE
    ][:MAX_FILES]

    # Randomly sample up to MAX_FILES from the matching files
    sampled_files = random.sample(recent_files, min(MAX_FILES, len(recent_files)))

    # Load the dataset using the sampled files
    video_metadata = []
    with TemporaryDirectory() as temp_dir:
        omega_dataset = load_dataset(
            HF_DATASET, data_files=sampled_files, cache_dir=temp_dir
        )["train"]
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


# endregion Utility functions for OMEGA Metadata Dashboard


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header in API_KEYS:
        return api_key_header
    else:
        raise HTTPException(status_code=401, detail="Invalid API Key")


async def get_focus_api_key(focus_api_key_header: str = Security(focus_api_key_header)):
    if focus_api_key_header in FOCUS_API_KEYS:
        return focus_api_key_header
    else:
        raise HTTPException(status_code=401, detail="Invalid API Key")


class VideoMetadataUpload(BaseModel):
    metadata: List[VideoMetadata]
    description_relevance_scores: List[float]
    query_relevance_scores: List[float]
    topic_query: str
    novelty_score: Optional[float] = None
    total_score: Optional[float] = None
    miner_hotkey: Optional[str] = None


class FocusScoreResponse(BaseModel):
    video_id: str
    video_score: float
    video_details: dict


class VideoPurchaseRevert(BaseModel):
    video_id: str


def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    # print(f"Username: {credentials.username}, Password: {credentials.password}")
    try:
        keypair = Keypair(ss58_address=credentials.username)
        # print(f"Keypair: {keypair}")
        if keypair.verify(credentials.username, credentials.password):
            return credentials.username
    except Exception as e:
        print(f"Error verifying keypair: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error verifying keypair: {e}, make sure Basic Auth username is your hotkey SS58 address and the password is your hotkey's signature hex string (not private key!).",
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


async def run_focus_scoring(
    video_id: Annotated[str, Body()],
    focusing_task: Annotated[str, Body()],
    focusing_description: Annotated[str, Body()],
) -> Dict[str, Any]:
    score_details = None
    embeddings = None
    try:
        async with get_db_context() as db:
            query = select(FocusVideoRecord).filter(
                FocusVideoRecord.video_id == video_id,
                FocusVideoRecord.deleted_at.is_(None),
            )
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            if video_record is None:
                raise HTTPException(404, detail="Focus video not found")

        score_details, embeddings = await focus_scoring_service.score_video(
            video_id,
            focusing_task,
            focusing_description,
            bypass_checks=video_record.task_type == TaskType.MARKETPLACE.value,
        )
        print(f"Score for focus video <{video_id}>: {score_details.final_score}")
        MIN_FINAL_SCORE = 0.1
        # todo: measure and tune these
        # MIN_TASK_UNIQUENESS_SCORE = 0
        # MIN_VIDEO_UNIQUENESS_SCORE = 0
        # get the db after scoring the video so it's not open for too long
        async with get_db_context() as db:
            if video_record.task_type == TaskType.MARKETPLACE.value:
                # if the video is a marketplace video, we need the AI feedback to set the score
                # and then we need to update the video record to pending human review
                await set_focus_video_score(db, video_id, score_details, embeddings)
                update_stmt = (
                    update(FocusVideoRecord)
                    .where(FocusVideoRecord.video_id == video_id)
                    .values(
                        processing_state=FocusVideoStateExternal.PENDING_HUMAN_REVIEW.value
                    )
                )
                await db.execute(update_stmt)
                await db.commit()
                return {"success": True}
            if score_details.final_score < MIN_FINAL_SCORE:
                rejection_reason = f"""This video got a score of {score_details.final_score * 100:.2f}%, which is lower than the minimum score of {MIN_FINAL_SCORE * 100}%.
Feedback from AI: {score_details.completion_score_breakdown.rationale}"""
                await mark_video_rejected(
                    db,
                    video_id,
                    rejection_reason,
                    score_details=score_details,
                    embeddings=embeddings,
                )
            else:
                await set_focus_video_score(db, video_id, score_details, embeddings)
        print(f"finished get_focus_score | video_id <{video_id}>")
        return {"success": True}

    except Exception as e:
        exception_string = traceback.format_exc()
        error_string = f"{str(e)}\n{exception_string}"
        print(f"Error scoring focus video <{video_id}>: {error_string}")

        # Determine appropriate rejection reason based on error type
        if isinstance(e, VideoTooShortError):
            rejection_reason = "Video is too short. Please ensure the video is at least 10 seconds long."
        elif isinstance(e, VideoTooLongError):
            rejection_reason = "Video is too long. Please ensure the video is less than 10 minutes long."
        elif isinstance(e, VideoUniquenessError):
            rejection_reason = "Task recording is not unique. If you believe this is an error, please contact a team member."
        elif isinstance(e, LegitimacyCheckError):
            rejection_reason = "An anomaly was detected in the video. If you believe this is an error, please contact a team member via the OMEGA Focus Discord channel."
        else:
            rejection_reason = "Error scoring video"

        async with get_db_context() as db:
            await mark_video_rejected(
                db,
                video_id,
                rejection_reason,
                score_details=score_details,
                embeddings=embeddings,
                exception_string=exception_string,
            )
        return {"success": False, "error": error_string}


async def main():
    async def startup_event():
        print("Startup event fired, API starting.")

    async def shutdown_event():
        print("Shutdown event fired, attempting dataset upload of current batch.")
        video_dataset_uploader.submit()
        audio_dataset_uploader.submit()

    async def lifespan(app: FastAPI):
        await startup_event()
        yield
        await shutdown_event()

    app = FastAPI(lifespan=lifespan)
    # Mount the static directory to serve static files
    app.mount("/static", StaticFiles(directory="validator-api/static"), name="static")

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)
    focus_video_cache = FocusVideoCache()

    commune_client = None
    commune_keys = None
    if ENABLE_COMMUNE:
        commune_client = CommuneClient(
            get_node_url(use_testnet=True if COMMUNE_NETWORK == "test" else False)
        )
        commune_keys = update_commune_keys(commune_client, commune_keys)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            try:
                # Sync the metagraph.
                print("syncing metagraph")
                metagraph.sync(subtensor=subtensor)
                print("metagraph synced")

                # Sync latest commune keys
                if ENABLE_COMMUNE:
                    update_commune_keys(commune_client, commune_keys)
                    print("commune keys synced")

            # In case of unforeseen errors, the api will log the error and continue operations.
            except Exception as err:
                print("Error during metagraph sync", str(err))
                print_exception(type(err), err, err.__traceback__)

            await asyncio.sleep(90)

    @app.middleware("http")
    async def detect_blocking_middleware(request: Request, call_next):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        # Get basic auth credentials if available
        auth = request.headers.get("authorization")
        username = None
        if auth and auth.startswith("Basic "):
            try:
                import base64

                decoded = base64.b64decode(auth.split()[1]).decode()
                username = decoded.split(":")[0]
            except:
                pass

        async with detect_blocking(request.url.path, username):
            response = await call_next(request)

        mem_after = process.memory_info().rss
        mem_diff = mem_after - mem_before
        print(
            f"Memory change for {request.url.path}: {mem_diff / 1024 / 1024:.2f} MB, now at {mem_after / 1024 / 1024:.2f} MB"
        )

        return response

    @app.post("/api/get_pinecone_novelty")
    async def get_pinecone_novelty(
        metadata: List[VideoMetadata],
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> List[float]:
        print("get_pinecone_novelty()")

        if not authenticate_with_bittensor(
            hotkey, metagraph
        ) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required.",
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
        print(
            f"Returning novelty scores={novelty_scores} for {validator_chain} validator={uid} in {time.time() - start_time:.2f}s"
        )
        return novelty_scores

    @app.post("/api/upload_video_metadata")
    async def upload_video_metadata(
        upload_data: VideoMetadataUpload,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> bool:
        print("upload_video_metadata()")
        if not authenticate_with_bittensor(
            hotkey, metagraph
        ) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required.",
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
        video_ids = await score.upload_video_metadata(
            metadata, description_relevance_scores, query_relevance_scores, topic_query
        )
        print(
            f"Uploaded {len(video_ids)} video metadata from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s"
        )

        if upload_data.miner_hotkey is not None:
            # Calculate and upsert leaderboard data
            datapoints = len(video_ids)
            avg_desc_relevance = sum(description_relevance_scores) / len(
                description_relevance_scores
            )
            avg_query_relevance = sum(query_relevance_scores) / len(
                query_relevance_scores
            )
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
                cursor.execute(
                    query,
                    (
                        miner_hotkey,
                        is_bittensor,
                        is_commune,
                        datapoints,
                        avg_desc_relevance,
                        avg_query_relevance,
                        novelty_score,
                        total_score,
                    ),
                )
                connection.commit()
                print(
                    f"Upserted leaderboard data for {miner_hotkey} from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s"
                )

            except mysql.connector.Error as err:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error fetching data from MySQL database: {err}",
                )
            finally:
                if connection:
                    connection.close()
        else:
            print(
                "Skipping leaderboard update because either non-production environment or vali running outdated code."
            )

        return True

    @app.post("/api/upload_audio_metadata")
    @limiter.limit("1/minute")
    async def upload_audio_metadata(
        request: Request,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> bool:
        print("upload_audio_metadata()")

        if not authenticate_with_bittensor(
            hotkey, metagraph
        ) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required.",
            )

        uid = None
        if ENABLE_COMMUNE and hotkey in commune_keys.values():
            # get uid of commune validator
            for key_uid, key_hotkey in commune_keys.items():
                if key_hotkey == hotkey:
                    uid = key_uid
                    break
        elif uid is None and hotkey in metagraph.hotkeys:
            # get uid of bittensor validator
            uid = metagraph.hotkeys.index(hotkey)

        # Note: by passing in the request object, we can choose to load the body of the request when
        # we are ready to process it, which is important because the request body here can be huge
        audio_ids, upload_data = await score.upload_audio_metadata(request)
        # inverse_der = upload_data.inverse_der
        # audio_length_score = upload_data.audio_length_score
        # audio_quality_total_score = upload_data.audio_quality_total_score
        # audio_query_score = upload_data.audio_query_score
        # total_score = upload_data.total_score
        # print(f"Uploaded {len(audio_ids)} audio metadata from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")

        # if upload_data.miner_hotkey is not None:
        #     # Calculate and upsert leaderboard data
        #     datapoints = len(audio_ids)
        #     total_score = upload_data.total_score
        #     miner_hotkey = upload_data.miner_hotkey

        #     try:
        #         start_time = time.time()
        #         connection = connect_to_db()

        #         leaderboard_table_name = "miner_leaderboard_audio"
        #         if not IS_PROD:
        #             leaderboard_table_name += "_test"
        #         query = f"""
        #         INSERT INTO {leaderboard_table_name} (
        #             hotkey,
        #             is_bittensor,
        #             is_commune,
        #             datapoints,
        #             avg_der,
        #             avg_length_score,
        #             avg_quality_score,
        #             avg_query_score,
        #             avg_score,
        #             last_updated
        #         ) VALUES (
        #             %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        #         ) ON DUPLICATE KEY UPDATE
        #             datapoints = datapoints + VALUES(datapoints),
        #             avg_der = ((avg_der * (datapoints - VALUES(datapoints))) + (VALUES(avg_der) * VALUES(datapoints))) / datapoints,
        #             avg_length_score = ((avg_length_score * (datapoints - VALUES(datapoints))) + (VALUES(avg_length_score) * VALUES(datapoints))) / datapoints,
        #             avg_quality_score = ((avg_quality_score * (datapoints - VALUES(datapoints))) + (VALUES(avg_quality_score) * VALUES(datapoints))) / datapoints,
        #             avg_query_score = ((avg_query_score * (datapoints - VALUES(datapoints))) + (VALUES(avg_query_score) * VALUES(datapoints))) / datapoints,
        #             avg_score = ((avg_score * (datapoints - VALUES(datapoints))) + (VALUES(avg_score) * VALUES(datapoints))) / datapoints,
        #             last_updated = NOW();
        #         """
        #         cursor = connection.cursor()
        #         cursor.execute(query, (
        #             miner_hotkey,
        #             is_bittensor,
        #             is_commune,
        #             datapoints,
        #             inverse_der,
        #             audio_length_score,
        #             audio_quality_total_score,
        #             audio_query_score,
        #             total_score
        #         ))
        #         connection.commit()
        #         print(
        #             f"Upserted leaderboard data for {miner_hotkey} from {validator_chain} validator={uid} in {time.time() - start_time:.2f}s")

        #     except mysql.connector.Error as err:
        #         raise HTTPException(
        #             status_code=500, detail=f"Error fetching data from MySQL database: {err}")
        #     finally:
        #         if connection:
        #             connection.close()
        # else:
        #     print("Skipping leaderboard update because either non-production environment or vali running outdated code.")

        return True

    @app.post("/api/get_proxy")
    async def get_proxy(hotkey: Annotated[str, Depends(get_hotkey)]) -> str:
        if not authenticate_with_bittensor(
            hotkey, metagraph
        ) and not authenticate_with_commune(hotkey, commune_keys):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required.",
            )

        return random.choice(PROXY_LIST)

    ################ START OMEGA FOCUS ENDPOINTS ################
    @app.get("/api/focus/get_alpha_to_tao_rate")
    async def get_alpha_to_tao_rate(
        request: Request,
    ) -> float:
        try:
            return focus_video_cache.alpha_to_tao_rate()
        except Exception as e:
            print(e)
            return FIXED_ALPHA_TAO_ESTIMATE

    @app.post("/api/focus/get_focus_score")
    async def get_focus_score(
        api_key: str = Security(get_focus_api_key),
        video_id: Annotated[str, Body()] = None,
        focusing_task: Annotated[str, Body()] = None,
        focusing_description: Annotated[str, Body()] = None,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> Dict[str, bool]:
        print(f"starting get_focus_score | video_id <{video_id}>")

        async def run_focus_scoring_task(
            video_id: str, focusing_task: str, focusing_description: str
        ):
            await run_focus_scoring(video_id, focusing_task, focusing_description)

        background_tasks.add_task(
            run_focus_scoring_task, video_id, focusing_task, focusing_description
        )
        return {"success": True}

    @app.get("/api/focus/get_list")
    @limiter.limit("5/minute")
    async def _get_available_focus_video_list(request: Request):
        """
        Return available focus videos for purchase
        """
        return focus_video_cache.get_all_available_focus()

    @app.post("/api/focus/purchase")
    @limiter.limit("2/minute")
    async def purchase_video(
        request: Request,
        background_tasks: BackgroundTasks,
        video_id: Annotated[str, Body(embed=True)],
        hotkey: Annotated[str, Depends(get_hotkey)],
        db: AsyncSession = Depends(get_db),
    ):
        banned_until = await miner_banned_until(db, hotkey)
        if banned_until:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Miner is banned from purchasing focus videos until {banned_until} due to too many failed purchases in a row. Contact a team member if you believe this is an error.",
            )

        if focus_video_cache.already_purchased_max_focus_tao():
            print(
                "Purchases in the last 24 hours have reached the max focus tao limit."
            )
            raise HTTPException(
                400,
                "Purchases in the last 24 hours have reached the max focus tao limit, please try again later.",
            )
        print(f"purchase_video | video_id <{video_id}> hotkey <{hotkey}>")
        # run with_lock True
        availability = await check_availability(db, video_id, hotkey, True)
        print("availability", availability)
        if availability["status"] == "success":
            amount = availability["price"]
            video_owner_coldkey = await get_video_owner_coldkey(
                db, video_id
            )  # run with_lock True

            # Create a standalone async function for the background task
            async def run_confirm_video_purchased(video_id: str):
                await confirm_video_purchased(video_id, True)

            background_tasks.add_task(run_confirm_video_purchased, video_id)

            return {
                "status": "success",
                "address": video_owner_coldkey,
                "amount": amount,
            }
        else:
            return availability

    # @app.post("/api/focus/revert-pending-purchase")
    # @limiter.limit("4/minute")
    # async def revert_pending_purchase(
    #     request: Request,
    #     miner_hotkey: Annotated[str, Depends(get_hotkey)],
    #     video: VideoPurchaseRevert,
    #     db: AsyncSession = Depends(get_db),
    # ):
    #     return await mark_video_submitted(db, video.video_id, miner_hotkey, with_lock=True)

    @app.post("/api/focus/verify-purchase")
    @limiter.limit("4/minute")
    async def verify_purchase(
        request: Request,
        miner_hotkey: Annotated[str, Depends(get_hotkey)],
        video_id: Annotated[str, Body()],
        block_hash: Annotated[str, Body()],
        db: AsyncSession = Depends(get_db),
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ):
        async def run_stake(video_id):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{FOCUS_API_URL}/auth/stake",
                    json={"video_id": video_id},
                    headers={"FOCUS_API_KEY": FOCUS_API_KEYS[0]},
                ) as response:
                    res = await response.json()
                    print(f"Got res={res} from {FOCUS_API_URL}/auth/stake")
                    return res

        video_owner_coldkey = await get_video_owner_coldkey(db, video_id)
        result = await confirm_transfer(
            db, video_owner_coldkey, video_id, miner_hotkey, block_hash
        )
        if result:
            background_tasks.add_task(run_stake, video_id)
            return {
                "status": "success",
                "message": "Video purchase verification was successful",
            }
        else:
            return {
                "status": "error",
                "message": f"Video purchase verification failed for video_id {video_id} on block_hash {block_hash} by miner_hotkey {miner_hotkey}",
            }

    @app.get("/api/focus/miner_purchase_scores/{miner_hotkeys}")
    async def miner_purchase_scores_hotkeys(
        miner_hotkeys: str,
    ) -> Dict[str, MinerPurchaseStats]:
        return focus_video_cache.miner_purchase_stats()

    @app.get("/api/focus/miner_purchase_scores")
    async def miner_purchase_scores() -> Dict[str, MinerPurchaseStats]:
        return focus_video_cache.miner_purchase_stats()

    class TaskTypeMap(BaseModel):
        task_type_map: Dict[TaskType, float]

    @app.get("/api/focus/get_task_percentage_map")
    async def get_task_percentage_map():
        return TaskTypeMap(task_type_map=TASK_TYPE_MAP)

    @app.get("/api/focus/get_rewards_percent")
    async def get_rewards_percent():
        return FOCUS_REWARDS_PERCENT

    @app.get("/api/focus/get_max_focus_alpha")
    async def _get_max_focus_alpha() -> float:
        return await get_max_focus_alpha_per_day()

    @app.get("/api/focus/get_variable_reward_pool_alpha")
    async def _get_variable_reward_pool_alpha() -> float:
        return await get_variable_reward_pool_alpha()

    @app.get("/api/focus/get_fixed_reward_pool_alpha")
    async def _get_fixed_reward_pool_alpha() -> float:
        return await get_fixed_reward_pool_alpha()

    async def cache_max_focus_alpha():
        while True:
            """Re-caches the value of max_focus_tao."""
            print("cache_max_focus_alpha()")

            max_attempts = 3
            attempt = 0

            while attempt < max_attempts:
                try:
                    await get_max_focus_alpha_per_day()
                    break  # Exit the loop if the function succeeds

                # In case of unforeseen errors, the api will log the error and continue operations.
                except Exception as err:
                    attempt += 1
                    print(
                        f"Error during recaching of max_focus_alpha (Attempt {attempt}/{max_attempts}):",
                        str(err),
                    )

                    if attempt >= max_attempts:
                        print("Max attempts reached. Skipping this caching this cycle.")
                        break

            # Sleep in seconds
            await asyncio.sleep(1800)  # 30 minutes

    ################ END OMEGA FOCUS ENDPOINTS ################

    @app.get("/")
    @limiter.limit("10/minute")
    async def healthcheck(
        request: Request,
    ):
        return datetime.utcnow()

    ################ START MULTI-MODAL API / OPENTENSOR CONNECTOR ################
    @app.get("/api/mm/topics")
    async def get_mm_topics(api_key: str = Security(get_api_key)):
        try:
            connection = connect_to_db()
            query = "SELECT DISTINCT query FROM omega_multimodal"
            cursor = connection.cursor()
            cursor.execute(query)
            data = [row[0] for row in cursor.fetchall()]

            cursor.close()
            connection.close()
            return data
        except mysql.connector.Error as err:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from MySQL database: {err}",
            )

    @app.get("/api/mm/topic_video_count")
    async def get_mm_topic_video_count(api_key: str = Security(get_api_key)):
        try:
            connection = connect_to_db()
            query = "SELECT query, COUNT(*) AS num_videos FROM omega_multimodal GROUP BY query"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()

            cursor.close()
            connection.close()
            return data
        except mysql.connector.Error as err:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from MySQL database: {err}",
            )

    @app.get("/api/mm/topic_relevant/{topic}")
    async def get_mm_topic_relevant(
        api_key: str = Security(get_api_key), topic: str = Path(...)
    ):
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
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from MySQL database: {err}",
            )

    ################ END MULTI-MODAL API / OPENTENSOR CONNECTOR ################

    ################ START LEADERBOARD ################
    @app.get("/api/leaderboard")
    async def get_leaderboard_data(
        hotkey: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ):
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
                    "last_updated": "last_updated",
                }
                sort_column = valid_sort_columns.get(sort_by, sort_column)
            if sort_order:
                # Validate and map sort_order to actual values if necessary
                valid_sort_orders = {"asc": "ASC", "desc": "DESC"}
                sort_order = valid_sort_orders.get(sort_order.lower(), sort_order)

            query += f" ORDER BY {sort_column} {sort_order}"

            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            data = cursor.fetchall()

            cursor.close()
            connection.close()
            return data
        except mysql.connector.Error as err:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from MySQL database: {err}",
            )

    @app.get("/leaderboard")
    async def leaderboard():
        return FileResponse("./validator-api/static/leaderboard.html")

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
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching leaderboard dataset data from MySQL database: {err}",
            )

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
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching leaderboard miner data from MySQL database: {err}",
            )

    @app.get("/api/leaderboard-focus-data")
    async def get_leaderboard_focus_data():
        try:
            connection = connect_to_db()
            query = "SELECT * FROM focus_kpi_snapshots ORDER BY snapshot_date ASC"
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query)
            data = cursor.fetchall()

            cursor.close()
            connection.close()
            return data
        except mysql.connector.Error as err:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching focus kpi data from MySQL database: {err}",
            )

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
                    print(
                        f"Error during dataset sync (Attempt {attempt}/{max_attempts}):",
                        str(err),
                    )
                    # print_exception(type(err), err, err.__traceback__)

                    if attempt >= max_attempts:
                        print("Max attempts reached. Skipping this sync cycle.")
                        break

            # Sleep in seconds
            await asyncio.sleep(1800)  # 30 minutes

    @app.get("/dashboard/get-video-metadata")
    async def get_video_metadata(
        sort_by: Optional[str] = "submitted_at",
        sort_order: Optional[str] = "desc",
        page: Optional[int] = 1,
        items_per_page: Optional[int] = 50,
    ):
        print("get_video_metadata()")
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
                "submitted_at": 8,
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
                # Round description_relevance_score
                video[5] = round(video[5], 4)
                video[6] = round(video[6], 4)  # Round query_relevance_score
                date_time = datetime.fromtimestamp(video[8])
                video[8] = date_time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )  # Format submitted_at

            return {
                "total_items": total_items,
                "page": page,
                "items_per_page": items_per_page,
                "data": paginated_descriptions,
            }
        else:
            return {"error": "Cache file not found"}

    @app.get("/dashboard")
    async def dashboard():
        print("dashboard()")
        return FileResponse("validator-api/static/dashboard.html")

    ################ END DASHBOARD ################

    async def run_server():
        print("run_server()")
        config = uvicorn.Config(app=app, host="0.0.0.0", port=PORT)
        server = uvicorn.Server(config)
        await server.serve()

    server_task = asyncio.create_task(run_server())
    try:
        # Wait for the server to start
        tasks_list = [
            server_task,
            # resync_metagraph(),
            cache_max_focus_alpha(),
        ]
        if IS_PROD:
            tasks_list.append(resync_metagraph())
            tasks_list.append(resync_dataset())
        await asyncio.gather(*tasks_list)
    except asyncio.CancelledError:
        server_task.cancel()
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
