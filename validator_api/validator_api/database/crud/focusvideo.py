from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, or_, exists
from typing import Optional, Dict, List, Any
import json
import traceback
import asyncio
import bittensor
from sqlalchemy.sql import select
import random

from validator_api.validator_api.config import FOCUS_API_URL, NETWORK, NETUID
from validator_api.validator_api.database import get_db_context
from validator_api.validator_api.database.models.focus_video_record import (
    FocusVideoRecord,
    FocusVideoInternal,
    FocusVideoStateInternal,
    TaskType,
)
from validator_api.validator_api.database.models.user import UserRecord
from validator_api.validator_api.database.models.user_roles import UserRoleRecordPG, UserRoleEnum
from validator_api.validator_api.utils.marketplace import (
    get_max_focus_alpha_per_day,
    get_max_focus_tao_per_day,
    get_variable_reward_pool_alpha,
)
from pydantic import BaseModel
from validator_api.validator_api.scoring.scoring_service import (
    VideoScore,
    FocusVideoEmbeddings,
)


MIN_REWARD_TAO = 0.001
MIN_REWARD_ALPHA = 0.5


class CachedValue:
    def __init__(self, fetch_func, update_interval: int = 90):
        """
        Args:
            fetch_func: An async function that fetches the new value.
            update_interval: How often (in seconds) to refresh the cache.
        """
        self._fetch_func = fetch_func
        self._update_interval = update_interval
        self._value = None
        self.is_initialized = False
        asyncio.create_task(self._background_update())

    async def _background_update(self):
        while True:
            try:
                new_value = await self._fetch_func()
                self._value = new_value
                if not self.is_initialized:
                    self.is_initialized = True
                    print(f"Cache {self._fetch_func.__name__} initialized")
                else:
                    print(
                        f"Cache {self._fetch_func.__name__} updated at {datetime.utcnow()}"
                    )
            except Exception as e:
                # Log error or handle as needed; do not crash the loop
                print(
                    f"Background cache update failed: {e}\n{traceback.format_exc()}")
            await asyncio.sleep(self._update_interval)

    def get(self):
        """Return the cached value (raises an exception if not yet initialized)."""
        if not self.is_initialized:
            raise Exception("Cache is not initialized yet")
        return self._value


async def _get_oldest_rewarded_user_id(db: AsyncSession) -> None:
    """
    Get the user that has waited the longest for a reward.
    Get users from the user_roles table with role "trusted".
    Sort by latest_mkt_rewarded_at ascending.
    Make sure the user has at least one MARKETPLACE video in the "submitted" state.
    """
    # Query to find all trusted users who have at least one MARKETPLACE video in SUBMITTED state
    query = (
        select(
            UserRecord.id,
            UserRecord.email,
            UserRecord.latest_mkt_rewarded_at,
            func.count(FocusVideoRecord.video_id).label(
                'submitted_video_count')
        )
        .join(
            UserRoleRecordPG,
            UserRecord.id == UserRoleRecordPG.user_id
        )
        .outerjoin(
            FocusVideoRecord,
            and_(
                FocusVideoRecord.user_id == UserRecord.id,
                FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED.value,
                FocusVideoRecord.task_type == TaskType.MARKETPLACE.value,
                FocusVideoRecord.deleted_at.is_(None)
            )
        )
        .filter(
            UserRoleRecordPG.role == UserRoleEnum.trusted.value
        )
        .group_by(
            UserRecord.id,
            UserRecord.email,
            UserRecord.latest_mkt_rewarded_at
        )
        .having(
            func.count(FocusVideoRecord.video_id) > 0
        )
        .order_by(
            # Handle NULL values by putting them first (oldest)
            UserRecord.latest_mkt_rewarded_at.asc().nulls_first()
        )
    )

    result = await db.execute(query)
    users = result.all()

    return users[0].id if users else None


async def _get_oldest_rewarded_user_videos(db: AsyncSession, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Get the videos from the user that has waited the longest for a reward
    """
    oldest_rewarded_user_id = await _get_oldest_rewarded_user_id(db)
    print(f"Oldest rewarded user ID: {oldest_rewarded_user_id}")

    oldest_user_videos = []
    if oldest_rewarded_user_id:
        oldest_user_query = (
            select(
                FocusVideoRecord.video_id,
                FocusVideoRecord.video_score,
                FocusVideoRecord.expected_reward_tao
            )
            .filter(
                FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED.value,
                FocusVideoRecord.deleted_at.is_(None),
                FocusVideoRecord.task_type == TaskType.MARKETPLACE.value,
                FocusVideoRecord.user_id == oldest_rewarded_user_id,
                FocusVideoRecord.expected_reward_tao > MIN_REWARD_TAO
            )
            .order_by(FocusVideoRecord.updated_at.asc())
            .limit(2)
        )
        result = await db.execute(oldest_user_query)
        oldest_user_videos = result.all()
    return oldest_user_videos


async def _fetch_marketplace_tasks(db: AsyncSession, limit: int = 9, oldest_rewarded_user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch marketplace videos ordered by oldest first
    But, always include videos from the user that has waited the longest for a reward
    """
    oldest_user_videos = await _get_oldest_rewarded_user_videos(db, limit)

    # Then, get the remaining marketplace videos (excluding the ones from the oldest user)
    remaining_limit = limit - len(oldest_user_videos)
    marketplace_videos = []
    if remaining_limit > 0:
        marketplace_videos_query = (
            select(
                FocusVideoRecord.video_id,
                FocusVideoRecord.video_score,
                FocusVideoRecord.expected_reward_tao
            )
            .filter(
                FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED.value,
                FocusVideoRecord.deleted_at.is_(None),
                FocusVideoRecord.task_type == TaskType.MARKETPLACE.value,
                FocusVideoRecord.expected_reward_tao > MIN_REWARD_TAO
            )
        )

        # Exclude videos from the oldest user if we have any, to avoid duplicates
        if oldest_user_videos:
            oldest_user_video_ids = [video[0] for video in oldest_user_videos]
            marketplace_videos_query = marketplace_videos_query.filter(
                FocusVideoRecord.video_id.notin_(oldest_user_video_ids)
            )

        marketplace_videos_query = marketplace_videos_query.order_by(
            FocusVideoRecord.updated_at.asc()).limit(remaining_limit)
        result = await db.execute(marketplace_videos_query)
        marketplace_videos = result.all()

    all_videos = oldest_user_videos + marketplace_videos
    print(f"All videos: {all_videos}")
    return all_videos


async def _fetch_user_and_boosted_tasks(db: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    user_and_boosted_videos_query = (
        select(
            FocusVideoRecord.video_id,
            FocusVideoRecord.video_score,
            FocusVideoRecord.expected_reward_tao
        )
        .filter(
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.SUBMITTED.value,
            FocusVideoRecord.deleted_at.is_(None),
            FocusVideoRecord.task_type != TaskType.MARKETPLACE.value,
        )
        .order_by(FocusVideoRecord.updated_at.asc())
        .limit(limit)
    )

    result = await db.execute(user_and_boosted_videos_query)
    print(f"User and boosted videos: {result.all()}")
    return result.all()


async def _can_purchase_user_videos(db: AsyncSession, mkt_videos_exist: bool) -> bool:
    """
    If there are no marketplace videos, or no user videos have been purchased in the last 4 videos,
    then we can purchase user videos
    """
    # print(f"DEBUG: _can_purchase_user_videos called with mkt_videos_exist={mkt_videos_exist}")
    
    if not mkt_videos_exist:
        # print("DEBUG: No marketplace videos exist, allowing user videos")
        return True
    
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASED.value,
        FocusVideoRecord.deleted_at.is_(None)
    ).order_by(FocusVideoRecord.updated_at.desc()).limit(4)
    result = await db.execute(query)
    last_4_videos = result.scalars().all()
    
    # print(f"DEBUG: Found {len(last_4_videos)} recent purchased videos")
    # for i, video in enumerate(last_4_videos):
    #     print(f"DEBUG:   {i+1}. Video ID: {video.video_id}, Task Type: {video.task_type}")
    
    for video in last_4_videos:
        if video.task_type != TaskType.MARKETPLACE.value:
            # print(f"DEBUG: Found non-marketplace video {video.video_id} with type {video.task_type}, blocking user videos")
            return False
    
    print("DEBUG: All recent videos are marketplace type, allowing user videos")
    return True


async def _get_purchaseable_videos() -> List[Dict[str, Any]]:
    """
    Fetch purchaseable videos
    If marketplace videos exist:
        - Return all marketplace videos (ordered by oldest updated_at)
        - Plus at most 1 other video type (ordered by oldest updated_at)
    If no marketplace videos exist:
        - Return up to 10 other videos (ordered by oldest updated_at)
    """
    async with get_db_context() as db:
        marketplace_limit = 9
        marketplace_items = await _fetch_marketplace_tasks(db, marketplace_limit)
        all_items = marketplace_items
        # print(f"DEBUG: Got {len(marketplace_items)} marketplace items")
        
        # Check if we can purchase user videos
        can_purchase_user = await _can_purchase_user_videos(db, len(marketplace_items) > 2)
        # print(f"DEBUG: len(marketplace_items) > 2: {len(marketplace_items) > 2}")
        print(f"DEBUG: Can purchase user videos: {can_purchase_user}")
        
        if can_purchase_user:
            no_marketplace_items = len(marketplace_items)
            allow_more_user_videos = no_marketplace_items < 3
            user_and_boosted_limit = 7 if allow_more_user_videos else 1
            print(f"DEBUG: Fetching user videos with limit: {user_and_boosted_limit}")
            user_and_boosted_items = await _fetch_user_and_boosted_tasks(db, user_and_boosted_limit)
            print(f"DEBUG: Got {len(user_and_boosted_items)} user/boosted items")
            all_items += user_and_boosted_items
        else:
            print("DEBUG: Not adding user videos due to policy")

        random.shuffle(all_items)
        print(f"All items: {all_items}")

        return [
            {
                "video_id": item[0],
                "video_score": item[1],
                "expected_reward_tao": item[2]
            } for item in all_items
        ]


async def _alpha_to_tao_rate() -> float:
    async with bittensor.AsyncSubtensor(network=NETWORK) as subtensor:
        subnet = await subtensor.subnet(NETUID)
        balance = subnet.alpha_to_tao(1)
        return balance.tao


async def _already_purchased_max_focus_tao() -> bool:
    async with get_db_context() as db:
        query = select(func.sum(FocusVideoRecord.earned_reward_tao)).filter(
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.PURCHASED.value,
            FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24),
        )

        result = await db.execute(query)
        total_earned_tao = result.scalar() or 0
        # max_focus_alpha_per_day = await get_max_focus_alpha_per_day()
        # effective_max_focus_alpha = max_focus_alpha_per_day *   0.9
        # effective_max_focus_tao = effective_max_focus_alpha * await _alpha_to_tao_rate()
        # print(f"Effective max focus tao: {effective_max_focus_tao}")
        # return total_earned_tao >= effective_max_focus_tao

        max_focus_tao_per_day = await get_max_focus_tao_per_day()
        # Using 90% of the max focus tao per day as the effective max focus tao per day
        effective_max_focus_tao = max_focus_tao_per_day * 0.9
        # print(f"Max focus tao per day: {max_focus_tao_per_day}")
        print(f"Effective max focus tao: {effective_max_focus_tao}")
        print(f"Total earned tao: {total_earned_tao}")
        return total_earned_tao >= effective_max_focus_tao




class MinerPurchaseStats(BaseModel):
    total_focus_points: float
    max_focus_points: float
    focus_points_percentage: float


async def _get_miner_purchase_stats() -> Dict[str, MinerPurchaseStats]:
    async with get_db_context() as db:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.PURCHASED.value,
            FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24),
        )
        result = await db.execute(query)
        purchased_videos_records = result.scalars().all()

    # Calculate total earned tao
    total_earned_tao = sum(
        record.earned_reward_tao or 0 for record in purchased_videos_records
    )

    # Group records by miner hotkey
    videos_by_miner = {}
    for record in purchased_videos_records:
        if record.miner_hotkey not in videos_by_miner:
            videos_by_miner[record.miner_hotkey] = []
        videos_by_miner[record.miner_hotkey].append(record)

    # Process stats for each miner
    stats = {}
    for miner_hotkey, miner_videos in videos_by_miner.items():
        miner_earned_tao = sum(
            video_record.earned_reward_tao for video_record in miner_videos
        )
        tao_percentage = (
            miner_earned_tao / total_earned_tao if total_earned_tao > 0 else 0
        )
        stats[miner_hotkey] = MinerPurchaseStats(
            total_focus_points=miner_earned_tao,
            max_focus_points=total_earned_tao,
            focus_points_percentage=tao_percentage,
        )

    return stats


class FocusVideoCache:
    def __init__(self):
        self._available_focus_cache = CachedValue(
            fetch_func=_get_purchaseable_videos, update_interval=180
        )
        self._alpha_to_tao_cache = CachedValue(fetch_func=_alpha_to_tao_rate)
        self._already_purchased_cache = CachedValue(
            fetch_func=_already_purchased_max_focus_tao
        )
        self._miner_purchase_stats_cache = CachedValue(
            fetch_func=_get_miner_purchase_stats, update_interval=180
        )

    def get_all_available_focus(self):
        try:
            return self._available_focus_cache.get()
        except Exception:
            return []

    def already_purchased_max_focus_tao(self) -> bool:
        return self._already_purchased_cache.get()

    def alpha_to_tao_rate(self) -> float:
        return self._alpha_to_tao_cache.get()

    def miner_purchase_stats(self) -> Dict[str, MinerPurchaseStats]:
        return self._miner_purchase_stats_cache.get()


async def get_video_owner_coldkey(db: AsyncSession, video_id: str) -> str:
    try:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(
                None)
        )
        result = await db.execute(query)
        video_record = result.scalar_one_or_none()

        if video_record is None:
            raise HTTPException(404, detail="Focus video not found")

        # Store the user_email to avoid lazy loading issues
        user_email = video_record.user_email

        query = select(UserRecord).filter(UserRecord.email == user_email)
        result = await db.execute(query)
        user_record = result.scalar_one_or_none()

        if user_record is None:
            raise HTTPException(404, detail="User not found")

        # Return the coldkey directly to avoid any potential lazy loading
        return user_record.coldkey
    except Exception as e:
        print(f"Error in get_video_owner_coldkey: {str(e)}")
        raise HTTPException(
            500, detail=f"Error retrieving video owner: {str(e)}")


async def check_availability(
    db: AsyncSession, video_id: str, miner_hotkey: str, with_lock: bool = False
):
    try:
        # Use explicit loading strategy to avoid lazy loading issues
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None),
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.SUBMITTED.value,  # is available for purchase
            # FocusVideoRecord.expected_reward_tao > MIN_REWARD_TAO,
            # FocusVideoRecord.expected_reward_alpha > MIN_REWARD_ALPHA,
        )

        if with_lock:
            query = query.with_for_update()

        result = await db.execute(query)
        video_record = result.scalar_one_or_none()
        # print(f"video_record: {video_record}")
        if video_record is None:
            return {
                "status": "error",
                "message": f"video {video_id} not found or not available for purchase",
            }

        if video_record.expected_reward_tao is None:
            raise HTTPException(
                500,
                detail="The video record is missing the expected reward tao, investigate this bug",
            )

        # TODO: This is commented out because expected_reward_alpha is not filled in for all videos yet, need to migrate
        # if video_record.expected_reward_alpha is None:
        #     raise HTTPException(500, detail="The video record is missing the expected reward alpha, investigate this bug")

        # Create a copy of the values we need to avoid lazy loading issues
        expected_reward_tao = video_record.expected_reward_tao
        expected_reward_alpha = video_record.expected_reward_alpha

        # mark the purchase as pending i.e. a miner has claimed the video for purchase and now just needs to pay
        video_record.processing_state = FocusVideoStateInternal.PURCHASE_PENDING.value
        video_record.miner_hotkey = miner_hotkey
        video_record.updated_at = datetime.utcnow()

        # NOTE: we don't set the video_record.earned_reward_tao here, because we don't know if the
        # miner will successfully purchase the video or not. We set it later in cron/confirm_purchase.py

        db.add(video_record)
        await db.commit()

        return {
            "status": "success",
            "price": expected_reward_tao,
            "price_alpha": expected_reward_alpha,
        }

    except Exception as e:
        print(f"Error in check_availability: {str(e)}")
        traceback.print_exc()
        # Make sure to rollback the transaction in case of error
        await db.rollback()
        raise HTTPException(500, detail="Internal error")


async def check_video_metadata(
    db: AsyncSession, video_id: str, user_email: str, miner_hotkey: str
):
    try:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.user_email == user_email,
            FocusVideoRecord.miner_hotkey == miner_hotkey,
            FocusVideoRecord.deleted_at.is_(None),
        )
        result = await db.execute(query)
        video_info = result.scalar_one_or_none()

        if (
            video_info is not None
            and video_info.processing_state == FocusVideoStateInternal.PURCHASED.value
        ):
            # # FV TODO: why do we need the task info?
            # task_info = db.query(models.Task).filter_by(id=video_info.task_id).first()

            # if task_info is not None:
            #     video_info.status = FocusVideoEnum.Submitted
            #     db.add(video_info)
            #     db.commit()
            #     video_score = await score.score_video(task_info.focusing_task, task_info.clip_link)
            #     print(f"Video score: {video_score}")
            #     return {
            #         'success': True,
            #         'score': video_score
            #     }

            # return {
            #     'success': False,
            #     'message': 'No task found.'
            # }

            # video_info.processing_state = FocusVideoStateInternal.VALIDATING
            db.add(video_info)
            await db.commit()

            # video_score = await score.score_video(task_info.focusing_task, task_info.clip_link)
            # print(f"Video score: {video_score}")
            video_score = video_info.video_score

            return {"success": True, "score": video_score}

        return {"success": False, "message": "No video found."}

    except Exception as e:
        print(e)
        return {"success": False, "message": "Internal Server Errror"}


async def set_focus_video_score(
    db: AsyncSession,
    video_id: str,
    score_details: VideoScore,
    embeddings: FocusVideoEmbeddings,
):
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(
            None)
    )
    result = await db.execute(query)
    video_record = result.scalar_one_or_none()

    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    video_record.video_score = score_details.final_score
    video_record.video_details = {
        **video_record.video_details,
        **json.loads(score_details.model_dump_json()),
    }
    video_record.embeddings = json.loads(embeddings.model_dump_json())
    video_record.processing_state = FocusVideoStateInternal.READY.value
    video_record.updated_at = datetime.utcnow()
    if video_record.task_type != TaskType.MARKETPLACE.value:
        video_record.task_type = (
            TaskType.BOOSTED.value
            if score_details.boosted_multiplier > 1.0
            else TaskType.USER.value
        )
    db.add(video_record)
    await db.commit()


async def mark_video_rejected(
    db: AsyncSession,
    video_id: str,
    rejection_reason: str,
    score_details: Optional[VideoScore] = None,
    embeddings: Optional[FocusVideoEmbeddings] = None,
    exception_string: Optional[str] = None,
):
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id, FocusVideoRecord.deleted_at.is_(
            None)
    )
    result = await db.execute(query)
    video_record = result.scalar_one_or_none()

    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    video_details = {**video_record.video_details}

    if score_details:
        video_details = {
            **video_details,
            **json.loads(score_details.model_dump_json()),
        }

    if exception_string:
        video_details["exception"] = exception_string

    if score_details or exception_string:
        video_record.video_details = video_details

    if embeddings:
        video_record.embeddings = json.loads(embeddings.model_dump_json())

    video_record.processing_state = FocusVideoStateInternal.REJECTED.value
    video_record.rejection_reason = rejection_reason
    db.add(video_record)
    await db.commit()


async def mark_video_submitted(
    db: AsyncSession, video_id: str, miner_hotkey: str, with_lock: bool = False
):
    # Mark video as "SUBMITTED" if in the "PURCHASE_PENDING" state.
    query = select(
        FocusVideoRecord
    ).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.processing_state
        == FocusVideoStateInternal.PURCHASE_PENDING.value,
        FocusVideoRecord.deleted_at.is_(None),
        FocusVideoRecord.miner_hotkey
        # make sure the miner requesting the cancellation is the one who was trying to buy it!
        == miner_hotkey,
    )
    if with_lock:
        query = query.with_for_update()

    result = await db.execute(query)
    video_record = result.scalar_one_or_none()

    if video_record is None:
        raise HTTPException(
            404,
            detail="Focus video not found or not in the correct state: PURCHASE_PENDING",
        )

    video_record.processing_state = FocusVideoStateInternal.SUBMITTED.value
    video_record.updated_at = datetime.utcnow()
    db.add(video_record)
    await db.commit()

async def generate_task_feedback(
    video_id: str,
) -> bool:
    import aiohttp
    from validator_api.validator_api.config import FOCUS_API_KEYS, FOCUS_API_URL
    url = f"{FOCUS_API_URL}/focus_videos/task_feedback/{video_id}"
    headers = {
        "X-SN24-API-Key": FOCUS_API_KEYS[0]
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return True
                    else:
                        print(f"Task feedback request failed with status {response.status}, attempt {attempt + 1}/3")
        except Exception as e:
            print(f"Task feedback request error: {e}, attempt {attempt + 1}/3")
    
    return False