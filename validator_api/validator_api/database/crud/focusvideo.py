from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, or_, exists, text
from typing import Optional, Dict, List, Any
import json
import traceback
import asyncio
import bittensor
from sqlalchemy.sql import select
import random
import uuid
import hashlib
from typing import Set

from validator_api.validator_api.config import (
    FOCUS_API_URL, 
    NETWORK, 
    NETUID,
    SUBNET_VIDEOS_WALLET_COLDKEY,
    SUBNET_VIDEOS_TAO_REWARD,
    SUBNET_VIDEOS_COUNT,
)
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
# Note: subnet_video_purchase model removed - using SubnetVideoRecord for all tracking
from validator_api.validator_api.database.models.subnet_video import SubnetVideoRecord


MIN_REWARD_TAO = 0.001
MIN_REWARD_ALPHA = 0.5

# Database-based video locking for GCP clusters
PURCHASE_LOCK_TIMEOUT_MINUTES = 5  # Lock expires after 5 minutes


def _video_id_to_lock_key(video_id: str) -> int:
    """
    Convert video_id string to a 64-bit integer for PostgreSQL advisory lock.
    Uses hash to ensure consistent mapping across all pods.
    """
    # Use MD5 hash and take first 8 bytes as signed 64-bit integer
    hash_bytes = hashlib.md5(video_id.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder='big', signed=True)


async def acquire_video_lock(video_id: str, miner_hotkey: str) -> bool:
    """
    Try to acquire a database lock for a video purchase.
    Uses PostgreSQL advisory lock to synchronize across multiple pods.
    Returns True if lock was acquired, False if already locked.
    """
    async with get_db_context() as db:
        try:
            lock_key = _video_id_to_lock_key(video_id)

            # Try to acquire PostgreSQL advisory lock (non-blocking)
            # pg_try_advisory_lock returns true if lock acquired, false if already held
            lock_result = await db.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": lock_key}
            )
            lock_acquired = lock_result.scalar()

            if not lock_acquired:
                print(f"Advisory lock not acquired for video {video_id} (lock_key={lock_key}) - another pod has it")
                return False

            print(f"Advisory lock acquired for video {video_id} (lock_key={lock_key})")

            # Now check if video is available (with row lock for safety)
            query = select(FocusVideoRecord).filter(
                FocusVideoRecord.video_id == video_id,
                FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED.value,
                FocusVideoRecord.deleted_at.is_(None),
            ).with_for_update()

            result = await db.execute(query)
            video_record = result.scalar_one_or_none()

            if video_record is None:
                # Video not available - release advisory lock
                await db.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key}
                )
                print(f"Video {video_id} not available, released advisory lock")
                return False

            # Update the video to PURCHASE_PENDING to lock it
            video_record.processing_state = FocusVideoStateInternal.PURCHASE_PENDING.value
            video_record.miner_hotkey = miner_hotkey
            video_record.updated_at = datetime.utcnow()

            db.add(video_record)
            await db.commit()

            # Release advisory lock after commit (state change is now persistent)
            await db.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": lock_key}
            )
            print(f"Video {video_id} locked for miner {miner_hotkey}, advisory lock released")
            return True

        except Exception as e:
            print(f"Error acquiring video lock for {video_id}: {e}")
            # Try to release advisory lock on error
            try:
                lock_key = _video_id_to_lock_key(video_id)
                await db.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key}
                )
            except:
                pass
            await db.rollback()
            return False


async def release_video_lock(video_id: str):
    """Release a video purchase lock by reverting to SUBMITTED state."""
    async with get_db_context() as db:
        try:
            query = select(FocusVideoRecord).filter(
                FocusVideoRecord.video_id == video_id,
                FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING.value,
                FocusVideoRecord.deleted_at.is_(None),
            )
            
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            
            if video_record:
                video_record.processing_state = FocusVideoStateInternal.SUBMITTED.value
                video_record.miner_hotkey = None  # Clear miner_hotkey when releasing lock
                video_record.updated_at = datetime.utcnow()
                db.add(video_record)
                await db.commit()
                print(f"Released video lock for {video_id}")
                
        except Exception as e:
            print(f"Error releasing video lock for {video_id}: {e}")
            await db.rollback()


async def is_video_locked(video_id: str) -> bool:
    """Check if a video is currently locked (in PURCHASE_PENDING state)."""
    async with get_db_context() as db:
        try:
            query = select(FocusVideoRecord).filter(
                FocusVideoRecord.video_id == video_id,
                FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING.value,
                FocusVideoRecord.deleted_at.is_(None),
            )
            
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            return video_record is not None
            
        except Exception as e:
            print(f"Error checking video lock for {video_id}: {e}")
            return False


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
    # print(f"User and boosted videos: {result.all()}")
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


async def _generate_subnet_videos(count: int = None) -> List[Dict[str, Any]]:
    """
    Generate synthetic subnet videos when no marketplace or user videos are available.
    First checks if subnet videos exist in database, returns those if found.
    Only creates new records if database table is empty.
    
    Args:
        count: Number of videos to generate (defaults to SUBNET_VIDEOS_COUNT)
    
    Returns:
        List of synthetic video records with video_id, video_score, and expected_reward_tao
    """
    if count is None or count == 0:
        count = SUBNET_VIDEOS_COUNT
    
    if not SUBNET_VIDEOS_WALLET_COLDKEY:
        print("WARNING: SUBNET_VIDEOS_WALLET_COLDKEY not configured, skipping subnet videos")
        return []
    
    # First check if we have existing subnet videos in the database
    existing_videos = await get_available_subnet_videos()
    if existing_videos:
        # Convert database result tuples to dictionary format
        subnet_videos = []
        for video_tuple in existing_videos:
            subnet_videos.append({
                "video_id": video_tuple[0],
                "video_score": video_tuple[1], 
                "expected_reward_tao": video_tuple[2]
            })
        print(f"Found {len(subnet_videos)} existing subnet videos in database")
        return subnet_videos
    
    # No existing videos found, create new ones
    print("No existing subnet videos found, generating new ones...")
    subnet_videos = await create_subnet_video_records(count)
    print(f"Generated and saved {len(subnet_videos)} new subnet videos to database")
    return subnet_videos


# Note: Subnet video purchase tracking is now handled directly in SubnetVideoRecord
# The purchase process updates the video record with purchase details


async def create_subnet_video_records(count: int = None) -> List[Dict[str, Any]]:
    """
    Create subnet video records in database for tracking like focus videos
    """
    if count is None:
        count = SUBNET_VIDEOS_COUNT
    
    if not SUBNET_VIDEOS_WALLET_COLDKEY:
        print("WARNING: SUBNET_VIDEOS_WALLET_COLDKEY not configured, skipping subnet video creation")
        return []
    
    async with get_db_context() as db:
        try:
            subnet_videos = []
            for _ in range(count):
                # Generate a unique video ID using UUID
                video_id = f"subnet_{uuid.uuid4().hex[:12]}"
                
                # Generate a random score between 0.5 and 1.0 for variety
                video_score = round(random.uniform(0.5, 1.0), 3)
                
                # Create database record
                subnet_video = SubnetVideoRecord(
                    video_id=video_id,
                    video_score=video_score,
                    expected_reward_tao=SUBNET_VIDEOS_TAO_REWARD,
                    processing_state="SUBMITTED",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                db.add(subnet_video)
                
                subnet_videos.append({
                    "video_id": video_id,
                    "video_score": video_score,
                    "expected_reward_tao": SUBNET_VIDEOS_TAO_REWARD
                })
            
            await db.commit()
            print(f"Created {len(subnet_videos)} subnet video records in database")
            return subnet_videos
            
        except Exception as e:
            print(f"Error creating subnet video records: {e}")
            await db.rollback()
            return []


async def get_available_subnet_videos() -> List[Dict[str, Any]]:
    """
    Get available subnet videos from database (SUBMITTED state)
    """
    async with get_db_context() as db:
        try:
            query = select(
                SubnetVideoRecord.video_id,
                SubnetVideoRecord.video_score,
                SubnetVideoRecord.expected_reward_tao
            ).filter(
                SubnetVideoRecord.processing_state == "SUBMITTED",
                SubnetVideoRecord.deleted_at.is_(None)
            ).order_by(SubnetVideoRecord.updated_at.asc()).limit(10)
            
            result = await db.execute(query)
            return result.all()
            
        except Exception as e:
            print(f"Error getting available subnet videos: {e}")
            return []


async def mark_subnet_video_purchase_pending(video_id: str, miner_hotkey: str) -> bool:
    """
    Mark a subnet video as PURCHASE_PENDING (equivalent to acquiring lock)
    Uses PostgreSQL advisory lock to synchronize across multiple pods.
    """
    async with get_db_context() as db:
        try:
            lock_key = _video_id_to_lock_key(video_id)

            # Try to acquire PostgreSQL advisory lock (non-blocking)
            lock_result = await db.execute(
                text("SELECT pg_try_advisory_lock(:lock_key)"),
                {"lock_key": lock_key}
            )
            lock_acquired = lock_result.scalar()

            if not lock_acquired:
                print(f"Advisory lock not acquired for subnet video {video_id} - another pod has it")
                return False

            print(f"Advisory lock acquired for subnet video {video_id}")

            query = select(SubnetVideoRecord).filter(
                SubnetVideoRecord.video_id == video_id,
                SubnetVideoRecord.processing_state == "SUBMITTED",
                SubnetVideoRecord.deleted_at.is_(None)
            ).with_for_update()

            result = await db.execute(query)
            subnet_video = result.scalar_one_or_none()

            if subnet_video is None:
                # Release advisory lock
                await db.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key}
                )
                print(f"Subnet video {video_id} not available, released advisory lock")
                return False

            subnet_video.processing_state = "PURCHASE_PENDING"
            subnet_video.miner_hotkey = miner_hotkey
            subnet_video.updated_at = datetime.utcnow()

            db.add(subnet_video)
            await db.commit()

            # Release advisory lock after commit
            await db.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": lock_key}
            )
            print(f"Subnet video {video_id} locked for miner {miner_hotkey}, advisory lock released")
            return True

        except Exception as e:
            print(f"Error marking subnet video purchase pending: {e}")
            try:
                lock_key = _video_id_to_lock_key(video_id)
                await db.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": lock_key}
                )
            except:
                pass
            await db.rollback()
            return False


async def mark_subnet_video_purchased(video_id: str, block_hash: str = None) -> bool:
    """
    Mark a subnet video as PURCHASED and set earned rewards
    """
    async with get_db_context() as db:
        try:
            query = select(SubnetVideoRecord).filter(
                SubnetVideoRecord.video_id == video_id,
                SubnetVideoRecord.processing_state == "PURCHASE_PENDING",
                SubnetVideoRecord.deleted_at.is_(None)
            )
            
            result = await db.execute(query)
            subnet_video = result.scalar_one_or_none()
            
            if subnet_video is None:
                return False
            
            subnet_video.processing_state = "PURCHASED"
            subnet_video.earned_reward_tao = subnet_video.expected_reward_tao
            subnet_video.block_hash = block_hash
            subnet_video.updated_at = datetime.utcnow()
            
            db.add(subnet_video)
            await db.commit()
            return True
            
        except Exception as e:
            print(f"Error marking subnet video purchased: {e}")
            await db.rollback()
            return False


async def revert_subnet_video_to_submitted(video_id: str) -> bool:
    """
    Revert a subnet video from PURCHASE_PENDING back to SUBMITTED (release lock)
    """
    async with get_db_context() as db:
        try:
            query = select(SubnetVideoRecord).filter(
                SubnetVideoRecord.video_id == video_id,
                SubnetVideoRecord.processing_state == "PURCHASE_PENDING",
                SubnetVideoRecord.deleted_at.is_(None)
            )
            
            result = await db.execute(query)
            subnet_video = result.scalar_one_or_none()
            
            if subnet_video is None:
                return False
            
            subnet_video.processing_state = "SUBMITTED"
            subnet_video.miner_hotkey = None
            subnet_video.updated_at = datetime.utcnow()
            
            db.add(subnet_video)
            await db.commit()
            return True
            
        except Exception as e:
            print(f"Error reverting subnet video to submitted: {e}")
            await db.rollback()
            return False


async def _get_purchaseable_videos() -> List[Dict[str, Any]]:
    """
    Fetch purchaseable videos - marketplace and subnet videos only.
    """
    async with get_db_context() as db:
        marketplace_limit = 9
        marketplace_items = await _fetch_marketplace_tasks(db, marketplace_limit)
        all_items = marketplace_items
        print(f"DEBUG: Got {len(marketplace_items)} marketplace items")

        # USER VIDEOS DISABLED - commented out
        # # Check if we can purchase user videos
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

        
        # Always add subnet videos alongside marketplace videos
        if not all_items:
            print("DEBUG: Generating subnet videos")
            subnet_videos = await _generate_subnet_videos()
            all_items += [(item["video_id"], item["video_score"], item["expected_reward_tao"]) for item in subnet_videos]
            print(f"DEBUG: Got {len(subnet_videos)} subnet videos")

        random.shuffle(all_items)

        # Filter out videos that are currently locked for purchase
        filtered_items = []
        for item in all_items:
            video_id = item[0]
            if not await is_video_locked(video_id):
                filtered_items.append(item)

        print(f"All items: {len(all_items)}, Available after lock filter: {len(filtered_items)}")

        return [
            {
                "video_id": item[0],
                "video_score": item[1],
                "expected_reward_tao": item[2]
            } for item in filtered_items
        ]


async def _alpha_to_tao_rate() -> float:
    async with bittensor.AsyncSubtensor(network=NETWORK) as subtensor:
        subnet = await subtensor.subnet(NETUID)
        balance = subnet.alpha_to_tao(1)
        return balance.tao


async def _already_purchased_max_focus_tao() -> bool:
    async with get_db_context() as db:
        # Get regular focus video purchases
        focus_query = select(func.sum(FocusVideoRecord.earned_reward_tao)).filter(
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.PURCHASED.value,
            FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24),
        )
        
        focus_result = await db.execute(focus_query)
        focus_earned_tao = focus_result.scalar() or 0
        
        # Get subnet video purchases
        subnet_query = select(func.sum(SubnetVideoRecord.earned_reward_tao)).filter(
            SubnetVideoRecord.processing_state == "PURCHASED",
            SubnetVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24),
        )
        
        subnet_result = await db.execute(subnet_query)
        subnet_earned_tao = subnet_result.scalar() or 0
        
        # Total TAO spent in last 24 hours
        total_earned_tao = focus_earned_tao + subnet_earned_tao

        max_focus_tao_per_day = await get_max_focus_tao_per_day()
        # Using 90% of the max focus tao per day as the effective max focus tao per day
        
        effective_max_focus_tao = max_focus_tao_per_day * 0.9

        # effective_max_focus_tao = 5
        
        print(f"Effective max focus tao: {effective_max_focus_tao}")
        print(f"Focus videos earned tao: {focus_earned_tao}")
        print(f"Subnet videos earned tao: {subnet_earned_tao}")
        print(f"Total earned tao: {total_earned_tao}")
        
        return total_earned_tao >= effective_max_focus_tao




class MinerPurchaseStats(BaseModel):
    total_focus_points: float
    max_focus_points: float
    focus_points_percentage: float


async def _get_miner_purchase_stats() -> Dict[str, MinerPurchaseStats]:
    print(f"\n{'='*60}")
    print(f"[_get_miner_purchase_stats] CACHE REFRESH START")
    print(f"{'='*60}")

    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    print(f"[_get_miner_purchase_stats] Querying purchases since: {cutoff_time}")
    print(f"[_get_miner_purchase_stats] Current time: {datetime.utcnow()}")

    async with get_db_context() as db:
        # Get regular focus video purchases - only include verified on-chain miner purchases
        # Filter by extrinsic_id to exclude admin/client purchases outside Bittensor
        focus_query = select(FocusVideoRecord).filter(
            FocusVideoRecord.processing_state
            == FocusVideoStateInternal.PURCHASED.value,
            FocusVideoRecord.updated_at >= cutoff_time,
            FocusVideoRecord.extrinsic_id.isnot(None),  # Must have on-chain transaction proof
            FocusVideoRecord.miner_hotkey.isnot(None),  # Must have miner hotkey
        )
        focus_result = await db.execute(focus_query)
        purchased_videos_records = focus_result.scalars().all()

        print(f"[_get_miner_purchase_stats] Focus videos found (with extrinsic_id): {len(purchased_videos_records)}")

        # Log each focus video found
        for i, record in enumerate(purchased_videos_records):
            print(f"  Focus #{i+1}: video_id={record.video_id}, miner_hotkey={record.miner_hotkey}, "
                  f"earned_tao={record.earned_reward_tao}, state={record.processing_state}, "
                  f"extrinsic_id={record.extrinsic_id}, updated_at={record.updated_at}")

        # Get subnet video purchases - only include verified on-chain miner purchases
        subnet_query = select(SubnetVideoRecord).filter(
            SubnetVideoRecord.processing_state == "PURCHASED",
            SubnetVideoRecord.updated_at >= cutoff_time,
            SubnetVideoRecord.extrinsic_id.isnot(None),  # Must have on-chain transaction proof
            SubnetVideoRecord.miner_hotkey.isnot(None),  # Must have miner hotkey
        )
        subnet_result = await db.execute(subnet_query)
        subnet_purchase_records = subnet_result.scalars().all()

        print(f"[_get_miner_purchase_stats] Subnet videos found (with extrinsic_id): {len(subnet_purchase_records)}")

        # Log each subnet video found
        for i, record in enumerate(subnet_purchase_records):
            print(f"  Subnet #{i+1}: video_id={record.video_id}, miner_hotkey={record.miner_hotkey}, "
                  f"earned_tao={record.earned_reward_tao}, state={record.processing_state}, "
                  f"extrinsic_id={record.extrinsic_id}, updated_at={record.updated_at}")

    # Calculate total earned tao from both sources
    focus_earned_tao = sum(
        record.earned_reward_tao or 0 for record in purchased_videos_records
    )
    subnet_earned_tao = sum(
        record.earned_reward_tao or 0 for record in subnet_purchase_records
    )
    total_earned_tao = focus_earned_tao + subnet_earned_tao

    print(f"[_get_miner_purchase_stats] Total TAO - Focus: {focus_earned_tao}, Subnet: {subnet_earned_tao}, Combined: {total_earned_tao}")

    # Group focus video records by miner hotkey
    videos_by_miner = {}
    for record in purchased_videos_records:
        if record.miner_hotkey is not None:
            if record.miner_hotkey not in videos_by_miner:
                videos_by_miner[record.miner_hotkey] = []
            videos_by_miner[record.miner_hotkey].append(record)

    # Group subnet video records by miner hotkey
    subnet_by_miner = {}
    for record in subnet_purchase_records:
        if record.miner_hotkey is not None:
            if record.miner_hotkey not in subnet_by_miner:
                subnet_by_miner[record.miner_hotkey] = []
            subnet_by_miner[record.miner_hotkey].append(record)

    # Process stats for each miner
    stats = {}
    all_miners = set(videos_by_miner.keys()) | set(subnet_by_miner.keys())

    print(f"[_get_miner_purchase_stats] Unique miners with purchases: {len(all_miners)}")

    for miner_hotkey in all_miners:
        # Calculate TAO from focus videos
        miner_focus_tao = sum(
            video_record.earned_reward_tao or 0
            for video_record in videos_by_miner.get(miner_hotkey, [])
        )

        # Calculate TAO from subnet videos
        miner_subnet_tao = sum(
            subnet_record.earned_reward_tao or 0
            for subnet_record in subnet_by_miner.get(miner_hotkey, [])
        )

        miner_total_tao = miner_focus_tao + miner_subnet_tao
        tao_percentage = (
            miner_total_tao / total_earned_tao if total_earned_tao > 0 else 0
        )

        stats[miner_hotkey] = MinerPurchaseStats(
            total_focus_points=miner_total_tao,
            max_focus_points=total_earned_tao,
            focus_points_percentage=tao_percentage,
        )

        print(f"  Miner {miner_hotkey[:16]}...: focus_tao={miner_focus_tao}, subnet_tao={miner_subnet_tao}, "
              f"total={miner_total_tao}, percentage={tao_percentage:.4f}")

    print(f"[_get_miner_purchase_stats] Returning stats for {len(stats)} miners")
    print(f"{'='*60}\n")

    return stats


class FocusVideoCache:
    def __init__(self):
        self._available_focus_cache = CachedValue(
            fetch_func=_get_purchaseable_videos, update_interval=60  # 1 minute
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



def is_subnet_video(video_id: str) -> bool:
    """Check if a video ID is a generated subnet video"""
    return video_id.startswith("subnet_")


async def get_video_owner_coldkey(db: AsyncSession, video_id: str) -> str:
    try:
        # Check if this is a subnet video
        if is_subnet_video(video_id):
            if not SUBNET_VIDEOS_WALLET_COLDKEY:
                raise HTTPException(500, detail="Subnet videos wallet coldkey not configured")
            return SUBNET_VIDEOS_WALLET_COLDKEY
        
        # Regular database lookup for actual focus videos
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
        # Handle subnet videos specially
        if is_subnet_video(video_id):
            if not SUBNET_VIDEOS_WALLET_COLDKEY:
                return {
                    "status": "error",
                    "message": "Subnet videos wallet coldkey not configured",
                }
            
            # Mark subnet video as PURCHASE_PENDING for verification process
            success = await mark_subnet_video_purchase_pending(video_id, miner_hotkey)
            if not success:
                return {
                    "status": "error",
                    "message": f"Subnet video {video_id} is not available or currently being purchased",
                }
            query = select(SubnetVideoRecord).filter(
            SubnetVideoRecord.video_id == video_id,
            SubnetVideoRecord.deleted_at.is_(None),
            SubnetVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING.value,
            SubnetVideoRecord.miner_hotkey == miner_hotkey,  # Ensure this miner locked it
            )

            result = await db.execute(query)
            video_record = result.scalar_one_or_none()
            expected_reward_tao = video_record.expected_reward_tao
            expected_reward_alpha = video_record.expected_reward_alpha
            return {
                "status": "success", 
                "price": expected_reward_tao,
                "price_alpha": expected_reward_alpha,  # Subnet videos don't use alpha rewards
                "is_subnet_video": True
            }

        # Try to acquire database lock for this video
        if not await acquire_video_lock(video_id, miner_hotkey):
            return {
                "status": "error",
                "message": f"Video {video_id} is not available or currently being purchased by another miner",
            }
        
        # Video is now locked, get the current video details
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None),
            FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING.value,
            FocusVideoRecord.miner_hotkey == miner_hotkey,  # Ensure this miner locked it
        )

        result = await db.execute(query)
        video_record = result.scalar_one_or_none()
        
        if video_record is None:
            await release_video_lock(video_id)
            return {
                "status": "error",
                "message": f"Video {video_id} lock was lost or invalid",
            }

        if video_record.expected_reward_tao is None:
            await release_video_lock(video_id)
            raise HTTPException(
                500,
                detail="The video record is missing the expected reward tao, investigate this bug",
            )

        # Create a copy of the values we need to avoid lazy loading issues
        expected_reward_tao = video_record.expected_reward_tao
        expected_reward_alpha = video_record.expected_reward_alpha

        return {
            "status": "success",
            "price": expected_reward_tao,
            "price_alpha": expected_reward_alpha,
            "is_subnet_video": False
        }

    except Exception as e:
        print(f"Error in check_availability: {str(e)}")
        traceback.print_exc()
        # Make sure to rollback the transaction in case of error
        await db.rollback()
        # Release the lock in case of error
        await release_video_lock(video_id)
        raise HTTPException(500, detail="Internal error")


async def check_video_metadata(
    db: AsyncSession, video_id: str, user_email: str, miner_hotkey: str
):
    try:
        # Handle subnet videos specially
        if is_subnet_video(video_id):
            # For subnet videos, we assume they're always "purchased" and return a fixed score
            return {
                "success": True, 
                "score": 0.8,  # Fixed score for subnet videos
                "is_subnet_video": True
            }
        
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

            return {"success": True, "score": video_score, "is_subnet_video": False}

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
    from validator_api.validator_api.config import FOCUS_API_KEY, FOCUS_API_URL
    url = f"{FOCUS_API_URL}/focus_videos/task_feedback/{video_id}"
    headers = {
        "X-SN24-API-Key": FOCUS_API_KEY
    }
    
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        return True
                    else:
                        print(f"Task feedback request failed with status {response.status}, attempt {attempt + 1}/3")
        except Exception as e:
            print(f"Task feedback request error: {e}, attempt {attempt + 1}/3")
    
    return False