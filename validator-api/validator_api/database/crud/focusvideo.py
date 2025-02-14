from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, Float
from typing import List, Optional, Dict
import json
import time
import asyncio
import bittensor

from validator_api.config import NETWORK, NETUID
from validator_api.database import get_db_context
from validator_api.database.models.focus_video_record import FocusVideoRecord, FocusVideoInternal, FocusVideoStateInternal, TaskType
from validator_api.database.models.user import UserRecord
# from validator_api.utils.marketplace import get_max_focus_tao, get_purchase_max_focus_tao, get_max_focus_points_available_today
from validator_api.utils.marketplace import get_max_focus_alpha_per_day, get_purchase_max_focus_alpha, get_max_focus_points_available_today
from pydantic import BaseModel
from validator_api.scoring.scoring_service import VideoScore, FocusVideoEmbeddings


MIN_REWARD_TAO = 0.001
MIN_REWARD_ALPHA = .5


class CachedValue:
    def __init__(self, duration: int = 90):
        self._value = None
        self._timestamp = 0
        self._duration = duration
        self._mutex = asyncio.Lock()

    def is_valid(self) -> bool:
        return (
            self._value is not None and 
            time.time() - self._timestamp < self._duration
        )

    async def get_or_update(self, fetch_func):
        if self.is_valid():
            return self._value

        try:
            async with self._mutex:
                # Double check after acquiring lock
                if not self.is_valid():
                    self._value = await fetch_func()
                    self._timestamp = time.time()
            return self._value

        except Exception as e:
            print(e)
            raise HTTPException(500, detail="Internal error")

async def _fetch_available_focus(db: Session):
    # Show oldest videos first so they get rewarded fastest
    items = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED,
        FocusVideoRecord.deleted_at.is_(None),
        FocusVideoRecord.expected_reward_tao > MIN_REWARD_TAO,
        # FocusVideoRecord.expected_reward_alpha > MIN_REWARD_ALPHA,
    ).order_by(FocusVideoRecord.updated_at.asc()).limit(10).all()
    return [FocusVideoInternal.model_validate(record) for record in items]

_available_focus_cache = CachedValue()

async def get_all_available_focus(db: Session):
    return await _available_focus_cache.get_or_update(
        lambda: _fetch_available_focus(db)
    )

def get_pending_focus(
    db: Session,
    miner_hotkey: str
):
    try:
        items = db.query(FocusVideoRecord).filter_by(
            processing_state=FocusVideoStateInternal.PURCHASE_PENDING,
            miner_hotkey=miner_hotkey
        ).all()
        return items
    
    except Exception as e:
        print(e)
        raise HTTPException(500, detail="Internal error")
    
async def check_availability(
    db: Session,
    video_id: str,
    miner_hotkey: str,
    with_lock: bool = False
):
    try:
        video_record = db.query(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None),
            FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED,  # is available for purchase
            FocusVideoRecord.expected_reward_tao > MIN_REWARD_TAO,
            # FocusVideoRecord.expected_reward_alpha > MIN_REWARD_ALPHA,
        )
        if with_lock:
            video_record = video_record.with_for_update()
        video_record = video_record.first()

        if video_record is None:
            return {
                'status': 'error',
                'message': f'video {video_id} not found or not available for purchase'
            }

        if video_record.expected_reward_tao is None:
            raise HTTPException(500, detail="The video record is missing the expected reward tao, investigate this bug")
        
        # TODO: This is commented out because expected_reward_alpha is not filled in for all videos yet, need to migrate
        # if video_record.expected_reward_alpha is None:
        #     raise HTTPException(500, detail="The video record is missing the expected reward alpha, investigate this bug")

        # mark the purchase as pending i.e. a miner has claimed the video for purchase and now just needs to pay
        video_record.processing_state = FocusVideoStateInternal.PURCHASE_PENDING
        video_record.miner_hotkey = miner_hotkey
        video_record.updated_at = datetime.utcnow()

        # NOTE: we don't set the video_record.earned_reward_tao here, because we don't know if the
        # miner will successfully purchase the video or not. We set it later in cron/confirm_purchase.py

        db.add(video_record)
        db.commit()

        return {
            'status': 'success',
            'price': video_record.expected_reward_tao,
            'price_alpha': video_record.expected_reward_alpha,
        }

    except Exception as e:
        print(e)
        raise HTTPException(500, detail="Internal error")

def get_purchased_list(
    db: Session,
    miner_hotkey: str
):
    try:
        purchased_list = db.query(FocusVideoRecord).filter_by(
            processing_state=FocusVideoStateInternal.PURCHASED,
            miner_hotkey=miner_hotkey
        ).all()
        
        # result = [
        #     {
        #         "id": video.id,
        #         "task_id": video.task_id,
        #         "link": video.link,
        #         "score": video.score,
        #         "creator": video.creator,
        #         "miner_uid": video.miner_uid,
        #         "miner_hotkey": video.miner_hotkey,
        #         "estimated_tao": video.estimated_tao,
        #         "reward_tao": video.reward_tao,
        #         "status": video.status,
        #         "created_at": video.created_at,
        #         "task_str": video.task.focusing_task if video.task else None
        #     }
        #     for video in purchased_list
        # ]

        # FV TODO: again, what is this for????
        # for video in purchased_list:
        #     task = get_task(db, video.task_id)
        #     video.task_str = task.focusing_task
            
        return purchased_list
    except Exception as e:
        print(e)
        # raise HTTPException(500, detail="Internal error")
        return []

# def get_consumed_list(
#     db: Session,
#     miner_hotkey: str
# ):
#     try:
#         list = db.query(FocusVideoRecord).filter_by(
#             processing_state=FocusVideoStateInternal.CONSUMED,
#             miner_hotkey=miner_hotkey
#         ).all()
        
#         return list
#     except Exception as e:
#         print(e)
#         # raise HTTPException(500, detail="Internal error")
#         return []

async def check_video_metadata(
    db: Session,
    video_id: str,
    user_email: str,
    miner_hotkey: str
):
    try:
        video_info = db.query(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.user_email == user_email,
            FocusVideoRecord.miner_hotkey == miner_hotkey,
            FocusVideoRecord.deleted_at.is_(None)
        ).first()

        if video_info is not None and video_info.processing_state == FocusVideoStateInternal.PURCHASED:

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
            db.commit()

            # video_score = await score.score_video(task_info.focusing_task, task_info.clip_link)
            # print(f"Video score: {video_score}")
            video_score = video_info.video_score

            return {
                'success': True,
                'score': video_score
            }

        return {
            'success': False,
            'message': 'No video found.'
        }

    except Exception as e:
        print(e)
        return {
            'success': False,
            'message': 'Internal Server Errror'
        }

# async def consume_video(db: Session, video_ids: str):
#     print(f"Consuming focus video: <{video_ids}>")
#     try:
#         videos = db.query(FocusVideoRecord).filter(
#             FocusVideoRecord.video_id.in_(video_ids)
#         ).all()
#         if len(videos) > 0:
#             for video in videos:
#                 if video.processing_state == FocusVideoStateInternal.CONSUMED:
#                     return {
#                         'success': False,
#                         'message': 'Already consumed.'
#                     }
#                 video.processing_state = FocusVideoStateInternal.CONSUMED
#                 db.add(video)
#             db.commit()
#             return {
#                 'success': True
#             }
#         else:
#             return {
#                 'success': False,
#                 'message': 'No Video Found'
#             }
#     except Exception as e:
#         print(e)
#         return {
#             'success': False,
#             'message': 'Internal Server Error'
#         }

# def add_task_str(db:Session, video: any):
#     task = get_task(db, video.task_id)
#     video.task_str = task.focusing_task
#     return video

def get_video_owner_coldkey(db: Session, video_id: str) -> str:
    video_record = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.deleted_at.is_(None)
    )
    video_record = video_record.first()

    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    user_record = db.query(UserRecord).filter(UserRecord.email == video_record.user_email,).first()
    if user_record is None:
        raise HTTPException(404, detail="User not found")

    return user_record.coldkey

_already_purchased_cache = CachedValue()

async def _already_purchased_max_focus_tao() -> bool:
    with get_db_context() as db:
        effective_max_focus_alpha = await get_purchase_max_focus_alpha()
        effective_max_focus_tao = effective_max_focus_alpha * await alpha_to_tao_rate()
        total_earned_tao = db.query(func.sum(FocusVideoRecord.earned_reward_tao)).filter(
            FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASED,
            FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24)
        ).scalar() or 0
        print(total_earned_tao, effective_max_focus_tao)
        return total_earned_tao >= effective_max_focus_tao

async def already_purchased_max_focus_tao() -> bool:
    return await _already_purchased_cache.get_or_update(
        lambda: _already_purchased_max_focus_tao()
    )

_alpha_to_tao_cache = CachedValue()

async def _alpha_to_tao_rate() -> bool:
    sub = bittensor.subtensor(network=NETWORK)
    subnet = sub.subnet(NETUID)
    balance = subnet.alpha_to_tao(1)
    return balance.tao

async def alpha_to_tao_rate() -> float:
    return await _alpha_to_tao_cache.get_or_update(
        lambda: _alpha_to_tao_rate()
    )

# async def _already_purchased_max_focus_alpha() -> bool:
#     with get_db_context() as db:
#         effective_max_focus_alpha = await get_purchase_max_focus_alpha()
#         total_earned_alpha = db.query(func.sum(FocusVideoRecord.earned_reward_alpha)).filter(
#             FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASED,
#             FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24)
#         ).scalar() or 0
#         return total_earned_alpha >= effective_max_focus_alpha

# async def already_purchased_max_focus_alpha() -> bool:
#     return await _already_purchased_cache.get_or_update(
#         lambda: _already_purchased_max_focus_alpha()
#     )

class MinerPurchaseStats(BaseModel):
    purchased_videos: List[FocusVideoInternal]
    total_focus_points: float
    max_focus_points: float
    focus_points_percentage: float

async def get_miner_purchase_stats(db: Session, miner_hotkey: str) -> MinerPurchaseStats:
    # Get videos purchased by miner in the last 24 hours
    purchased_videos_records = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.miner_hotkey == miner_hotkey,
        FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASED,
        FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24)
    )
    purchased_videos_records = purchased_videos_records.all()
    
    purchased_videos = [
        FocusVideoInternal.model_validate(video_record)
        for video_record in purchased_videos_records
    ]

    # Calculate total score for purchased videos (focus points = score * 100)
    total_focus_points = sum(video.video_score * 100 for video in purchased_videos)

    # Calculate percentage
    # max_focus_tao = await get_max_focus_tao()
    # max_focus_points = get_max_focus_points_available_today(max_focus_tao)
    max_focus_alpha = await get_max_focus_alpha_per_day()
    max_focus_points = get_max_focus_points_available_today(max_focus_alpha)
    focus_points_percentage = total_focus_points / max_focus_points if max_focus_points > 0 else 0

    return MinerPurchaseStats(
        purchased_videos=purchased_videos,
        total_focus_points=total_focus_points,
        max_focus_points=max_focus_points,
        focus_points_percentage=focus_points_percentage
    )

def set_focus_video_score(db: Session, video_id: str, score_details: VideoScore, embeddings: FocusVideoEmbeddings):
    video_record = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.deleted_at.is_(None)
    ).first()
    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    video_record.video_score = score_details.final_score
    video_record.video_details = {
        **video_record.video_details,
        **json.loads(score_details.model_dump_json()),
    }
    video_record.embeddings = json.loads(embeddings.model_dump_json())
    video_record.processing_state = FocusVideoStateInternal.READY
    video_record.updated_at = datetime.utcnow()
    video_record.task_type = TaskType.BOOSTED if score_details.boosted_multiplier > 1.0 else TaskType.USER
    db.add(video_record)
    db.commit()

def mark_video_rejected(
    db: Session,
    video_id: str,
    rejection_reason: str,
    score_details: Optional[VideoScore]=None,
    embeddings: Optional[FocusVideoEmbeddings]=None,
    exception_string: Optional[str]=None,
):
    video_record = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.deleted_at.is_(None)
    ).first()
    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    video_details = { **video_record.video_details }

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

    video_record.processing_state = FocusVideoStateInternal.REJECTED
    video_record.rejection_reason = rejection_reason
    db.add(video_record)
    db.commit()

def mark_video_submitted(db: Session, video_id: str, with_lock: bool = False):
    # Mark video as "SUBMITTED" if in the "PURCHASE_PENDING" state.
    video_record = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING,
        FocusVideoRecord.deleted_at.is_(None)
    )
    if with_lock:
        video_record = video_record.with_for_update()
    video_record = video_record.first()
    
    if video_record is None:
        raise HTTPException(404, detail="Focus video not found or not in the correct state: PURCHASE_PENDING")

    video_record.processing_state = FocusVideoStateInternal.SUBMITTED
    video_record.updated_at = datetime.utcnow()
    db.add(video_record)
    db.commit()

_focus_points_cache = CachedValue(duration=60)  # Cache for 60 seconds

async def _fetch_focus_points(db: Session) -> Dict[TaskType, float]:
    results = db.query(
        FocusVideoRecord.task_type,
        func.sum(
            func.cast(FocusVideoRecord.video_details['duration'].astext, Float) * 
            FocusVideoRecord.video_score
        ).label('focus_points')
    ).filter(
        FocusVideoRecord.processing_state.in_([
            FocusVideoStateInternal.SUBMITTED,
            FocusVideoStateInternal.PURCHASED
        ]),
        FocusVideoRecord.created_at >= datetime.utcnow() - timedelta(hours=24)
    ).group_by(FocusVideoRecord.task_type).all()

    # Initialize dict with all TaskType values set to 0
    focus_points = {task_type: 0 for task_type in TaskType}

    # Update with actual results
    for task_type, points in results:
        focus_points[task_type] = points or 0

    return focus_points

async def get_focus_points_from_last_24_hours(db: Session) -> Dict[TaskType, float]:
    return await _focus_points_cache.get_or_update(
        lambda: _fetch_focus_points(db)
    )