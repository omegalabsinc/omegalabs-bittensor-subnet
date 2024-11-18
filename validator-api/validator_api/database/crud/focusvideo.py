from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Optional
import json
import time
import asyncio

from validator_api.database.models.focus_video_record import FocusVideoRecord, FocusVideoInternal, FocusVideoStateInternal
from validator_api.database.models.user import UserRecord
from validator_api.utils.marketplace import estimate_tao, get_max_focus_tao, get_max_focus_points_available_today
from pydantic import BaseModel
from validator_api.services.scoring_service import VideoScore

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
    items = db.query(FocusVideoRecord).filter_by(
        processing_state=FocusVideoStateInternal.SUBMITTED,
        deleted_at=None,
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
        max_focus_tao = await get_max_focus_tao()

        video_record = db.query(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None),
            FocusVideoRecord.processing_state == FocusVideoStateInternal.SUBMITTED,  # is available for purchase
        )
        if with_lock:
            video_record = video_record.with_for_update()
        video_record = video_record.first()

        if video_record is None:
            return {
                'status': 'error',
                'message': f'video {video_id} not found or not available for purchase'
            }

        actual_reward_tao = estimate_tao(video_record.video_score, video_record.get_duration(), max_focus_tao)
        print(f"Expected reward TAO: {video_record.expected_reward_tao}, actual reward TAO: {actual_reward_tao}")
        if actual_reward_tao == 0:
            raise HTTPException(422, detail="Max reward TAO is 0")

        # mark the purchase as pending i.e. a miner has claimed the video for purchase and now just needs to pay
        video_record.processing_state = FocusVideoStateInternal.PURCHASE_PENDING
        video_record.miner_hotkey = miner_hotkey
        video_record.updated_at = datetime.utcnow()
        # set the actual reward TAO as what is expected to be paid and needs to from purchasing miner
        video_record.expected_reward_tao = actual_reward_tao

        # NOTE: we don't set the video_record.earned_reward_tao here, because we don't know if the
        # miner will successfully purchase the video or not. We set it later in cron/confirm_purchase.py

        db.add(video_record)
        db.commit()

        return {
            'status': 'success',
            'price': actual_reward_tao
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

async def _already_purchased_max_focus_tao(db: Session) -> bool:
    max_focus_tao = await get_max_focus_tao()
    total_earned_tao = db.query(func.sum(FocusVideoRecord.earned_reward_tao)).filter(
        FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASED,
        FocusVideoRecord.updated_at >= datetime.utcnow() - timedelta(hours=24)
    ).scalar() or 0
    return total_earned_tao >= max_focus_tao * 0.9

async def already_purchased_max_focus_tao(db: Session) -> bool:
    return await _already_purchased_cache.get_or_update(
        lambda: _already_purchased_max_focus_tao(db)
    )

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
    max_focus_tao = await get_max_focus_tao()
    max_focus_points = get_max_focus_points_available_today(max_focus_tao)
    focus_points_percentage = total_focus_points / max_focus_points if max_focus_points > 0 else 0

    return MinerPurchaseStats(
        purchased_videos=purchased_videos,
        total_focus_points=total_focus_points,
        max_focus_points=max_focus_points,
        focus_points_percentage=focus_points_percentage
    )

def set_focus_video_score(db: Session, video_id: str, score_details: VideoScore):
    video_record = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.deleted_at.is_(None)
    ).first()
    if video_record is None:
        raise HTTPException(404, detail="Focus video not found")

    video_record.video_score = score_details.combined_score
    video_record.video_details = {
        **video_record.video_details,
        **json.loads(score_details.model_dump_json()),
    }
    video_record.processing_state = FocusVideoStateInternal.READY
    db.add(video_record)
    db.commit()

def mark_video_rejected(
    db: Session,
    video_id: str,
    rejection_reason: str,
    score_details: Optional[VideoScore]=None,
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
