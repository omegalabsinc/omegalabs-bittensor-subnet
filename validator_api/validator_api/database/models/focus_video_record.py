from datetime import datetime
import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, String, DateTime, Float, Enum

from validator_api.validator_api.database import Base
from sqlalchemy.dialects.postgresql import JSONB
from validator_api.validator_api.config import DB_STRING_LENGTH

import enum


class TaskType(enum.Enum):
    USER = "USER"
    BOOSTED = "BOOSTED"
    MARKETPLACE = "MARKETPLACE"


class FocusVideoStateExternal(enum.Enum):
    PROCESSING = "PROCESSING"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    READY = "READY"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    REWARDED = "REWARDED"


class FocusVideoStateInternal(enum.Enum):
    # OMEGA Focus user facing states
    IN_PROGRESS = "IN_PROGRESS"
    PROCESSING = "PROCESSING"  # User has completed task, we are currently calculating their score and checking if the video is legit
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"  # Video belongs to a marketplace task and needs human review
    READY = "READY"  # Score has been calculated and task is eligible for submission
    REJECTED = "REJECTED"  # Turns out that the task was NOT eligible for submission, lifecycle ended here
    SUBMITTED = "SUBMITTED"  # User has pressed "Submit" and the task is now listed on the marketplace, for SN24 miners to buy

    # Miner purchase states
    PURCHASE_PENDING = "PURCHASE_PENDING"  # a miner has request to buy the video, and we have sent them the amount of tao that they need to send the focus user
    PURCHASED = "PURCHASED"  # our background cron has confirmed that the miner has bought the focus video


def map_focus_video_state(state: FocusVideoStateInternal) -> FocusVideoStateExternal:
    """
    The first 4 states are the ones that the user sees. The last 4 states are the ones that the
    miner sees. All the user needs to know is whether the video has been purchased by a miner.
    """
    state_mapping = {
        FocusVideoStateInternal.IN_PROGRESS: FocusVideoStateExternal.PROCESSING,
        FocusVideoStateInternal.PROCESSING: FocusVideoStateExternal.PROCESSING,
        FocusVideoStateInternal.PENDING_HUMAN_REVIEW: FocusVideoStateExternal.PENDING_HUMAN_REVIEW,
        FocusVideoStateInternal.READY: FocusVideoStateExternal.READY,
        FocusVideoStateInternal.REJECTED: FocusVideoStateExternal.REJECTED,
        FocusVideoStateInternal.SUBMITTED: FocusVideoStateExternal.SUBMITTED,
        FocusVideoStateInternal.PURCHASE_PENDING: FocusVideoStateExternal.SUBMITTED,
        FocusVideoStateInternal.PURCHASED: FocusVideoStateExternal.REWARDED,
        # FocusVideoStateInternal.VALIDATING: FocusVideoStateExternal.REWARDED,
        # FocusVideoStateInternal.CONSUMED: FocusVideoStateExternal.REWARDED,
    }
    if state in state_mapping:
        return state_mapping[state]
    else:
        raise ValueError(f"Invalid focus video state: {state}")


class FocusVideoRecord(Base):
    __tablename__ = "focus_videos"

    video_id = Column(
        String(DB_STRING_LENGTH),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
    )
    task_id = Column(String(DB_STRING_LENGTH), nullable=False)
    user_id = Column(String, nullable=False)
    user_email = Column(String, nullable=False)
    processing_state = Column(
        Enum(
            *FocusVideoStateInternal.__members__,
            name="focus_videos_processing_state",
            schema="public",
        ),
        nullable=False,
        default=FocusVideoStateInternal.PROCESSING,
    )
    task_type = Column(
        Enum(*TaskType.__members__, name="focus_videos_task_type", schema="public"),
        nullable=False,
        default=TaskType.USER,
    )
    video_score = Column(Float, nullable=True)
    video_details = Column(JSONB, nullable=True)
    embeddings = Column(JSONB, nullable=True)
    rejection_reason = Column(String(1000), nullable=True)
    expected_reward_tao = Column(Float, nullable=True)
    expected_reward_alpha = Column(Float, nullable=True)
    earned_reward_tao = Column(Float, nullable=True)
    earned_reward_alpha = Column(Float, nullable=True)
    miner_hotkey = Column(String(DB_STRING_LENGTH), nullable=True)
    extrinsic_id = Column(String(DB_STRING_LENGTH), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)
    rewarded_at = Column(DateTime, nullable=True)
    trajectories = Column(JSONB, nullable=True)

    def get_duration(self) -> float:
        return float(self.video_details.get("duration", 0.0))


class FocusVideoBase(BaseModel):
    video_id: str
    task_id: str
    user_email: str
    task_type: TaskType
    video_score: Optional[float]
    rejection_reason: Optional[str]
    expected_reward_tao: Optional[float]
    earned_reward_tao: Optional[float]
    expected_reward_alpha: Optional[float]
    earned_reward_alpha: Optional[float]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]


class FocusVideoInternal(FocusVideoBase):
    model_config = ConfigDict(from_attributes=True)

    processing_state: FocusVideoStateInternal
    miner_hotkey: Optional[str]
