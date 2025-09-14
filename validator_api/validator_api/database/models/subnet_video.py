from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class SubnetVideoRecord(Base):
    """
    Model for subnet videos that mirrors focus_videos structure
    """
    __tablename__ = "subnet_videos"

    video_id = Column(String(255), primary_key=True, index=True)
    video_score = Column(Float, nullable=False, default=0.8)
    expected_reward_tao = Column(Float, nullable=False)
    expected_reward_alpha = Column(Float, nullable=True)
    processing_state = Column(String(50), nullable=False, default='SUBMITTED', index=True)
    miner_hotkey = Column(String(255), nullable=True, index=True)
    earned_reward_tao = Column(Float, nullable=True)
    earned_reward_alpha = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    deleted_at = Column(DateTime, nullable=True, index=True)
    extrinsic_id = Column(String(255), nullable=True)
    block_hash = Column(String(255), nullable=True)