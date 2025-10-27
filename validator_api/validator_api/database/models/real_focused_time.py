from datetime import datetime

from sqlalchemy import Column, String, BigInteger, DateTime, ForeignKey

from validator_api.validator_api.database import Base


class RealFocusedTime(Base):
    __tablename__ = "real_focused_time"

    id = Column(String, primary_key=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    real_focused_duration = Column(BigInteger, default=0, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    video_id = Column(
        String,
        ForeignKey("focus_videos.video_id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(String, ForeignKey("users.id"), nullable=True)