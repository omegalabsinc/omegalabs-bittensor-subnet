from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean
from validator_api.validator_api.database import Base
from datetime import datetime


class BoostedTask(Base):
    __tablename__ = "boosted_tasks"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    title = Column(String(1000), nullable=False)
    description = Column(String(1000), nullable=False)
    multiplier = Column(Float, nullable=False)
    active = Column(Boolean, nullable=False, default=True)
