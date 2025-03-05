from sqlalchemy import Column, String, Boolean, Float, DateTime, Integer
from validator_api.config import DB_STRING_LENGTH
from validator_api.database import Base
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Optional


class TaskRecordPG(Base):
    __tablename__ = "tasks"
    id = Column(String(DB_STRING_LENGTH), primary_key=True, nullable=False)
    info = Column(String(DB_STRING_LENGTH))
    description = Column(String(DB_STRING_LENGTH))
    checked = Column(Boolean, default=False)
    date = Column(DateTime, default=datetime.utcnow)
    theme = Column(String(DB_STRING_LENGTH), nullable=True)
    score = Column(Float)
    user_id = Column(String(DB_STRING_LENGTH))
    chat_id = Column(String(DB_STRING_LENGTH), nullable=True)
    reason = Column(String(DB_STRING_LENGTH), nullable=True)
    boosted_id = Column(Integer, nullable=True)


class Task(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    info: str
    description: str
    checked: bool
    date: datetime
    theme: Optional[str]
    score: float
    user_id: str
    chat_id: Optional[str]
    reason: Optional[str]
    boosted_id: Optional[int]
