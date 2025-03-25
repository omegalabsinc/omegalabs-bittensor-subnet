from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime
from pydantic import BaseModel

from validator_api.validator_api.config import DB_STRING_LENGTH
from validator_api.validator_api.database import Base


class UserRecord(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, nullable=False)
    email = Column(String(DB_STRING_LENGTH), primary_key=True, nullable=False)
    name = Column(String(DB_STRING_LENGTH))
    coldkey = Column(String(DB_STRING_LENGTH))
    hotkey = Column(String(DB_STRING_LENGTH))
    tao_balance = Column(Float)
    tao_check_time = Column(DateTime, nullable=True)
    focused_task_id = Column(String(DB_STRING_LENGTH), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(BaseModel):
    id: str
    email: str
    name: str
    tao_balance: float
    tao_check_time: datetime
    focused_task_id: str
    created_at: datetime


class UserInternal(BaseModel):
    coldkey: str
    hotkey: str
