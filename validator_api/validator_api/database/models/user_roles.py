from datetime import datetime
import enum
from sqlalchemy import Column, String, Enum, ForeignKey
from pydantic import BaseModel, ConfigDict

from validator_api.validator_api.database import Base
# from config import DB_STRING_LENGTH


class UserRoleEnum(enum.Enum):
    admin = "admin"
    trusted = "trusted"


class UserRoleRecordPG(Base):
    __tablename__ = 'user_roles'

    user_id = Column(String, ForeignKey('users.id'),
                     primary_key=True, nullable=False)
    role = Column(Enum(UserRoleEnum, name='user_role_enum', create_constraint=False), nullable=False)


class UserRole(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: str
    role: UserRoleEnum
