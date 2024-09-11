import bcrypt
import uuid
from datetime import datetime
from sqlalchemy import inspect
from sqlalchemy import Column, String, Float, Boolean, DateTime, Integer, Enum, JSON, ForeignKey
from validator_api.database.schemas import TaskStatusEnum, FocusVideoEnum

from validator_api.database import Base

class User(Base):
    __tablename__ = 'users'

    email = Column(String, primary_key=True, nullable=False)
    password = Column(String, nullable=False)
    email_verified = Column(Boolean)
    nick_name = Column(String)
    name = Column(String)
    avatar = Column(String)
    google_sso = Column(Boolean)
    coldkey = Column(String)
    hotkey = Column(String)
    cold_seed = Column(String)
    hot_seed = Column(String)
    balance = Column(Float)
    coin_balance = Column(Float)
    tao_balance = Column(Float)
    test_tao_balance = Column(Float)
    tao_check_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime)

    def __init__(
        self,
        email: str, 
        password: str,
        nick_name: str,
        coldkey: str,
        hotkey: str,
        cold_seed: str,
        hot_seed: str,
        email_verified: bool = False,
        name: str = None,
        avatar: str = None,
        google_sso: bool = False,
        balance: float = 0.0,
        coin_balance: float =0.0,
        tao_balance: float = 0.0,
        test_tao_balance: float = 0.0,
    ):
        self.email = email
        self.password = self.set_password(password)
        self.nick_name = nick_name
        self.coldkey = coldkey
        self.hotkey = hotkey
        self.cold_seed = cold_seed
        self.hot_seed = hot_seed
        self.balance = balance
        self.coin_balance = coin_balance
        self.tao_balance = tao_balance
        self.test_tao_balance = test_tao_balance
        self.email_verified = email_verified
        self.name = name
        self.avatar = avatar
        self.google_sso = google_sso
        self.created_at = datetime.now()
        self.tao_check_time = None

    def verify_password(self, password: str):
        return bcrypt.checkpw(password.encode('utf-8') , self.password.encode('utf-8'))
    
    def set_password(self, password: str):
        salt = bcrypt.gensalt()
        hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hash.decode('utf-8')
    
    def as_dict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }
    
class Task(Base):
    __tablename__ = 'task'

    id = Column(String, primary_key=True, nullable=False)
    focusing_task = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    user_email = Column(String, nullable=False)
    clip_link = Column(String)
    description = Column(String)
    date = Column(String, default=datetime.now())
    checked = Column(Boolean, default=False)
    score = Column(Float)
    status = Column(Enum(TaskStatusEnum), default=TaskStatusEnum.Ready)
    active = Column(Boolean, default=True)
    event = Column(JSON)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    def __init__(
        self, 
        focusing_task: String,
        duration: Float,
        user_email: String,
        score: Float
    ):
        self.id = str(uuid.uuid4())
        self.focusing_task = focusing_task
        self.duration = duration
        self.user_email = user_email
        self.score = score
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def as_dict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }

class IpfsUrl(Base):
    __tablename__ = 'ipfsurl'

    id = Column(String, primary_key=True, nullable=False)
    url = Column(String, nullable=False)
    miner_hotkey = Column(String, nullable=False)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    def __init__(
        self, 
        url: String,
        miner_hotkey: String,
    ):
        self.id = str(uuid.uuid4())
        self.url = url
        self.miner_hotkey = miner_hotkey
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def as_dict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }

class Log(Base):
    __tablename__ = 'log'

    id = Column(String, primary_key=True, nullable=False)
    task_id = Column(String, nullable=False)
    ss_link = Column(String)
    ss_topic = Column(String)
    score = Column(Float)
    is_focused = Column(Boolean)
    created_at = Column(DateTime)

    def __init__(
        self, 
        task_id: String,
        ss_link: String,
        ss_topic: String,
        score: Float,
        is_focused: Boolean
    ):
        self.id = str(uuid.uuid4())
        self.task_id = task_id
        self.ss_link = ss_link
        self.ss_topic = ss_topic
        self.score = score
        self.is_focused = is_focused
        self.created_at = datetime.now()

    def as_dict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }

class FocusVideo(Base):
    __tablename__ = 'focusvideo'

    id = Column(String, primary_key=True, nullable=False)
    task_id = Column(String, nullable=False)
    link = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    creator = Column(String, nullable=False)
    miner_uid = Column(String)
    miner_hotkey = Column(String)
    estimated_tao = Column(Float)
    reward_tao = Column(Float)
    status = Column(Enum(FocusVideoEnum), default=FocusVideoEnum.Uploaded)
    created_at = Column(DateTime)

    def __init__(
        self, 
        task_id: String,
        link: String,
        score: Float,
        creator: String,
        estimated_tao: float=None,
    ):
        self.id = str(uuid.uuid4())
        self.task_id = task_id
        self.link = link
        self.score = score
        self.creator = creator
        self.estimated_tao = estimated_tao
        self.created_at = datetime.now()

    def as_dict(self):
        return { c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs }
