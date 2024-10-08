from validator_api import config
from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

DB_HOST = config.FOCUS_DB_HOST
DB_NAME = config.FOCUS_DB_NAME
DB_USER = config.FOCUS_DB_USER
DB_PASSWORD = config.FOCUS_DB_PASSWORD

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # bumped up from default of 5
    max_overflow=20,  # bumped up from default of 10
    pool_timeout=15,  # bumped down from default of 30
    pool_pre_ping=True,  # Good practice for most scenarios
    pool_recycle=3600,  # Recycle connections after 1 hour
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_context():
    return contextmanager(get_db)()
