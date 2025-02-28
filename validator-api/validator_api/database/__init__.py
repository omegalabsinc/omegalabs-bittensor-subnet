from validator_api import config
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.schema import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker
from contextlib import asynccontextmanager

DB_HOST = config.FOCUS_DB_HOST
DB_NAME = config.FOCUS_DB_NAME
DB_USER = config.FOCUS_DB_USER
DB_PASSWORD = config.FOCUS_DB_PASSWORD
DB_PORT = config.FOCUS_DB_PORT
DB_POOL_SIZE = config.FOCUS_DB_POOL_SIZE
DB_MAX_OVERFLOW = config.FOCUS_DB_MAX_OVERFLOW

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=15,  # bumped down from default of 30
    pool_pre_ping=True,  # Good practice for most scenarios
    pool_recycle=300,  # Recycle connections after 5 minutes
)
SessionLocal = async_sessionmaker(class_=AsyncSession, autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()

@asynccontextmanager
async def get_db_context():
    async for db in get_db():
        yield db
