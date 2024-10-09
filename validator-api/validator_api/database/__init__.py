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
DB_PORT = config.FOCUS_DB_PORT

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
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
