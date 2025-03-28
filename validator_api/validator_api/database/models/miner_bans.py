from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, DateTime, Integer, select

from validator_api.validator_api.database import Base
from validator_api.validator_api.config import DB_STRING_LENGTH
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta


class MinerBan(Base):
    __tablename__ = "miner_bans"

    miner_hotkey = Column(String(DB_STRING_LENGTH), primary_key=True, nullable=False)
    purchases_failed_in_a_row = Column(Integer, nullable=False)
    banned_until = Column(DateTime(timezone=True), nullable=True)


async def miner_banned_until(db: AsyncSession, miner_hotkey: str) -> Optional[datetime]:
    """
    Check if a miner is currently banned and return their ban expiration time if they are.

    Args:
        db: Database session
        miner_hotkey: The miner's hotkey to check

    Returns:
        datetime: The banned_until time if the miner is currently banned
        None: If the miner is not currently banned
    """
    query = select(MinerBan).filter(
        MinerBan.miner_hotkey == miner_hotkey, MinerBan.banned_until > datetime.utcnow()
    )
    result = await db.execute(query)
    ban = result.scalar_one_or_none()

    return ban.banned_until if ban else None


async def get_or_create_miner(db: AsyncSession, miner_hotkey: str) -> MinerBan:
    """
    Get a miner's ban record or create it if it doesn't exist.

    Args:
        db: Database session
        miner_hotkey: The miner's hotkey

    Returns:
        MinerBan: The miner's ban record
    """
    query = select(MinerBan).filter(MinerBan.miner_hotkey == miner_hotkey)
    result = await db.execute(query)
    miner = result.scalar_one_or_none()

    if not miner:
        miner = MinerBan(
            miner_hotkey=miner_hotkey, purchases_failed_in_a_row=0, banned_until=None
        )
        db.add(miner)
        await db.commit()

    return miner


async def increment_failed_purchases(db: AsyncSession, miner_hotkey: str):
    """
    Increment the number of purchases failed in a row for a miner.
    Creates the miner record if it doesn't exist.

    """
    miner = await get_or_create_miner(db, miner_hotkey)
    miner.purchases_failed_in_a_row += 1
    print(
        f"increment_failed_purchases | miner_hotkey <{miner_hotkey}> purchases_failed_in_a_row <{miner.purchases_failed_in_a_row}>"
    )
    check_and_ban_miner(miner)
    await db.commit()


async def reset_failed_purchases(db: AsyncSession, miner_hotkey: str):
    """
    In the case of a successful purchase, reset the number of purchases failed in a row for a miner.
    Creates the miner record if it doesn't exist.
    """
    miner = await get_or_create_miner(db, miner_hotkey)
    miner.purchases_failed_in_a_row = 0
    miner.banned_until = None
    await db.commit()


BAN_PURCHASES_FAILED_IN_A_ROW = 5


def check_and_ban_miner(miner: MinerBan):
    """
    If a miner fails more than BAN_PURCHASES_FAILED_IN_A_ROW purchases in a row, ban them for 24 hours.
    Creates the miner record if it doesn't exist.
    """
    if miner.purchases_failed_in_a_row >= BAN_PURCHASES_FAILED_IN_A_ROW:
        miner.purchases_failed_in_a_row = 0
        miner.banned_until = datetime.utcnow() + timedelta(hours=24)
