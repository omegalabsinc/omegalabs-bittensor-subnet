import asyncio
import validator_api.config as config
from fastapi import HTTPException
from sqlalchemy.orm import Session
from validator_api.database import models
from validator_api.utils.wallet import check_wallet_tao_balance
from datetime import datetime, timedelta

async def update_user_tao_balance_from_email(db: Session, email: str):
    user = db.query(models.User).filter_by(email=email).first()
    if user is None:
        raise HTTPException(422, detail="User with the given email does not exist.")
    return await update_user_tao_balance(db, user)

async def update_user_tao_balance(db: Session, user: models.User):
    finney_tao_balance, test_tao_balance = await asyncio.gather(
        check_wallet_tao_balance(user.coldkey, config.BT_MAINNET),
        check_wallet_tao_balance(user.coldkey, config.BT_TESTNET)
    )
    print(f"TAO balance: {finney_tao_balance}, Test TAO balance: {test_tao_balance}")
    user.tao_balance = finney_tao_balance
    user.test_tao_balance = test_tao_balance
    user.tao_check_time = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

async def check_user_tao_balances(db: Session, email: str) -> float:
    user = db.query(models.User).filter_by(email=email).first()
    if user is None:
        raise HTTPException(422, detail="User with the given email does not exist.")

    if user.tao_check_time is None or user.tao_check_time < datetime.utcnow() - timedelta(minutes=config.TAO_REFRESH_INTERVAL_MINUTES):
        user = await update_user_tao_balance(db, user)

    return user.tao_balance, user.test_tao_balance
