import asyncio
import time
from datetime import datetime
import validator_api.config as config

from sqlalchemy.orm import Session

from validator_api.database.models import User
from validator_api.database.models.focus_video_record import FocusVideoRecord, FocusVideoStateInternal
from validator_api.database.crud.user import update_user_tao_balance_from_email

import bittensor as bt

from validator_api.utils.wallet import get_transaction_from_block_hash

def extrinsic_already_confirmed(db: Session, extrinsic_id: str) -> bool:
    record = db.query(FocusVideoRecord).filter(FocusVideoRecord.extrinsic_id == extrinsic_id).first()
    return record is not None

async def check_payment(db: Session, recipient_address: str, sender_address: str, amount: float, block_hash: str = None):
    try:
        print(f"Checking payment of {amount} from {sender_address} to {recipient_address}")

        sub = bt.subtensor(network=config.NETWORK)

        # Get all transfers associated with the recipient address
        transfers = await get_transaction_from_block_hash(sub, recipient_address, block_hash)

        # Filter transfers to find the specific payment
        for transfer in transfers:
            if (
                transfer["from"] == sender_address and
                transfer["to"] == recipient_address and
                round(float(transfer["amount"]), 5) == round(amount, 5)
            ):
                if extrinsic_already_confirmed(db, transfer["extrinsicId"]):
                    continue
                print(f"Payment of {amount} found from {sender_address} to {recipient_address}")
                return transfer["extrinsicId"]

        print(f"Payment of {amount} not found from {sender_address} to {recipient_address}")
        return None

    except Exception as e:
        print(f'Error in checking payment: {e}')
        return None

    finally:
        sub.close()

SUBTENSOR_RETRIES = 5
SUBTENSOR_DELAY_SECS = 2

async def confirm_transfer(
    db: Session,
    video_owner_coldkey: str,
    video_id: str,
    miner_hotkey: str,
    block_hash: str = None
):
    subtensor = bt.subtensor(network=config.NETWORK)

    video = db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.processing_state == FocusVideoStateInternal.PURCHASE_PENDING,
        FocusVideoRecord.deleted_at.is_(None),
    ).first()

    if not video:
        print(f"Video <{video_id}> not found")
        return False
    
    amount = video.expected_reward_tao

    current_time = datetime.utcnow()
    print(f"[{current_time}] | Scanning block hash <{block_hash}> for address <{video_owner_coldkey}> payment transaction from  ...")    
    for attempt in range(SUBTENSOR_RETRIES):
        try:
            miner_coldkey = subtensor.get_hotkey_owner(miner_hotkey)
            print(f"Miner coldkey: {miner_coldkey}")
            
            extrinsic_id = await check_payment(db, video_owner_coldkey, miner_coldkey, amount, block_hash)
            if extrinsic_id is not None:
                print(f"Miner <{miner_hotkey}> successfully purchased focus recording <{video_id}>!")
                video.miner_hotkey = miner_hotkey
                video.processing_state = FocusVideoStateInternal.PURCHASED
                video.updated_at = datetime.utcnow()
                video.extrinsic_id = extrinsic_id
                video.earned_reward_tao = amount
                db.add(video)
                db.commit()
                try:
                    await update_user_tao_balance_from_email(db, video.user_email)
                except Exception as e:
                    print(f"Error in updating user tao balance: {e}")
                return True

        except Exception as e:
            if attempt < SUBTENSOR_RETRIES - 1:  # if it's not the last attempt
                if "Broken pipe" in str(e) or "EOF occurred in violation of protocol" in str(e) or "[SSL: BAD_LENGTH]" in str(e):
                    print(f"Connection to subtensor was lost. Re-initializing subtensor and retrying in {SUBTENSOR_DELAY_SECS} seconds...")
                    subtensor = bt.subtensor(network=config.NETWORK)
                    time.sleep(SUBTENSOR_DELAY_SECS)
                else:
                    print(f"Attempt #{attempt + 1} to sub.get_hotkey_owner() and check_payment() failed. Retrying in {SUBTENSOR_DELAY_SECS} seconds...")
                    print(f"Error: {str(e)}")
                    time.sleep(SUBTENSOR_DELAY_SECS)
            else:
                print(f"All {SUBTENSOR_RETRIES} attempts failed. Unable to retrieve miner coldkey and confirm payment.")
                print(f"Final error: {str(e)}")
                return False
    # we got here because we could not confirm the payment. Let's return false to let the miner know
    return False


DELAY_SECS = 30  # 30s
RETRIES = 10  # 30s x 10 retries = 300s = 5 mins

async def confirm_video_purchased(
    db: Session,
    video_id: str,
):
    """
    The purpose of this function is to set the video back to the SUBMITTED state 
    if the miner has not confirmed the purchase in time.
    """
    current_time = datetime.utcnow()
    print(f"BACKGROUND TASK | {current_time} | Checking if video_id <{video_id}> has been marked as purchased ...")
    try:
        for i in range(0, RETRIES):
            try:
                await asyncio.sleep(DELAY_SECS)
                video = db.query(FocusVideoRecord).filter(
                    FocusVideoRecord.video_id == video_id,
                    FocusVideoRecord.deleted_at.is_(None),
                ).first()
                if video is not None and video.processing_state == FocusVideoStateInternal.PURCHASED:
                    print(f"Video <{video_id}> has been marked as PURCHASED.")
                    return True

                print(f"Video <{video_id}> has NOT been marked as PURCHASED. Retrying in {DELAY_SECS} seconds...")

            except Exception as e:
                print(f"Error in checking confirm_video_purchased loop: {e}")

        # we got here because we could not confirm the payment in time, so we need to revert
        # the video back to the SUBMITTED state (i.e. mark available for purchase)
        print(f"Video <{video_id}> has NOT been marked as PURCHASED. Reverting to SUBMITTED state...")
        video.processing_state = FocusVideoStateInternal.SUBMITTED
        video.updated_at = datetime.utcnow()
        db.add(video)
        db.commit()
        return False

    except Exception as e:
        print(f"Error in confirm_video_purchased: {e}")

    return False
