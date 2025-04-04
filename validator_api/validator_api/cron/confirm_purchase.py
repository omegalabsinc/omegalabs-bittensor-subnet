import asyncio
from datetime import datetime

import bittensor as bt
import validator_api.validator_api.config as config
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from validator_api.validator_api.database import get_db_context
from validator_api.validator_api.database.models.focus_video_record import (
    FocusVideoRecord,
    FocusVideoStateInternal,
)
from validator_api.validator_api.database.models.miner_bans import (
    increment_failed_purchases,
    reset_failed_purchases,
)
from validator_api.validator_api.utils.wallet import get_transaction_from_block_hash


async def extrinsic_already_confirmed(db: AsyncSession, extrinsic_id: str) -> bool:
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.extrinsic_id == extrinsic_id
    )
    result = await db.execute(query)
    return result.scalar_one_or_none() is not None


async def check_payment(
    db: AsyncSession,
    recipient_address: str,
    sender_address: str,
    amount: float,
    block_hash: str = None,
):
    try:
        print(
            f"Checking payment of {amount} from {sender_address} to {recipient_address}"
        )

        # Get all transfers associated with the recipient address
        transfers = await get_transaction_from_block_hash(recipient_address, block_hash)

        # Filter transfers to find the specific payment
        for transfer in transfers:
            if (
                transfer["from"] == sender_address
                and transfer["to"] == recipient_address
                and round(float(transfer["amount"]), 5) == round(amount, 5)
            ):
                if await extrinsic_already_confirmed(db, transfer["extrinsicId"]):
                    continue
                print(
                    f"Payment of {amount} found from {sender_address} to {recipient_address}"
                )
                return transfer["extrinsicId"]

        print(
            f"Payment of {amount} not found from {sender_address} to {recipient_address}"
        )
        return None

    except Exception as e:
        print(f"Error in checking payment: {e}")
        return None

    # finally:
    #     sub.close()


SUBTENSOR_RETRIES = 5
SUBTENSOR_DELAY_SECS = 2


async def confirm_transfer(
    db: AsyncSession,
    video_owner_coldkey: str,
    video_id: str,
    miner_hotkey: str,
    block_hash: str = None,
    with_lock: bool = False,
):
    subtensor = bt.subtensor(network=config.NETWORK)
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.processing_state
        == FocusVideoStateInternal.PURCHASE_PENDING.value,
        FocusVideoRecord.miner_hotkey == miner_hotkey,
        FocusVideoRecord.deleted_at.is_(None),
    )
    if with_lock:
        query = query.with_for_update()

    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if not video:
        print(
            f"confirm_transfer | video <{video_id}> not found or not in PURCHASE_PENDING state"
        )
        return False

    tao_amount = video.expected_reward_tao

    current_time = datetime.utcnow()
    print(
        f"[{current_time}] | Scanning block hash <{block_hash}> for address <{video_owner_coldkey}> payment transaction from  ..."
    )
    for attempt in range(SUBTENSOR_RETRIES):
        try:
            miner_coldkey = subtensor.get_hotkey_owner(miner_hotkey)
            print(f"Miner coldkey: {miner_coldkey}")

            extrinsic_id = await check_payment(
                db, video_owner_coldkey, miner_coldkey, tao_amount, block_hash
            )
            if extrinsic_id is not None:
                print(
                    f"Miner <{miner_hotkey}> successfully purchased focus recording <{video_id}>!"
                )
                video.miner_hotkey = miner_hotkey
                video.processing_state = FocusVideoStateInternal.PURCHASED.value
                video.updated_at = datetime.utcnow()
                video.extrinsic_id = extrinsic_id
                video.earned_reward_tao = tao_amount
                video.rewarded_at = datetime.utcnow()
                # TODO: this is only theoretical, actually do this properly by setting it when the specific earned tao amount is actually staked via OFB
                video.earned_reward_alpha = video.expected_reward_alpha
                db.add(video)
                await db.commit()
                return True

        except Exception as e:
            if attempt < SUBTENSOR_RETRIES - 1:  # if it's not the last attempt
                if (
                    "Broken pipe" in str(e)
                    or "EOF occurred in violation of protocol" in str(e)
                    or "[SSL: BAD_LENGTH]" in str(e)
                ):
                    print(
                        f"Connection to subtensor was lost. Re-initializing subtensor and retrying in {SUBTENSOR_DELAY_SECS} seconds..."
                    )
                    subtensor = bt.subtensor(network=config.NETWORK)
                    await asyncio.sleep(SUBTENSOR_DELAY_SECS)
                else:
                    print(
                        f"Attempt #{attempt + 1} to sub.get_hotkey_owner() and check_payment() failed. Retrying in {SUBTENSOR_DELAY_SECS} seconds..."
                    )
                    print(f"Error: {str(e)}")
                    await asyncio.sleep(SUBTENSOR_DELAY_SECS)
            else:
                print(
                    f"All {SUBTENSOR_RETRIES} attempts failed. Unable to retrieve miner coldkey and confirm payment."
                )
                print(f"Final error: {str(e)}")
                return False
    # we got here because we could not confirm the payment. Let's return false to let the miner know
    return False


DELAY_SECS = 30  # 30s
RETRIES = 6  # 30s x 10 retries = 180s = 3 mins


async def confirm_video_purchased(video_id: str, with_lock: bool = False):
    """
    The purpose of this function is to set the video back to the SUBMITTED state
    if the miner has not confirmed the purchase in time.
    """

    current_time = datetime.utcnow()
    print(
        f"BACKGROUND TASK | {current_time} | Checking if video_id <{video_id}> has been marked as purchased or reverted back to SUBMITTED ..."
    )
    try:
        for i in range(0, RETRIES):
            await asyncio.sleep(DELAY_SECS)
            try:
                async with get_db_context() as db:
                    query = select(FocusVideoRecord).filter(
                        FocusVideoRecord.video_id == video_id,
                        FocusVideoRecord.deleted_at.is_(None),
                    )
                    if with_lock:
                        query = query.with_for_update()

                    result = await db.execute(query)
                    video = result.scalar_one_or_none()

                    if not video:
                        print(f"Video <{video_id}> not found")
                        return False

                    if (
                        video is not None
                        and video.processing_state
                        == FocusVideoStateInternal.PURCHASED.value
                    ):
                        print(
                            f"Video <{video_id}> has been marked as PURCHASED. Stopping background task."
                        )
                        await reset_failed_purchases(db, video.miner_hotkey)
                        return True
                    elif (
                        video is not None
                        and video.processing_state
                        == FocusVideoStateInternal.SUBMITTED.value
                    ):
                        print(
                            f"Video <{video_id}> has been marked as SUBMITTED. Stopping background task."
                        )
                        return True

                    print(
                        f"Video <{video_id}> has NOT been marked as PURCHASED. Retrying in {DELAY_SECS} seconds..."
                    )
                    # close the db connection until next retry
                    await db.close()

            except Exception as e:
                print(f"Error in checking confirm_video_purchased loop: {e}")

        # we got here because we could not confirm the payment in time, so we need to revert
        # the video back to the SUBMITTED state (i.e. mark available for purchase)
        print(
            f"Video <{video_id}> has NOT been marked as PURCHASED. Reverting to SUBMITTED state..."
        )
        await increment_failed_purchases(db, video.miner_hotkey)
        video.processing_state = FocusVideoStateInternal.SUBMITTED.value
        video.updated_at = datetime.utcnow()
        db.add(video)
        await db.commit()
        await db.close()
        return False

    except Exception as e:
        print(f"Error in confirm_video_purchased: {e}")

    return False
