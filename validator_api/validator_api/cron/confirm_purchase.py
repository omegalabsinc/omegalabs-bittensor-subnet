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
from validator_api.validator_api.database.models.subnet_video import SubnetVideoRecord
from validator_api.validator_api.database.models.miner_bans import (
    increment_failed_purchases,
    reset_failed_purchases,
)
from validator_api.validator_api.utils.wallet import get_transaction_from_block_hash
from validator_api.validator_api.database.models.user import UserRecord
from validator_api.validator_api.database.models.focus_video_record import TaskType
from validator_api.validator_api.database.crud.focusvideo import release_video_lock


async def extrinsic_already_confirmed(db: AsyncSession, extrinsic_id: str) -> bool:
    query = select(FocusVideoRecord).filter(
        FocusVideoRecord.extrinsic_id == extrinsic_id
    )
    result = await db.execute(query)
    if result.scalar_one_or_none() is not None:
        return True
    
    # Also check subnet videos
    query = select(SubnetVideoRecord).filter(
        SubnetVideoRecord.extrinsic_id == extrinsic_id
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
            f"\n=== CHECK PAYMENT ==="
        )
        print(
            f"Looking for payment: {amount} TAO"
        )
        print(f"From (sender): {sender_address}")
        print(f"To (recipient): {recipient_address}")
        print(f"In block hash: {block_hash}")

        # Get all transfers associated with the recipient address
        transfers = await get_transaction_from_block_hash(recipient_address, block_hash)
        
        print(f"\nFound {len(transfers)} transfers from block")

        # Filter transfers to find the specific payment
        for i, transfer in enumerate(transfers):
            print(f"\nChecking transfer #{i+1}:")
            print(f"  From: {transfer['from']}")
            print(f"  To: {transfer['to']}")
            print(f"  Amount: {transfer['amount']} TAO")
            print(f"  Expected amount: {amount} TAO")
            
            # Check if addresses match
            from_match = transfer["from"] == sender_address
            to_match = transfer["to"] == recipient_address
            amount_match = round(float(transfer["amount"]), 5) == round(amount, 5)
            
            print(f"  From matches: {from_match}")
            print(f"  To matches: {to_match}")
            print(f"  Amount matches: {amount_match}")
            
            if from_match and to_match and amount_match:
                if await extrinsic_already_confirmed(db, transfer["extrinsicId"]):
                    print(f"  >>> Extrinsic already confirmed, skipping")
                    continue
                print(
                    f"  >>> PAYMENT FOUND! Extrinsic ID: {transfer['extrinsicId']}"
                )
                return transfer["extrinsicId"]
            else:
                print(f"  >>> Not a match")

        print(
            f"\n=== PAYMENT NOT FOUND ==="
        )
        print(f"Expected: {amount} TAO from {sender_address} to {recipient_address}")
        print(f"Block hash searched: {block_hash}")
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

    # Check if this is a subnet video
    # Accept both PURCHASE_PENDING and SUBMITTED states (in case background task already reverted)
    # The miner_hotkey filter ensures only the miner who initiated the purchase can verify it
    if video_id.startswith("subnet_"):
        query = select(SubnetVideoRecord).filter(
            SubnetVideoRecord.video_id == video_id,
            SubnetVideoRecord.processing_state.in_([
                FocusVideoStateInternal.PURCHASE_PENDING.value,
                FocusVideoStateInternal.SUBMITTED.value,
            ]),
            SubnetVideoRecord.miner_hotkey == miner_hotkey,
            SubnetVideoRecord.deleted_at.is_(None),
        )
    else:
        query = select(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.processing_state.in_([
                FocusVideoStateInternal.PURCHASE_PENDING.value,
                FocusVideoStateInternal.SUBMITTED.value,
            ]),
            FocusVideoRecord.miner_hotkey == miner_hotkey,
            FocusVideoRecord.deleted_at.is_(None),
        )

    if with_lock:
        query = query.with_for_update()

    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if not video:
        print(
            f"confirm_transfer | video <{video_id}> not found or miner_hotkey doesn't match"
        )
        return False

    print(f"confirm_transfer | found video <{video_id}> in state <{video.processing_state}>")

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
                
                # Only focus videos have task_type and user_id
                if not video_id.startswith("subnet_") and video.task_type == TaskType.MARKETPLACE.value:
                    user_query = select(UserRecord).filter(UserRecord.id == video.user_id)
                    user_result = await db.execute(user_query)
                    user = user_result.scalar_one_or_none()
                    if user:
                        user.latest_mkt_rewarded_at = current_time
                        db.add(user)
                
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


DELAY_SECS = 120 # 2 minutes
RETRIES = 10  # 120s x 10 retries = 1200s = 20 mins


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
                    # Check if this is a subnet video
                    if video_id.startswith("subnet_"):
                        query = select(SubnetVideoRecord).filter(
                            SubnetVideoRecord.video_id == video_id,
                            SubnetVideoRecord.deleted_at.is_(None),
                        )
                    else:
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
                        await release_video_lock(video_id)
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
                        await release_video_lock(video_id)
                        return True
                    elif (
                        video is not None
                        and video.processing_state
                        == FocusVideoStateInternal.SUBMITTED.value
                    ):
                        print(
                            f"Video <{video_id}> has been marked as SUBMITTED. Stopping background task."
                        )
                        await release_video_lock(video_id)
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

        # Need a fresh db context since the previous one was closed in the loop
        async with get_db_context() as db:
            # Re-fetch the video record
            if video_id.startswith("subnet_"):
                query = select(SubnetVideoRecord).filter(
                    SubnetVideoRecord.video_id == video_id,
                    SubnetVideoRecord.deleted_at.is_(None),
                )
            else:
                query = select(FocusVideoRecord).filter(
                    FocusVideoRecord.video_id == video_id,
                    FocusVideoRecord.deleted_at.is_(None),
                )

            result = await db.execute(query)
            video = result.scalar_one_or_none()

            if video and video.processing_state == FocusVideoStateInternal.PURCHASE_PENDING.value:
                await increment_failed_purchases(db, video.miner_hotkey)
                video.processing_state = FocusVideoStateInternal.SUBMITTED.value
                video.updated_at = datetime.utcnow()
                db.add(video)
                await db.commit()
                print(f"Video <{video_id}> reverted to SUBMITTED state")
            else:
                print(f"Video <{video_id}> is no longer in PURCHASE_PENDING state (current: {video.processing_state if video else 'not found'}), skipping revert")

        await release_video_lock(video_id)
        return False

    except Exception as e:
        print(f"Error in confirm_video_purchased: {e}")
        await release_video_lock(video_id)
        return False
