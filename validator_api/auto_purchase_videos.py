"""
This script is used to automatically purchase videos on the OMEGA Focus marketplace on a loop (about every 15 minutes).
Ideally this script will be run-and-forget in the background.
It will check for available videos and purchase them if the balance is sufficient.
It will also verify the purchase and save the purchase information to a file.
It will convert all staked dynamic TAO on subnet 24 and convert it to base TAO,
since emission rewards are paid in dynamic TAO but the purchase is in base TAO.
The unstaking loop occurs about every 1 hour; it doesn't need to happen more frequently
since emissions are only paid once every 360 blocks (1.2 hours).

IMPORTANT: if you want to run this script in the background, e.g. with pm2, you need to set up your miner wallet to be passwordless.
Otherwise, the script will hang when waiting for the wallet to be unlocked.
First test the script by running it directly with Python to verify it doesn't prompt for a password, before using pm2 or any other background process manager.
You can generate a passwordless wallet with the following `btcli` commands:
Generate coldkey: `btcli wallet new_coldkey --wallet.name <coldkey_name> --no-use-password`
Generate hotkey: `btcli wallet new_hotkey --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name> --no-use-password`
Then, set the environment variables in the `validator_api/.env` file to the generated coldkey and hotkey.

Then, make sure the wallet is registered on the subnet, and has enough TAO to purchase videos!!!
`btcli subnet register --netuid 24 --wallet.name <coldkey_name> --wallet.hotkey <hotkey_name>`
If the wallet is not registered, the script will exit. If you encounter SubstrateRequestException, keep retrying until it succeeds.

setup:
the ecosystem.config.js file is configured to kill the process if it detects that the miner is not registered on subnet 24.
`pm2 start ecosystem.config.js`
"""


import json
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import bittensor as bt
from bittensor.core.extrinsics.transfer import _do_transfer
from bittensor.utils import unlock_key
import requests
from bittensor import wallet as btcli_wallet
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto_purchase.log")
    ]
)
logger = logging.getLogger("auto_purchase")

load_dotenv()
WALLET_NAME = os.getenv("WALLET_NAME")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY")
WALLET_PATH = os.getenv("WALLET_PATH")

if not all([WALLET_NAME, WALLET_HOTKEY, WALLET_PATH]):
    logger.error("Missing required environment variables. Please check your .env file.")
    sys.exit(1)

SUBTENSOR_NETWORK = None  # "test" or None for mainnet
API_BASE = (
    "https://dev-sn24-api.omegatron.ai"
    if SUBTENSOR_NETWORK == "test"
    else "https://sn24-api.omegatron.ai"
)
PURCHASE_ATTEMPT_INTERVAL = 20 * 60  # 20 minutes in seconds
UNSTAKE_INTERVAL = 60 * 60  # 1 hour in seconds

wallet = btcli_wallet(
    name=WALLET_NAME,
    hotkey=WALLET_HOTKEY,
    path=WALLET_PATH,
)
coldkey_ss58 = wallet.get_coldkey().ss58_address
hotkey_ss58 = wallet.get_hotkey().ss58_address
logger.info(f"Coldkey: {coldkey_ss58}")
logger.info(f"Hotkey: {hotkey_ss58}")
unlock_status = unlock_key(wallet, unlock_type="coldkey")
if not unlock_status.success:
    logger.error(f"Failed to unlock wallet: {unlock_status.message}")
    sys.exit(1)
logger.info("Wallet unlocked")
subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)

last_successful_unstake_time = None

def unstake_balance(wallet):
    global last_successful_unstake_time
    try:
        stakes = subtensor.get_stake_for_coldkey(coldkey_ss58)
        logger.info(f"Stakes: {stakes}")
        if not stakes:
            logger.info("No stakes found for coldkey")
            return False
        for stake in stakes:
            amount = stake.stake.tao  # this is actually dynamic TAO
            netuid = stake.netuid
            if netuid == 24:
                logger.info(f"Attempting to unstake {amount} dTAO from netuid 24")
                stake_result = subtensor.unstake(
                    wallet=wallet,
                    amount=amount,
                    netuid=netuid,
                    wait_for_finalization=False,
                    wait_for_inclusion=True,
                )
                if stake_result:
                    logger.info(f"Successfully unstaked {amount} dTAO from netuid {netuid}")
                    last_successful_unstake_time = datetime.now()
                    return True
                else:
                    logger.error(f"Failed to unstake {amount} dTAO from netuid {netuid}")
                    return False
        return False
    except Exception as e:
        logger.error(f"Error unstaking balance: {str(e)}")
        return False


def attempt_unstake(wallet):
    """
    Attempts to unstake balance if more than UNSTAKE_INTERVAL has passed since the last successful unstake.
    Returns True if unstaking was attempted, False otherwise.
    """
    global last_successful_unstake_time
    
    current_time = datetime.now()
    should_unstake = False
    
    if last_successful_unstake_time is None:
        logger.info("No previous unstake recorded. Running unstake_balance.")
        should_unstake = True
    else:
        time_elapsed = current_time - last_successful_unstake_time
        if time_elapsed > timedelta(seconds=UNSTAKE_INTERVAL):
            logger.info(f"More than {UNSTAKE_INTERVAL/3600:.1f} hours since last successful unstake. Running unstake_balance.")
            should_unstake = True
        else:
            logger.info(f"Less than {UNSTAKE_INTERVAL/3600:.1f} hours since last successful unstake. Skipping unstake_balance.")
    
    if should_unstake:
        unstake_success = unstake_balance(wallet)
        if not unstake_success:
            logger.warning("Unstaking was not successful. Will try again in the next cycle.")
        return True
    
    return False


def get_auth_headers():
    # Get the hotkey object from the wallet
    hotkey = wallet.get_hotkey()
    # Sign the hotkey's SS58 address with the hotkey
    miner_hotkey_signature = f"0x{hotkey.sign(hotkey_ss58.encode()).hex()}"
    return hotkey_ss58, miner_hotkey_signature


def list_videos():
    try:
        videos_response = requests.get(
            API_BASE + "/api/focus/get_list",
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if videos_response.status_code != 200:
            logger.error(f"Error fetching focus videos: {videos_response.status_code}")
            return None
        return videos_response.json()
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        return None


async def transfer_operation(
    wallet, transfer_address_to: str, transfer_balance: bt.Balance
):
    try:
        logger.info(f"Transferring {transfer_balance} TAO to {transfer_address_to}")
        success, block_hash, err_msg = _do_transfer(
            subtensor,
            wallet,
            transfer_address_to,
            transfer_balance,
            wait_for_finalization=False,
            wait_for_inclusion=True,
        )
        logger.info(
            f"Transfer success: {success}, Block Hash: {block_hash}, Error: {err_msg}"
        )
        return success, block_hash, err_msg
    except Exception as e:
        logger.error(f"Transfer error: {str(e)}")
        return False, None, str(e)


async def is_miner_registered() -> bool:
    metagraph = subtensor.metagraph(netuid=24)
    return hotkey_ss58 in metagraph.hotkeys


async def purchase_video(video_id, wallet):
    try:
        miner_hotkey, miner_hotkey_signature = get_auth_headers()

        purchase_response = requests.post(
            API_BASE + "/api/focus/purchase",
            auth=(miner_hotkey, miner_hotkey_signature),
            json={"video_id": video_id},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        purchase_data = purchase_response.json()
        if purchase_response.status_code != 200:
            logger.error(f"Error purchasing video {video_id}: {purchase_response.status_code}")
            if "detail" in purchase_data:
                logger.error(f"Details: {purchase_data['detail']}")
            return False

        if "status" in purchase_data and purchase_data["status"] == "error":
            logger.error(f"Error purchasing video {video_id}: {purchase_data['message']}")
            return False

        transfer_address_to = purchase_data["address"]
        transfer_amount = purchase_data["amount"]
        transfer_balance = bt.Balance.from_tao(transfer_amount)

        success, block_hash, err_msg = await transfer_operation(
            wallet, transfer_address_to, transfer_balance
        )

        if success:
            logger.info(f"Transfer finalized. Block Hash: {block_hash}")
            save_purchase_info(
                video_id, miner_hotkey, block_hash, "purchased", transfer_amount
            )
            verify_result = await verify_purchase(video_id, block_hash)
            if not verify_result:
                logger.error(
                    "Error verifying purchase after transfer. Please verify manually."
                )
            return True
        else:
            logger.error(f"Failed to complete transfer for video {video_id}: {err_msg}")
            return False

    except Exception as e:
        logger.error(f"Error in purchase_video: {str(e)}")
        return False


async def verify_purchase(video_id, block_hash):
    try:
        miner_hotkey, miner_hotkey_signature = get_auth_headers()
        verify_response = requests.post(
            API_BASE + "/api/focus/verify-purchase",
            auth=(miner_hotkey, miner_hotkey_signature),
            json={
                "miner_hotkey": miner_hotkey,
                "video_id": video_id,
                "block_hash": block_hash,
            },
            headers={"Content-Type": "application/json"},
            timeout=90,
        )

        if verify_response.status_code == 200:
            logger.info("Purchase verified successfully!")
            save_purchase_info(video_id, miner_hotkey, block_hash, "verified")
            return True
        return False
    except Exception as e:
        logger.error(f"Error verifying purchase: {str(e)}")
        return False


def save_purchase_info(video_id, hotkey, block_hash, state, amount=None):
    purchases_file = os.path.expanduser("~/.omega/focus_videos.json")
    os.makedirs(os.path.dirname(purchases_file), exist_ok=True)

    purchases = []
    if os.path.exists(purchases_file):
        with open(purchases_file, "r") as f:
            purchases = json.load(f)

    for purchase in purchases:
        if purchase["video_id"] == video_id:
            purchase["state"] = state
            purchase["miner_hotkey"] = hotkey
            purchase["block_hash"] = block_hash
            if amount is not None:
                purchase["amount"] = amount
            break
    else:
        new_purchase = {
            "video_id": video_id,
            "miner_hotkey": hotkey,
            "block_hash": block_hash,
            "state": state,
            "created_at": datetime.now().isoformat(),
        }
        if amount is not None:
            new_purchase["amount"] = amount
        purchases.append(new_purchase)

    with open(purchases_file, "w") as f:
        json.dump(purchases, f, indent=2)


async def check_and_purchase_videos():
    while True:
        try:
            if not await is_miner_registered():
                # this prevents purchase attempts if the miner is not registered
                # if a miner is not registered, they won't receive emissions!
                # that would be very bad, so we exit
                logger.error("Miner not registered on subnet 24. Process will terminate and not restart.")
                with open("/tmp/miner_not_registered", "w") as f:
                    f.write("1")
                sys.exit(99)
            
            attempt_unstake(wallet)

            logger.info("Checking for available videos...")

            videos = list_videos()
            if not videos:
                logger.warning("No videos available or error fetching videos")
                continue

            for video in videos:
                video_cost = float(video["expected_reward_tao"])
                balance = subtensor.get_balance(coldkey_ss58).tao
                logger.info(f"Current wallet balance: {balance} TAO")
                if balance >= video_cost * 1.05:  # 5% buffer
                    logger.info(
                        f"Attempting to purchase video {video['video_id']} for {video_cost} TAO"
                    )
                    success = await purchase_video(video["video_id"], wallet)
                    if success:
                        logger.info(f"Successfully purchased video {video['video_id']}")
                    else:
                        logger.error(f"Failed to purchase video {video['video_id']}")
                else:
                    logger.warning(
                        f"Insufficient balance for video {video['video_id']} (cost: {video_cost} TAO)"
                    )
            logger.info(f"Waiting {PURCHASE_ATTEMPT_INTERVAL} seconds before next check...")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        await asyncio.sleep(PURCHASE_ATTEMPT_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(check_and_purchase_videos())
        # async def main():
        #     miner_hotkey, miner_hotkey_signature = get_auth_headers()
        #     return await verify_purchase(
        #         "p7sNMdfsv",
        #         miner_hotkey,
        #         "0xa6798908e3e8d895a327b1226ba9345d7234268c6b175ef2f7012208916bbbfb",
        #         miner_hotkey_signature
        #     )

        # logger.info(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)
