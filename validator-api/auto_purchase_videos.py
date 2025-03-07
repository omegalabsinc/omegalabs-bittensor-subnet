import argparse
import json
import os
import sys
import asyncio
from datetime import datetime
import bittensor as bt
from bittensor.core.extrinsics.transfer import _do_transfer
from bittensor.utils import unlock_key
import requests
from bittensor import wallet as btcli_wallet
from dotenv import load_dotenv

load_dotenv()
WALLET_NAME = os.getenv('WALLET_NAME')
WALLET_HOTKEY = os.getenv('WALLET_HOTKEY')
WALLET_PATH = os.getenv('WALLET_PATH')
WALLET_PASSWORD = os.getenv('WALLET_PASSWORD')

if not all([WALLET_NAME, WALLET_HOTKEY, WALLET_PATH, WALLET_PASSWORD]):
    print("Error: Missing required environment variables. Please check your .env file.")
    sys.exit(1)

SUBTENSOR_NETWORK = None  # "test" or None for mainnet
API_BASE = (
    "https://dev-sn24-api.omegatron.ai"
    if SUBTENSOR_NETWORK == "test" else
    "https://sn24-api.omegatron.ai"
)
CHECK_INTERVAL = 60  # 1 minute in seconds

wallet = btcli_wallet(
    name=WALLET_NAME,
    hotkey=WALLET_HOTKEY,
    path=WALLET_PATH,
)
coldkey = wallet.get_coldkey(password=WALLET_PASSWORD).ss58_address
print(f"Coldkey: {coldkey}")
unlock_status = unlock_key(wallet, unlock_type="coldkey")
if not unlock_status.success:
    print(f"Failed to unlock wallet: {unlock_status.message}")
    sys.exit(1)
print("Wallet unlocked")
subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)

def get_auth_headers(wallet):
    hotkey = wallet.get_hotkey()
    miner_hotkey = hotkey.ss58_address
    miner_hotkey_signature = f"0x{hotkey.sign(miner_hotkey).hex()}"
    return miner_hotkey, miner_hotkey_signature

def list_videos():
    try:
        videos_response = requests.get(
            API_BASE + "/api/focus/get_list",
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if videos_response.status_code != 200:
            print(f"Error fetching focus videos: {videos_response.status_code}")
            return None
        return videos_response.json()
    except Exception as e:
        print(f"Error listing videos: {str(e)}")
        return None

async def transfer_operation(wallet, transfer_address_to: str, transfer_balance: bt.Balance):
    try:
        print(f"Transferring {transfer_balance} TAO to {transfer_address_to}")
        success, block_hash, err_msg = _do_transfer(
            subtensor,
            wallet,
            transfer_address_to,
            transfer_balance,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
        print(f"Transfer success: {success}, Block Hash: {block_hash}, Error: {err_msg}")
        return success, block_hash, err_msg
    except Exception as e:
        print(f"Transfer error: {str(e)}")
        return False, None, str(e)

async def purchase_video(video_id, wallet):
    try:
        miner_hotkey, miner_hotkey_signature = get_auth_headers(wallet)
        
        purchase_response = requests.post(
            API_BASE + "/api/focus/purchase",
            auth=(miner_hotkey, miner_hotkey_signature),
            json={"video_id": video_id},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        purchase_data = purchase_response.json()
        if purchase_response.status_code != 200:
            print(f"Error purchasing video {video_id}: {purchase_response.status_code}")
            if "detail" in purchase_data:
                print(f"Details: {purchase_data['detail']}")
            return False
        
        if "status" in purchase_data and purchase_data["status"] == "error":
            print(f"Error purchasing video {video_id}: {purchase_data['message']}")
            return False

        transfer_address_to = purchase_data["address"]
        transfer_amount = purchase_data["amount"]
        transfer_balance = bt.Balance.from_tao(transfer_amount)

        success, block_hash, err_msg = await transfer_operation(wallet, transfer_address_to, transfer_balance)
        
        if success:
            print(f"Transfer finalized. Block Hash: {block_hash}")
            save_purchase_info(video_id, miner_hotkey, block_hash, "purchased", transfer_amount)
            verify_result = await verify_purchase(video_id, miner_hotkey, block_hash, miner_hotkey_signature)
            if not verify_result:
                print(f"Error verifying purchase after transfer. Please verify manually.")
            return True
        else:
            print(f"Failed to complete transfer for video {video_id}: {err_msg}")
            return False

    except Exception as e:
        print(f"Error in purchase_video: {str(e)}")
        return False

async def verify_purchase(video_id, miner_hotkey, block_hash, miner_hotkey_signature):
    try:
        verify_response = requests.post(
            API_BASE + "/api/focus/verify-purchase",
            auth=(miner_hotkey, miner_hotkey_signature),
            json={"miner_hotkey": miner_hotkey, "video_id": video_id, "block_hash": block_hash},
            headers={"Content-Type": "application/json"},
            timeout=90
        )
        
        if verify_response.status_code == 200:
            print(f"Purchase verified successfully!")
            save_purchase_info(video_id, miner_hotkey, block_hash, "verified")
            return True
        return False
    except Exception as e:
        print(f"Error verifying purchase: {str(e)}")
        return False

def save_purchase_info(video_id, hotkey, block_hash, state, amount=None):
    purchases_file = os.path.expanduser("~/.omega/focus_videos.json")
    os.makedirs(os.path.dirname(purchases_file), exist_ok=True)
    
    purchases = []
    if os.path.exists(purchases_file):
        with open(purchases_file, 'r') as f:
            purchases = json.load(f)
    
    for purchase in purchases:
        if purchase['video_id'] == video_id:
            purchase['state'] = state
            purchase['miner_hotkey'] = hotkey
            purchase['block_hash'] = block_hash
            if amount is not None:
                purchase['amount'] = amount
            break
    else:
        new_purchase = {
            "video_id": video_id,
            "miner_hotkey": hotkey,
            "block_hash": block_hash,
            "state": state,
            "created_at": datetime.now().isoformat()
        }
        if amount is not None:
            new_purchase['amount'] = amount
        purchases.append(new_purchase)
    
    with open(purchases_file, 'w') as f:
        json.dump(purchases, f, indent=2)

async def check_and_purchase_videos():
    while True:
        try:
            print(f"\n[{datetime.now()}] Checking for available videos...")
            
            videos = list_videos()
            if not videos:
                print("No videos available or error fetching videos")
                await asyncio.sleep(CHECK_INTERVAL)
                continue
            
            # Check each video
            for video in videos:
                video_cost = float(video['expected_reward_tao'])
                balance = subtensor.get_balance(coldkey).tao
                print(f"Current wallet balance: {balance} TAO")
                if balance >= video_cost * 1.05:  # 5% buffer
                    print(f"Attempting to purchase video {video['video_id']} for {video_cost} TAO")
                    success = await purchase_video(video['video_id'], wallet)
                    if success:
                        print(f"Successfully purchased video {video['video_id']}")
                    else:
                        print(f"Failed to purchase video {video['video_id']}")
                else:
                    print(f"Insufficient balance for video {video['video_id']} (cost: {video_cost} TAO)")
            
            print(f"Waiting {CHECK_INTERVAL} seconds before next check...")
            await asyncio.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(check_and_purchase_videos())
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)