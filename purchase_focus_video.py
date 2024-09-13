import os
import requests
import bittensor as bt
from bittensor import wallet as btcli_wallet
import argparse
import time

parser = argparse.ArgumentParser(description='Purchase a video from the API.')
parser.add_argument('video_id', nargs='+', help='The video id to purchase.')
args = parser.parse_args()

SUBTENSOR_NETWORK = None # "test" or None

API_BASE = (
    "https://dev-validator.api.omega-labs.ai"
    if SUBTENSOR_NETWORK == "test" else
    "https://validator.api.omega-labs.ai"
)

CYAN = "\033[96m" # field color
GREEN = "\033[92m" # indicating success
RED = "\033[91m" # indicating error
RESET = "\033[0m" # resetting color to the default

subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)

def main():
    if len(args.video_id) == 0:
        video_id = input(f"{CYAN}Enter focus video id (e.g. 0af028a8-dde9-4c10-bb7a-48a480ebb7f5): {RESET}")
    else:
        video_id = args.video_id[0]

    name = input(f"{CYAN}Enter wallet name (default: Coldkey): {RESET}") or "Coldkey"
    hotkey = input(f"{CYAN}Enter wallet hotkey name (default: Hotkey): {RESET}") or "Hotkey"
    path = input(f"{CYAN}Enter wallet path (default: ~/.bittensor/wallets/): {RESET}") or "~/.bittensor/wallets/"

    wallet = btcli_wallet(name=name, hotkey=hotkey, path=path)
    try:
        hotkey = wallet.get_hotkey()
    except Exception as e:
        print(f"{RED}Error loading hotkey: {e} {RESET}")
        exit(1)
    
    miner_hotkey = hotkey.ss58_address
    
    print(f"Purchasing video {video_id}...")
    # Post the request to the API
    purchase_response = requests.post(
        API_BASE + "/api/focus/purchase", 
        json={"video_id": video_id, "miner_hotkey": miner_hotkey}, 
        headers={"Content-Type": "application/json"},
        timeout=60
    )

    if purchase_response.status_code != 200:
        print(f"Error purchasing video {video_id}: {purchase_response.status_code}")
        return
    
    # Process the response
    purchase_data = purchase_response.json()
    print(purchase_data)
    if "status" in purchase_data and purchase_data["status"] == "error":
        print(f"Error purchasing video {video_id}: {purchase_data['message']}")
        return
    
    transfer_address_to = purchase_data["address"]
    transfer_amount = purchase_data["amount"]

    print(f"Initiating transfer of {transfer_amount} TAO for video {video_id}...")
    
    # Convert to bittensor.Balance
    if not isinstance(transfer_amount, bt.Balance):
        transfer_balance = bt.Balance.from_tao(transfer_amount)
    else:
        transfer_balance = transfer_amount
    
    success, block_hash, err_msg = subtensor._do_transfer(
        wallet,
        transfer_address_to,
        transfer_balance,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    if success:
        bt.__console__.print(
            ":white_heavy_check_mark: [green]Finalized[/green]"
        )
        bt.__console__.print(
            "[green]Block Hash: {}[/green]".format(block_hash)
        )

        if block_hash:
            print(f"Verifying purchase for video {video_id} on block hash {block_hash} ...")

            retries = 3
            for attempt in range(retries):
                try:
                    # Post the block hash to the API
                    verify_response = requests.post(
                        API_BASE + "/api/focus/verify-purchase",
                        json={"miner_hotkey": miner_hotkey, "video_id": video_id, "block_hash": block_hash},
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    print(f"Purchase verification response for video {video_id}:", verify_response.text)
                    if verify_response.status_code == 200:
                        return True
                    
                    if attempt < retries - 1:  # if it's not the last attempt
                        print(f"Attempt #{attempt + 1} to verify purchase failed. Retrying in 2 seconds...")
                        print(f"Error: {str(e)}")
                        time.sleep(2)

                except Exception as e:
                    if attempt < retries - 1:  # if it's not the last attempt
                        print(f"Attempt #{attempt + 1} to verify purchase failed. Retrying in 2 seconds...")
                        print(f"Error: {str(e)}")
                        time.sleep(2)
                    else:
                        print(f"All {retries} attempts failed. Unable to verify purchase.")
                        return False
                    
    else:
        print(f"Failed to complete transfer for video {video_id}: {err_msg}")

if __name__ == "__main__":
    main()