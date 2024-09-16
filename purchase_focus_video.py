"""
Using the OMEGA Focus Video Purchase System:

1. Setup:
   - Ensure you have the latest required libraries installed. See requirements.txt.
   - Make sure you have your SN24 Bittensor wallet set up. You *MUST* use your SN24 registered wallet to purchase videos.

2. Running the Script:
   - Open a terminal and navigate to the directory containing the script.
   - Run the script with: `python purchase_focus_video.py`

3. Main Menu Options:
   When you run the script, you'll see a menu with 5 options:

   1. View Focus Videos
   2. Purchase Focus Video
   3. Verify Purchase
   4. Display Order History
   5. Exit

4. Using the Options:

   Option 1: View Focus Videos
   - Displays a list of available focus videos with details like Video ID, Score, Cost, and Expected Reward.
   - The displayed cost is the amount of TAO tokens required to purchase the video.
   - The expected reward is the amount of TAO tokens you'll earn from SN24 emissions for purchasing the video.
   - Select a number from the list next to the video you want to purchase.

   Option 2: Purchase Focus Video
   - Allows you to purchase a video by entering its ID.
   - You'll need to provide your wallet information (name, hotkey, path).
   - The script will initiate a transfer of TAO tokens to the OMEGA Focus App user who created the video. This secures the purchase of the video.

   Option 3: Verify Purchase
   - This option is used when there are issues with the purchase verification during the purchase process. 
   - If you've successfully transferred the TAO tokens but the purchase wasn't verified, you can use this option to verify the purchase.
   - You'll need to provide the Video ID, Miner Hotkey, and Block Hash.

   Option 4: Display Order History
   - Shows a list of your previous purchases and their current status.

   Option 5: Exit
   - Closes the application.

5. Important Notes:
   - The script can be ran using Bittensor mainnet or testnet based on the SUBTENSOR_NETWORK variable. Set it to "test" for testnet. Set to None for mainnet.
   - Purchases are saved locally in '~/.omega/focus_videos.json'.
   - Always ensure you have sufficient TAO tokens in your wallet before making a purchase.

6. Wallet Information:
   - When purchasing, you'll need to provide your Bittensor wallet details.
   - You *MUST* use your SN24 registered wallet to purchase videos.
   - The default wallet path is '~/.bittensor/wallets/'.

Remember to keep your wallet information secure and never share your private keys.
"""

import os
import requests
import bittensor as bt
from bittensor import wallet as btcli_wallet
import argparse
import time
import json
from tabulate import tabulate
from datetime import datetime

parser = argparse.ArgumentParser(description='Interact with the OMEGA Focus Videos API.')
args = parser.parse_args()

SUBTENSOR_NETWORK = None # "test" or None

API_BASE = (
    "https://dev-validator.api.omega-labs.ai"
    if SUBTENSOR_NETWORK == "test" else
    "https://validator.api.omega-labs.ai"
)

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)

def list_videos():
    videos_response = requests.get(
        API_BASE + "/api/focus/get_list",
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    if videos_response.status_code != 200:
        print(f"{RED}Error fetching focus videos: {videos_response.status_code}{RESET}")
        return None
    
    videos_data = videos_response.json()
    return videos_data

def display_videos(videos_data):
    if not videos_data:
        print(f"{RED}No videos available.{RESET}")
        return

    print(f"\n{CYAN}Available Focus Videos:{RESET}")
    
    # Prepare the data for tabulate
    table_data = []
    for idx, video in enumerate(videos_data, 1):
        # Convert created_at to a more readable format
        created_at = datetime.fromisoformat(video['created_at'].replace('Z', '+00:00'))
        formatted_date = created_at.strftime("%Y-%m-%d %H:%M:%S")
        
        table_data.append([
            idx,
            video['video_id'],
            f"{video['video_score']:.3f}",
            f"{video['expected_reward_tao']:.5f}",
            f"{float(video['expected_reward_tao']) / 0.9:.5f}",
            #formatted_date
        ])
    
    # Create the table
    headers = ["#", "Video ID", "Score", "Cost (TAO)", "Expected Reward (TAO)"]
    table = tabulate(table_data, headers=headers, tablefmt="pretty")
    
    print(table)

def purchase_video(video_id=None):
    if not video_id:
        video_id = input(f"{CYAN}Enter focus video id: {RESET}")

    name = input(f"{CYAN}Enter wallet name (default: Coldkey): {RESET}") or "Coldkey"
    hotkey = input(f"{CYAN}Enter wallet hotkey name (default: Hotkey): {RESET}") or "Hotkey"
    path = input(f"{CYAN}Enter wallet path (default: ~/.bittensor/wallets/): {RESET}") or "~/.bittensor/wallets/"

    wallet = btcli_wallet(name=name, hotkey=hotkey, path=path)
    try:
        hotkey = wallet.get_hotkey()
    except Exception as e:
        print(f"{RED}Error loading hotkey: {e} {RESET}")
        return

    miner_hotkey = hotkey.ss58_address
    
    print(f"Purchasing video {video_id}...")
    purchase_response = requests.post(
        API_BASE + "/api/focus/purchase", 
        json={"video_id": video_id, "miner_hotkey": miner_hotkey}, 
        headers={"Content-Type": "application/json"},
        timeout=60
    )

    if purchase_response.status_code != 200:
        print(f"{RED}Error purchasing video {video_id}: {purchase_response.status_code}{RESET}")
        return
    
    purchase_data = purchase_response.json()
    if "status" in purchase_data and purchase_data["status"] == "error":
        print(f"{RED}Error purchasing video {video_id}: {purchase_data['message']}{RESET}")
        return
    
    transfer_address_to = purchase_data["address"]
    transfer_amount = purchase_data["amount"]

    print(f"Initiating transfer of {transfer_amount} TAO for video {video_id}...")
    
    transfer_balance = bt.Balance.from_tao(transfer_amount)
    
    success, block_hash, err_msg = subtensor._do_transfer(
        wallet,
        transfer_address_to,
        transfer_balance,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    if success:
        print(f"{GREEN}Transfer finalized. Block Hash: {block_hash}{RESET}")
        save_purchase_info(video_id, block_hash, "purchased", transfer_amount)
        verify_purchase(video_id, miner_hotkey, block_hash)
    else:
        print(f"{RED}Failed to complete transfer for video {video_id}.{RESET}")

def verify_purchase(video_id, miner_hotkey, block_hash):
    if block_hash:
        print(f"Verifying purchase for video {video_id} on block hash {block_hash} ...")

        retries = 3
        for attempt in range(retries):
            try:
                verify_response = requests.post(
                    API_BASE + "/api/focus/verify-purchase",
                    json={"miner_hotkey": miner_hotkey, "video_id": video_id, "block_hash": block_hash},
                    headers={"Content-Type": "application/json"},
                    timeout=90
                )
                print(f"Purchase verification response for video {video_id}:", verify_response.text)
                if verify_response.status_code == 200:
                    print(f"{GREEN}Purchase verified successfully!{RESET}")
                    save_purchase_info(video_id, block_hash, "verified")
                    return True
                
                if attempt < retries - 1:
                    print(f"{CYAN}Attempt #{attempt + 1} to verify purchase failed. Retrying in 2 seconds...{RESET}")
                    time.sleep(2)
            except Exception as e:
                if attempt < retries - 1:
                    print(f"{CYAN}Attempt #{attempt + 1} to verify purchase failed. Retrying in 2 seconds...{RESET}")
                    print(f"{RED}Error: {str(e)}{RESET}")
                    time.sleep(2)
                else:
                    print(f"{RED}All {retries} attempts failed. Unable to verify purchase.{RESET}")
                    return False

def display_saved_orders():
    purchases_file = os.path.expanduser("~/.omega/focus_videos.json")
    if not os.path.exists(purchases_file):
        print(f"{RED}No saved orders found.{RESET}")
        return

    with open(purchases_file, 'r') as f:
        purchases = json.load(f)

    if not purchases:
        print(f"{RED}No saved orders found.{RESET}")
        return

    # Sort purchases by most recent first
    # Assuming 'created_at' field exists, if not, we'll need to add it when saving purchase info
    purchases.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    print(f"\n{CYAN}Saved Orders:{RESET}")
    
    table_data = []
    for purchase in purchases:
        created_at = purchase.get('created_at', 'N/A')
        if created_at != 'N/A':
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
        
        table_data.append([
            purchase['video_id'],
            purchase['state'],
            purchase.get('amount', 'N/A'),
            f"{float(purchase['amount']) / 0.9:.5f}",
            purchase['block_hash'],
            created_at
        ])
    
    headers = ["Video ID", "Purchase State", "Cost (TAO)", "Estimated Reward (TAO)", "Block Hash", "Purchase Date"]
    table = tabulate(table_data, headers=headers, tablefmt="pretty")
    
    print(table)

def save_purchase_info(video_id, block_hash, state, amount=None):
    purchases_file = os.path.expanduser("~/.omega/focus_videos.json")
    os.makedirs(os.path.dirname(purchases_file), exist_ok=True)
    
    purchases = []
    if os.path.exists(purchases_file):
        with open(purchases_file, 'r') as f:
            purchases = json.load(f)
    
    # Check if the video_id already exists
    for purchase in purchases:
        if purchase['video_id'] == video_id:
            purchase['state'] = state
            purchase['block_hash'] = block_hash
            if amount is not None:
                purchase['amount'] = amount
            break
    else:
        # If the video_id doesn't exist, create a new entry
        new_purchase = {
            "video_id": video_id,
            "block_hash": block_hash,
            "state": state,
            "created_at": datetime.now().isoformat()  # Add creation timestamp
        }
        if amount is not None:
            new_purchase['amount'] = amount
        purchases.append(new_purchase)
    
    with open(purchases_file, 'w') as f:
        json.dump(purchases, f, indent=2)
    
    print(f"{GREEN}Purchase information {'updated' if state == 'verified' else 'saved'} to {purchases_file}{RESET}")

def main():
    while True:
        print(f"\n{CYAN}Welcome to the OMEGA Focus Videos Purchase System{RESET}")
        print("1. View Focus Videos")
        print("2. Purchase Focus Video")
        print("3. Verify Purchase")
        print("4. Display Order History")
        print("5. Exit")
        
        choice = input(f"{CYAN}Enter your choice (1-5): {RESET}")
        
        if choice == '1':
            videos_data = list_videos()
            if videos_data:
                display_videos(videos_data)
                purchase_option = input(f"\n{CYAN}Enter the number of the video you want to purchase or press 'n' to return to menu: {RESET}").lower()
                if purchase_option.isdigit():
                    video_index = int(purchase_option) - 1
                    if 0 <= video_index < len(videos_data):
                        purchase_video(videos_data[video_index]['video_id'])
                    else:
                        print(f"{RED}Invalid video number.{RESET}")
                elif purchase_option != 'n':
                    print(f"{RED}Invalid input. Returning to main menu.{RESET}")
        elif choice == '2':
            purchase_video()
        elif choice == '3':
            video_id = input(f"{CYAN}Enter video ID: {RESET}")
            miner_hotkey = input(f"{CYAN}Enter miner hotkey: {RESET}")
            block_hash = input(f"{CYAN}Enter block hash: {RESET}")
            verify_purchase(video_id, miner_hotkey, block_hash)
        elif choice == '4':
            display_saved_orders()
        elif choice == '5':
            print(f"{GREEN}Thank you for using the OMEGA Focus Videos Purchase System. Goodbye!{RESET}")
            break
        else:
            print(f"{RED}Invalid choice. Please try again.{RESET}")

if __name__ == "__main__":
    main()
