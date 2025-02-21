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
   - After the transfer is complete, the script will attempt to verify the purchase. 
   - Once successful, you're all set! SN24 validators will automatically detect your purchase and reward your expected TAO emissions.

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
   - Once a purchase has been verified successful, SN24 validators will automatically detect your purchase and reward your expected TAO emissions.

6. Wallet Information:
   - When purchasing, you'll need to provide your Bittensor wallet details.
   - You *MUST* use your SN24 registered wallet to purchase videos.
   - The default wallet path is '~/.bittensor/wallets/'.

Remember to keep your wallet information secure and never share your private keys.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime

import bittensor as bt
import requests
from bittensor import wallet as btcli_wallet
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Interact with the OMEGA Focus Videos API.')
args = parser.parse_args()

SUBTENSOR_NETWORK = None # "test" or None

API_BASE = (
    "https://dev-sn24-api.omegatron.ai"
    if SUBTENSOR_NETWORK == "test" else
    "https://sn24-api.omegatron.ai"
)
# API_BASE = "http://localhost:8000"

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def initialize_subtensor():
    try:
        subtensor = bt.subtensor(network=SUBTENSOR_NETWORK)
        #print(f"{GREEN}Subtensor initialized successfully.{RESET}")
        return subtensor
    except Exception as e:
        print(f"{RED}Error initializing subtensor: {str(e)}{RESET}")
        raise

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
    if not videos_data or len(videos_data) == 0:
        print(f"\n{RED}No videos available.{RESET}")
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


class TransferTimeout(Exception):
    pass

def reset_terminal():
    # Try multiple methods to reset the terminal
    os.system('stty sane')
    os.system('reset')
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

def transfer_operation(wallet, transfer_address_to, transfer_balance, result_queue):
    try:
        subtensor = initialize_subtensor()
        success, block_hash, err_msg = subtensor._do_transfer(
            wallet,
            transfer_address_to,
            transfer_balance,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
        result_queue.put((success, block_hash, err_msg))
    except Exception as e:
        result_queue.put((False, None, str(e)))

def transfer_with_timeout(wallet, transfer_address_to, transfer_balance):
    result_queue = multiprocessing.Queue()
    
    transfer_process = multiprocessing.Process(
        target=transfer_operation,
        args=(wallet, transfer_address_to, transfer_balance, result_queue)
    )
    
    transfer_process.start()
    transfer_process.join(timeout=150)  # 2m 30s = 150 seconds
    
    if transfer_process.is_alive():
        transfer_process.terminate()
        transfer_process.join()
        reset_terminal()
        print("\nTransfer operation timed out after 2 minutes 30 seconds. Exiting.")
    
    if not result_queue.empty():
        return result_queue.get()
    else:
        return False, None, "Transfer process exited without result"

def get_wallet(wallet_name=None, wallet_hotkey=None, wallet_path=None):
    if wallet_name is not None:
        name = wallet_name
    else:
        name = input(f"{CYAN}Enter wallet name (default: Coldkey): {RESET}") or "Coldkey"
    if wallet_hotkey is not None:
        hotkey_name = wallet_hotkey
    else:
        hotkey_name = input(f"{CYAN}Enter wallet hotkey name (default: Hotkey): {RESET}") or "Hotkey"
    if wallet_path is not None:
        path = wallet_path
    else:
        path = input(f"{CYAN}Enter wallet path (default: ~/.bittensor/wallets/): {RESET}") or "~/.bittensor/wallets/"
    
    wallet = btcli_wallet(name=name, hotkey=hotkey_name, path=path)
    try:
        hotkey = wallet.get_hotkey()
    except Exception as e:
        print(f"{RED}Error loading hotkey: {e} {RESET}")
        return
    return wallet, name, hotkey_name, path

def get_auth_headers(wallet):
    hotkey = wallet.get_hotkey()
    miner_hotkey = hotkey.ss58_address
    miner_hotkey_signature = f"0x{hotkey.sign(miner_hotkey).hex()}"
    return miner_hotkey, miner_hotkey_signature

def purchase_video(
    video_id=None,
    wallet_name=None,
    wallet_hotkey=None,
    wallet_path=None
):
    if not video_id:
        video_id = input(f"{CYAN}Enter focus video id: {RESET}")

    wallet, name, hotkey_name, path = get_wallet(wallet_name, wallet_hotkey, wallet_path)
    miner_hotkey, miner_hotkey_signature = get_auth_headers(wallet)
    
    print(f"Purchasing video {video_id}...")
    print(f"{RED}You will only have 2 minutes and 30 seconds to complete the transfer of TAO tokens, otherwise the purchase will be reverted.{RESET}")
    purchase_response = requests.post(
        API_BASE + "/api/focus/purchase",
        auth=(miner_hotkey, miner_hotkey_signature),
        json={"video_id": video_id},
        headers={"Content-Type": "application/json"},
        timeout=60
    )

    purchase_data = purchase_response.json()
    if purchase_response.status_code != 200:
        print(f"{RED}Error purchasing video {video_id}: {purchase_response.status_code}{RESET}")
        if "detail" in purchase_data:
            print(f"{RED}Details: {purchase_data['detail']}{RESET}")
        return
    
    if "status" in purchase_data and purchase_data["status"] == "error":
        print(f"{RED}Error purchasing video {video_id}: {purchase_data['message']}{RESET}")
        return
    
    try:
        transfer_address_to = purchase_data["address"]
        transfer_amount = purchase_data["amount"]

        print(f"Initiating transfer of {transfer_amount} TAO for video {video_id}...")
        
        transfer_balance = bt.Balance.from_tao(transfer_amount)
        

        try:
            success, block_hash, err_msg = transfer_with_timeout(wallet, transfer_address_to, transfer_balance)
        except TransferTimeout:
            print(f"\n{RED}Transfer operation timed out after 2 minutes and 30 seconds. Aborting purchase.{RESET}")
            reset_terminal()
            revert_pending_purchase(video_id, miner_hotkey, miner_hotkey_signature)
            repurchase_input(video_id, name, hotkey_name, path)
            return
        
        """
        success, block_hash, err_msg = subtensor._do_transfer(
            wallet,
            transfer_address_to,
            transfer_balance,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
        """

        if success:
            print(f"{GREEN}Transfer finalized. Block Hash: {block_hash}{RESET}")
            save_purchase_info(video_id, miner_hotkey, block_hash, "purchased", transfer_amount)
            verify_result = verify_purchase(video_id, miner_hotkey, block_hash, miner_hotkey_signature)
            if not verify_result:
                print(f"{RED}There was an error verifying your purchase after successfully transferring TAO. Please try the 'Verify Purchase' option immediately and contact an admin if you are unable to successfully verify.{RESET}")
        else:
            print(f"{RED}Failed to complete transfer for video {video_id}.{RESET}")
            revert_pending_purchase(video_id, miner_hotkey, miner_hotkey_signature)
            repurchase_input(video_id, name, hotkey_name, path)

    except Exception as e:
        print(f"{RED}Error transferring TAO tokens: {str(e)}{RESET}")
        if "EOF occurred in violation of protocol" in str(e):
            print(f"{RED}Subtensor connection error detected. Re-initializing subtensor.{RESET}")
            initialize_subtensor()
        revert_pending_purchase(video_id, miner_hotkey, miner_hotkey_signature)
        repurchase_input(video_id, name, hotkey_name, path)

def revert_pending_purchase(video_id, miner_hotkey, miner_hotkey_signature):
    print(f"Reverting Pending Purchasing of video {video_id}...")
    revert_response = requests.post(
        API_BASE + "/api/focus/revert-pending-purchase",
        auth=(miner_hotkey, miner_hotkey_signature),
        json={"video_id": video_id},
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    if revert_response.status_code != 200:
        print(f"{RED}Error reverting pending purchase of video {video_id}: {revert_response.status_code}{RESET}")
        return
    if revert_response.status_code == 200:
        print(f"{GREEN}Pending purchase of video {video_id} reverted successfully.{RESET}")
    return

def repurchase_input(video_id, wallet_name=None, wallet_hotkey=None, wallet_path=None):
    repurchase = input(f"{CYAN}Do you want to repurchase video {video_id}? (y/n): {RESET}").lower()
    if repurchase == 'y':
        purchase_video(video_id, wallet_name, wallet_hotkey, wallet_path)
    elif repurchase != 'n':
        print(f"{RED}Invalid input. Please enter 'y' or 'n'.{RESET}")
        repurchase_input(video_id, wallet_name, wallet_hotkey, wallet_path)

def display_saved_orders(for_verification=False):
    purchases_file = os.path.expanduser("~/.omega/focus_videos.json")
    if not os.path.exists(purchases_file):
        print(f"{RED}No saved orders found.{RESET}")
        return None

    with open(purchases_file, 'r') as f:
        purchases = json.load(f)

    if not purchases:
        print(f"{RED}No saved orders found.{RESET}")
        return None

    purchases.sort(key=lambda x: x.get('created_at', ''), reverse=True)

    print(f"\n{CYAN}Saved Orders:{RESET}")
    
    table_data = []
    for idx, purchase in enumerate(purchases, 1):
        created_at = purchase.get('created_at', 'N/A')
        if created_at != 'N/A':
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S")
        
        table_data.append([
            idx,
            purchase['video_id'],
            purchase['state'],
            purchase.get('amount', 'N/A'),
            f"{float(purchase.get('amount', 0)) / 0.9:.5f}",
            purchase.get('miner_hotkey', 'N/A')[:5] + '...' + purchase.get('miner_hotkey', 'N/A')[-5:],
            purchase['block_hash'][:5] + '...' + purchase['block_hash'][-5:],
            created_at
        ])
    
    headers = ["#", "Video ID", "Purchase State", "Cost (TAO)", "Estimated Reward (TAO)", "Purchasing Hotkey", "Block Hash", "Purchase Date"]
    table = tabulate(table_data, headers=headers, tablefmt="pretty")
    
    print(table)
    return purchases

def select_order_for_verification():
    purchases = display_saved_orders()

    while True:
        if purchases:
            print(f"*** NOTE: A purchase is finalized when the purchase state is 'verified'. ***")
            choice = input(f"{CYAN}Enter the number of the order to verify, 'm' for manual input, or 'n' to cancel: {RESET}").lower()
        else:
            choice = 'm'

        if choice == 'n':
            return None, None, None
        elif choice == 'm':
            video_id = input(f"{CYAN}Enter video ID: {RESET}")
            miner_hotkey = input(f"{CYAN}Enter miner hotkey: {RESET}")
            block_hash = input(f"{CYAN}Enter block hash: {RESET}")
            return video_id, miner_hotkey, block_hash
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(purchases):
                selected = purchases[idx]
                return selected['video_id'], selected.get('miner_hotkey', ''), selected['block_hash']
            else:
                print(f"{RED}Invalid selection. Please try again.{RESET}")
        else:
            print(f"{RED}Invalid input. Please try again.{RESET}")

def select_order_for_full_display(purchases):
    while True:
        choice = input(f"{CYAN}Enter the number of the order to see full details, or 'n' to return to menu: {RESET}").lower()
        
        if choice == 'n':
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(purchases):
                selected = purchases[idx]
                # Display full details
                print(f"\n{CYAN}Order Details:{RESET}")
                print(f"Video ID: {selected['video_id']}")
                print(f"Purchase State: {selected['state']}")
                print(f"Cost (TAO): {selected.get('amount', 'N/A')}")
                print(f"Estimated Reward (TAO): {float(selected.get('amount', 0)) / 0.9:.5f}")
                print(f"Purchasing Hotkey: {selected.get('miner_hotkey', 'N/A')}")
                print(f"Block Hash: {selected['block_hash']}")
                print(f"Purchase Date: {selected.get('created_at', 'N/A')}")
                return
            else:
                print(f"{RED}Invalid selection. Please try again.{RESET}")
        else:
            print(f"{RED}Invalid input. Please try again.{RESET}")

def verify_purchase(video_id=None, miner_hotkey=None, block_hash=None, miner_hotkey_signature=None):
    if miner_hotkey_signature is None:
        wallet, name, hotkey_name, path = get_wallet()
        miner_hotkey, miner_hotkey_signature = get_auth_headers(wallet)

    if not all([video_id, miner_hotkey, block_hash]):
        video_id, miner_hotkey, block_hash = select_order_for_verification()
        if not all([video_id, miner_hotkey, block_hash]):
            print(f"{CYAN}Verification cancelled.{RESET}")
            return

    print(f"Verifying purchase for video {video_id} on block hash {block_hash} ...")

    retries = 3
    for attempt in range(retries):
        try:
            verify_response = requests.post(
                API_BASE + "/api/focus/verify-purchase",
                auth=(miner_hotkey, miner_hotkey_signature),
                json={"miner_hotkey": miner_hotkey, "video_id": video_id, "block_hash": block_hash},
                headers={"Content-Type": "application/json"},
                timeout=90
            )
            print(f"Purchase verification response for video {video_id}:", verify_response.text)
            if verify_response.status_code == 200:
                print(f"{GREEN}Purchase verified successfully!{RESET}")
                save_purchase_info(video_id, miner_hotkey, block_hash, "verified")
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

def save_purchase_info(video_id, hotkey, block_hash, state, amount=None):
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
            purchase['miner_hotkey'] = hotkey
            purchase['block_hash'] = block_hash
            if amount is not None:
                purchase['amount'] = amount
            break
    else:
        # If the video_id doesn't exist, create a new entry
        new_purchase = {
            "video_id": video_id,
            "miner_hotkey": hotkey,
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
        print("1. View + Purchase Focus Videos")
        print("2. Manually Purchase Focus Video")
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
            else:
                print(f"\n{RED}No videos available for purchase at this time.{RESET}")
        elif choice == '2':
            purchase_video()
        elif choice == '3':
            verify_purchase()
        elif choice == '4':
            purchases = display_saved_orders()
            select_order_for_full_display(purchases)
        elif choice == '5':
            print(f"{GREEN}Thank you for using the OMEGA Focus Videos Purchase System. Goodbye!{RESET}")
            break
        else:
            print(f"{RED}Invalid choice. Please try again.{RESET}")

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting.")
        reset_terminal()
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        reset_terminal()
        sys.exit(1)
