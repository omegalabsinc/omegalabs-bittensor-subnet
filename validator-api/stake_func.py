import asyncio
import bittensor as bt


NETWORK = "finney"
miner_hotkey = "5DZP1sfhucEzHkyvpLUjXFG4FWJZ15NK5PmmhWuHWSmmjauB"

NETWORK = "test"
miner_hotkey = "5DaNytPVo6uFZFr2f9pZ6ck2gczNyYebLgrYZoFuccPS6qMi"

subtensor = bt.subtensor(network=NETWORK)
TEMP_NETUID = 24 if NETWORK == "finney" else 96
subnet = subtensor.subnet(netuid=TEMP_NETUID)
miner_coldkey = subtensor.get_hotkey_owner(miner_hotkey)
breakpoint()

from validator_api.config import NETWORK, NETUID, STAKE_HOTKEY
from validator_api.utils.wallet import get_transaction_from_block_hash

from bittensor_cli.src.commands.stake import add as add_stake


async def stake(user_email: str, amount: float):
    # stake {amount} from {user_email}'s wallet to NETUID on NETWORK with STAKE_HOTKEY
    wallet = None  # TODO: get wallet from user_email
    add_stake.stake_add(
        wallet=wallet,
        subtensor=subtensor,
        netuid=NETUID,
        amount=amount,
        prompt=False,
        safe_staking=False,
        rate_tolerance=0.05,
        allow_partial_stake=False,
    )

async def main():
    # transfers = await get_transaction_from_block_hash(subtensor, miner_coldkey, "0xde4f146fd8dfca247031fef4758dc0b5942bcf3d2c61771900f1104b3102c8cc")
    # print(transfers)
    # await stake("salman@omega-labs.ai", 0.001)
    pass


if __name__ == "__main__":
    asyncio.run(main())
