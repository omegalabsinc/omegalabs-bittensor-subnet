import bittensor as bt
import aiohttp
import time
from validator_api.utils import run_with_retries, run_async
from typing import List


# Global cache for TAO/USD rate
tao_usd_cache = {
    'rate': None,
    'timestamp': 0
}

CACHE_DURATION = 30 * 60  # 30 minutes in seconds

async def get_tao_usd_rate() -> float:
    global tao_usd_cache
    current_time = time.time()

    # Check if cached data is still valid
    if tao_usd_cache['rate'] is not None and current_time - tao_usd_cache['timestamp'] < CACHE_DURATION:
        return tao_usd_cache['rate']

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://taostats.io/data.json') as response:
                if response.status == 200:
                    data = await response.json()
                    rate = float(data[0]['price'])

                    # Update cache
                    tao_usd_cache['rate'] = rate
                    tao_usd_cache['timestamp'] = current_time

                    return rate
                else:
                    print(f"Failed to fetch TAO/USD rate. Status code: {response.status}")
                    return tao_usd_cache['rate']
    except Exception as e:
        print(f"Error fetching TAO/USD rate: {str(e)}")
        return tao_usd_cache['rate']

async def check_wallet_tao_balance(wallet_key: str, subtensor_network: str) -> float:
    def _internal_sync() -> float:
        subtensor = bt.subtensor(network=subtensor_network)
        balance = subtensor.get_balance(wallet_key).tao
        return balance

    async def _internal_async() -> float:
        return await run_async(_internal_sync)

    return await run_with_retries(_internal_async)


API_URL = "https://api.subquery.network/sq/TaoStats/bittensor-indexer"
MAX_TXN = 50
GRAPHQL_QUERY = """
query ($first: Int!, $after: Cursor, $filter: TransferFilter, $order: [TransfersOrderBy!]!) {
    transfers(first: $first, after: $after, filter: $filter, orderBy: $order) {
        nodes {
            id
            from
            to
            amount
            extrinsicId
            blockNumber
        }
        pageInfo {
            endCursor
            hasNextPage
            hasPreviousPage
        }
        totalCount
    }
}
"""

async def get_transaction_from_block_hash(subtensor, wallet_address: str, block_hash: str) -> List[dict]:
    """Get all transfers associated with the provided wallet address and block_hash."""
    transactions = []
    divisor = 1e9
    
    block = subtensor.substrate.get_block(block_hash)
    block_num = block['header']['number']

    for extrinsic in block['extrinsics']:
        extrinsic = extrinsic.value
        if 'call' in extrinsic and extrinsic['call']['call_module'] == 'Balances':
            if extrinsic['call']['call_function'] in ['transfer', 'transfer_allow_death']:
                sender = extrinsic.get('address', 'Unknown')
                recipient = extrinsic['call']['call_args'][0]['value']
                amount = int(extrinsic['call']['call_args'][1]['value'])

                if sender == wallet_address or recipient == wallet_address:
                    transactions.append({
                        'id': extrinsic['extrinsic_hash'],
                        'from': sender,
                        'to': recipient,
                        'amount': amount / divisor,
                        # the Id is not actually supposed to be the hash, but we'll let it fly
                        # for now cause all we need is a unique identifier, which the hash is
                        'extrinsicId': extrinsic['extrinsic_hash'],
                        'blockNumber': block_num
                    })

    return transactions[::-1]
