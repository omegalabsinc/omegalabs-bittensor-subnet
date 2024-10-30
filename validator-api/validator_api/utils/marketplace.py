import time
from typing import Tuple
import requests
import bittensor as bt
from validator_api.config import NETWORK, BT_TESTNET, NETUID, FOCUS_REWARDS_PERCENT, MAX_FOCUS_POINTS_PER_HOUR, FIXED_TAO_USD_ESTIMATE
from validator_api.utils import run_with_retries, run_async

async def get_subtensor_and_metagraph() -> Tuple[bt.subtensor, bt.metagraph]:

    def _internal() -> Tuple[bt.subtensor, bt.metagraph]:
        subtensor = bt.subtensor(network=NETWORK)
        metagraph = bt.metagraph(NETUID)
        return subtensor, metagraph

    return await run_with_retries(_internal)


async def get_tao_price() -> float:
    return await run_with_retries(
        lambda: float(
            requests.get(
                "https://api.kucoin.com/api/v1/market/stats?symbol=TAO-USDT"
            ).json()["data"]["last"]
        )
    )

# Global cache for max focus TAO
max_focus_tao_cache = {
    'value': None,
    'timestamp': 0
}

CACHE_DURATION = 30 * 60  # 30 minutes in seconds

async def get_max_focus_tao() -> float:
    global max_focus_tao_cache
    current_time = time.time()

    # Check if cached data is still valid
    if max_focus_tao_cache['value'] is not None and current_time - max_focus_tao_cache['timestamp'] < CACHE_DURATION:
        return max_focus_tao_cache['value']

    # If cache is invalid or empty, recalculate
    subtensor, metagraph = await get_subtensor_and_metagraph()

    def _internal_sync():
        current_block = metagraph.block.item()
        metagraph.sync(current_block - 10, lite=False, subtensor=subtensor)

        total_vali_and_miner_emission = 0
        for uid in metagraph.uids.tolist():
            total_vali_and_miner_emission += metagraph.emission[uid]

        total_miner_emission = total_vali_and_miner_emission / 2  # per tempo
        total_miner_emission_per_day = total_miner_emission * 20  # 20 tempo intervals per day
        max_focus_tao = total_miner_emission_per_day * FOCUS_REWARDS_PERCENT

        if NETWORK == BT_TESTNET:
            max_focus_tao = max(2, max_focus_tao)
            # max_focus_tao = max(18, max_focus_tao)  # 92 tao per day cuz 3.12% emissions * 20% budget

        return max_focus_tao

    async def _internal_async() -> float:
        return await run_async(_internal_sync)

    max_focus_tao = await run_with_retries(_internal_async)

    # Update cache
    max_focus_tao_cache['value'] = max_focus_tao
    max_focus_tao_cache['timestamp'] = current_time

    return max_focus_tao

def get_dollars_available_today(max_focus_tao: float) -> float:
    """ Use a fixed TAO - USD estimate to keep consistent for the sake of miner rewards """
    return max_focus_tao * FIXED_TAO_USD_ESTIMATE

def get_max_focus_points_available_today(max_focus_tao: float) -> float:
    # 1 point = 1 dollar
    return int(get_dollars_available_today(max_focus_tao))

def estimate_tao(productive_score: float, video_duration: float, max_focus_tao: float):
    try:
        max_focus_points_for_video = video_duration / 3600 * MAX_FOCUS_POINTS_PER_HOUR
        actual_focus_points_for_video = max_focus_points_for_video * float(productive_score)
        max_focus_points_available_today = get_max_focus_points_available_today(max_focus_tao)
        tao = min(actual_focus_points_for_video / max_focus_points_available_today, 1.0) * float(max_focus_tao)
        return round(tao, 5)
    except Exception as e:
        print(e)
        return 0
