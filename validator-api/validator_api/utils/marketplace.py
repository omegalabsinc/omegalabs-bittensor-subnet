import time
from typing import Tuple, Dict
import requests
import bittensor as bt
from validator_api.config import (
    NETWORK, BT_TESTNET, NETUID, FOCUS_REWARDS_PERCENT, FIXED_TAO_USD_ESTIMATE,
    BOOSTED_TASKS_PERCENTAGE,
)
from validator_api.utils import run_with_retries, run_async
from validator_api.database.models.focus_video_record import TaskType

TASK_TYPE_MAP = {
    TaskType.USER: 1 - BOOSTED_TASKS_PERCENTAGE,
    TaskType.BOOSTED: BOOSTED_TASKS_PERCENTAGE,
}


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

MAX_TASK_REWARD_TAO = 0.1

def estimate_tao(
    score: float,
    duration: int,  # in seconds
    task_type: TaskType,
    max_focus_tao: float,
    focus_points_last_24_hours: Dict[TaskType, float],
) -> float:
    """
    Calculate reward for a focus video based on its score, duration and max rewards available
    """
    # Get all tasks from last hour
    total_focus_points = focus_points_last_24_hours[task_type]

    # Add current task's contribution
    current_focus_points = score * duration
    score_duration_with_current = total_focus_points + current_focus_points

    # Calculate portion for current task
    task_portion = current_focus_points / score_duration_with_current if score_duration_with_current > 0 else 0

    # Get max rewards per hour and calculate final reward
    task_percentage = TASK_TYPE_MAP[task_type]
    max_rewards = max_focus_tao * task_percentage
    reward = task_portion * max_rewards
    reward = min(reward, MAX_TASK_REWARD_TAO)

    return reward
