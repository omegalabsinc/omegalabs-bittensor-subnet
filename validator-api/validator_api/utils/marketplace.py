import time
from typing import Tuple, Dict
import requests
import bittensor as bt
from validator_api.config import (
    NETWORK, BT_TESTNET, NETUID, FOCUS_REWARDS_PERCENT, FIXED_ALPHA_USD_ESTIMATE,
    BOOSTED_TASKS_PERCENTAGE,
)
from validator_api.utils import run_with_retries, run_async
from validator_api.database.models.focus_video_record import TaskType
from validator_api.database.crud.focusvideo import FocusVideoRecord
from validator_api.database import get_db_context
from sqlalchemy import select, func
from datetime import datetime, timedelta

TASK_TYPE_MAP = {
    TaskType.USER: 1 - BOOSTED_TASKS_PERCENTAGE,
    TaskType.BOOSTED: BOOSTED_TASKS_PERCENTAGE,
}


async def get_subtensor() -> bt.subtensor:
    def _internal() -> bt.subtensor:
        return bt.subtensor(network=NETWORK)
    return await run_with_retries(_internal)


async def get_tao_price() -> float:
    return await run_with_retries(
        lambda: float(
            requests.get(
                "https://api.kucoin.com/api/v1/market/stats?symbol=TAO-USDT"
            ).json()["data"]["last"]
        )
    )

# Global cache for max focus alpha
max_focus_alpha_per_day_cache = {
    'value': None,
    'timestamp': 0
}

CACHE_DURATION = 30 * 60  # 30 minutes in seconds

async def get_max_focus_alpha_per_day() -> float:
    """
    https://docs.bittensor.com/dynamic-tao/emission
    """
    global max_focus_alpha_per_day_cache
    current_time = time.time()

    if max_focus_alpha_per_day_cache['value'] is not None and current_time - max_focus_alpha_per_day_cache['timestamp'] < CACHE_DURATION:
        return max_focus_alpha_per_day_cache['value']

    # If cache is invalid or empty, recalculate
    subtensor = await get_subtensor()
    
    def _internal_sync():
        subnet = subtensor.subnet(netuid=NETUID)
        alpha_emission_per_block = subnet.alpha_out_emission.tao
        miner_alpha_emission_per_block = alpha_emission_per_block * 0.41
        miner_alpha_emission_per_tempo = miner_alpha_emission_per_block * 360
        miner_alpha_emission_per_day = miner_alpha_emission_per_tempo * 20
        max_focus_alpha_per_day = miner_alpha_emission_per_day * FOCUS_REWARDS_PERCENT
        # if NETWORK == BT_TESTNET:
        #     max_focus_alpha_per_day = max(200, max_focus_alpha_per_day)
        #     # max_focus_alpha_per_day = max(1800, max_focus_alpha_per_day)  # 92 alpha per day cuz 3.12% emissions * 20% budget
        return max_focus_alpha_per_day

    async def _internal_async() -> float:
        return await run_async(_internal_sync)

    max_focus_alpha_per_day = await run_with_retries(_internal_async)
    # print(f"max_focus_alpha_per_day: {max_focus_alpha_per_day}")
    # Update cache
    max_focus_alpha_per_day_cache['value'] = max_focus_alpha_per_day
    max_focus_alpha_per_day_cache['timestamp'] = current_time

    return max_focus_alpha_per_day


async def get_fixed_reward_pool_alpha() -> float:
    """
    the amount of alpha in the fixed reward pool (marketplace tasks) per day
    get the sum of the marketplace tasks alpha per day
    return 0 if no marketplace tasks have been completed in the last 24 hours
    """
    async with get_db_context() as db:
        twenty_four_hours_ago = datetime.utcnow() - timedelta(days=1)
        query = select(
            func.sum(FocusVideoRecord.earned_reward_alpha)
        ).where(
            FocusVideoRecord.task_type == TaskType.MARKETPLACE.value,
            FocusVideoRecord.updated_at >= twenty_four_hours_ago
        )
        result = await db.execute(query)
        return result.scalar() or 0.0


async def get_variable_reward_pool_alpha() -> float:
    """
    the amount of alpha that is available for users in the variable reward pool per day
    we want to limit the amount of focus tao that can be purchased to 90% of the max focus tao so miners can make some profit
    """
    max_miner_focus_alpha = await get_max_focus_alpha_per_day()
    max_reward_focus_alpha = max_miner_focus_alpha * 0.9
    # todo: subtract fixed reward pool alpha
    fixed_reward_pool_alpha = await get_fixed_reward_pool_alpha()
    variable_reward_pool_alpha = max_reward_focus_alpha - fixed_reward_pool_alpha
    return variable_reward_pool_alpha


def get_dollars_available_today(max_focus_alpha: float) -> float:
    """ Use a fixed ΩTAO - USD estimate to keep consistent for the sake of miner rewards """
    return max_focus_alpha * FIXED_ALPHA_USD_ESTIMATE

def get_max_focus_points_available_today(max_focus_alpha: float) -> float:
    # 1 point = 1 dollar
    return int(get_dollars_available_today(max_focus_alpha))

MAX_TASK_REWARD_TAO = 0.1
