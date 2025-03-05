import bittensor as bt
from bittensor_wallet import Wallet
from bittensor import Balance, tao
import time
import random


def main():
    subtensor = bt.subtensor(network="test")

    while True:
        subnet = subtensor.subnet(netuid=96)
        print(
            f"Tempo: {subnet.tempo} Block: {subtensor.block} alpha_out_emission: {subnet.alpha_out_emission.tao} alpha_out: {subnet.alpha_out.tao} "
        )

        sleep_time = 60 + random.uniform(-30, 30)
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
