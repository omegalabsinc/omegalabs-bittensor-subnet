# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Omega Labs, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
# Set USE_TORCH=1 environment variable to use torch instead of numpy
os.environ["USE_TORCH"] = "1"

import time
import json
import typing
import requests
import asyncio
import bittensor as bt

# Bittensor Miner Template:
import omega

from omega.base.miner import BaseMinerNeuron
from omega.imagebind_wrapper import ImageBind, IMAGEBIND_VERSION
from omega.miner_utils import search_and_embed_youtube_videos, embed_focus_videos
from omega.augment import LocalLLMAugment, OpenAIAugment, NoAugment
from omega.utils.config import QueryAugment
from omega.constants import VALIDATOR_TIMEOUT


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        query_augment_type = QueryAugment(self.config.neuron.query_augment)
        if query_augment_type == QueryAugment.NoAugment:
            self.augment = NoAugment(device=self.config.neuron.device)
        elif query_augment_type == QueryAugment.LocalLLMAugment:
            self.augment = LocalLLMAugment(device=self.config.neuron.device)
        elif query_augment_type == QueryAugment.OpenAIAugment:
            self.augment = OpenAIAugment(device=self.config.neuron.device)
        else:
            raise ValueError("Invalid query augment")
        
        self.imagebind = ImageBind()
        self.imagebind_v1 = ImageBind(disable_lora=True)

        self.focus_videos_api = (
            #"https://dev-focus-api.omegatron.ai/"
            "http://localhost:8000/"
            if self.config.subtensor.network == "test" else
            "https://focus-api.omegatron.ai/"
        )

    async def forward(
        self, synapse: omega.protocol.Videos
    ) -> omega.protocol.Videos:
        
        # Scrape Youtube videos
        bt.logging.info(f"Received scraping request: {synapse.num_videos} videos for query '{synapse.query}'")
        
        start = time.time()
        if synapse.vali_imagebind_version is not None and synapse.vali_imagebind_version == IMAGEBIND_VERSION:
            synapse.video_metadata = search_and_embed_youtube_videos(
                self.augment(synapse.query), synapse.num_videos, self.imagebind
            )
            synapse.miner_imagebind_version = IMAGEBIND_VERSION
        else:
            synapse.video_metadata = search_and_embed_youtube_videos(
                self.augment(synapse.query), synapse.num_videos, self.imagebind_v1
            )
            synapse.miner_imagebind_version = "1.0"
        
        time_elapsed = time.time() - start
        
        if len(synapse.video_metadata) == synapse.num_videos and time_elapsed < VALIDATOR_TIMEOUT:
            bt.logging.info(f"–––––– SCRAPING SUCCEEDED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")
        else:
            bt.logging.error(f"–––––– SCRAPING FAILED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")

        synapse.focus_metadata = []
        if self.config.neuron.focus_videos:
            # Retrieve marketplace video list
            response = requests.post(url=f'{self.focus_videos_api}/market/purchased_list',
                                    data=json.dumps(self.wallet.hotkey.ss58_address))
            
            video_data = response.json()
            if response.status_code == 200:
                bt.logging.warning(f'{len(video_data)} - {video_data}')
                if len(video_data) > 0:
                    bt.logging.info(f"Purchased FocusVideo list: {video_data} sending: {video_data[:synapse.num_focus_videos]}")
                    synapse.focus_metadata = embed_focus_videos(synapse.query, video_data[:synapse.num_focus_videos], self.imagebind)
                    bt.logging.info(f"focus metadata {synapse.focus_metadata}")
                else:
                    bt.logging.info(f"Failed to retrieve focus video list: No videos found.")
            else:
                bt.logging.info(f"Failed to retrieve market list: {response.status_code} - {response.reason}")

        return synapse

    async def blacklist(
        self, synapse: omega.protocol.Videos
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Videos): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"
        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        stake = self.metagraph.S[uid].item()
        if self.config.blacklist.validator_min_stake and stake < self.config.blacklist.validator_min_stake:
            bt.logging.warning(f"Blacklisting request from {synapse.dendrite.hotkey} [uid={uid}], not enough stake -- {stake}")
            return True, "Stake below minimum"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: omega.protocol.Videos) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Videos): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def save_state(self):
        """
        We define this function to avoid printing out the log message in the BaseNeuron class
        that says `save_state() not implemented`.
        """
        pass
    
    
    def get_video_list(self):
        response = requests.post(f"{self.focus_videos_api}/market/get_list")
        if response.status_code == 200:
            return response.json()
        else:
            bt.logging.warning(f"Fetching available video list failed: {response.status_code} - {response.reason}")
    
    def purchase_focus_video(self, video_id: str):
        response = requests.post(f"{self.focus_videos_api}/market/purchase", data=json.dumps({
            'video_id': video_id,
            'miner_hotkey': self.wallet.hotkey.ss58_address
        }))
        res_data = response.json()
        if response.status_code == 200 and res_data['status'] == 'success':
            bt.logging.info(f'Purchased new video: <{res_data["address"]}>')
            return res_data['address']
        else:
            bt.logging.warning(f'Purchasing failed. {response.status_code} - {response.reason}')
            return None
    
    async def check_consume_and_commit(self):
        if not self.config.neuron.focus_videos:
            return
        
        try:
            sub = bt.subtensor(config = self.config)
            commitStr = sub.get_commitment(self.config.netuid, self.uid)
            newIpfsUrlResponse = requests.post(url=f'{self.focus_videos_api}/ipfs_url/get',
                                    data=json.dumps(self.wallet.hotkey.ss58_address))
            newIpfsUrl = newIpfsUrlResponse.json().get('url')                    
            if not commitStr == newIpfsUrl:
                sub.commit(wallet=self.wallet, netuid=self.config.netuid, data=newIpfsUrl)
                bt.logging.info(f"commited new url {newIpfsUrl}")
        except Exception as e:
            bt.logging.error(e)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        last_action_time = time.time()
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
            
            current_time = time.time()
            if current_time - last_action_time >= 30:
                asyncio.run(miner.check_consume_and_commit())
                last_action_time = current_time
            
