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
from omega.imagebind_wrapper import ImageBind
from omega.miner_utils import search_and_diarize_youtube_videos, search_and_embed_youtube_videos
from omega.augment import LocalLLMAugment, OpenAIAugment, NoAugment
from omega.utils.config import QueryAugment
from omega.constants import VALIDATOR_TIMEOUT, VALIDATOR_TIMEOUT_AUDIO
from omega.diarization_pipeline import CustomDiarizationPipeline

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.
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
        
        
        self.diarization_pipeline = CustomDiarizationPipeline(
            overlap_detection_model_id = "tezuesh/overlapped-speech-detection",
            diarization_model_id="tezuesh/diarization",
            # device="cuda"
        )
        self.imagebind = ImageBind(v2=True)

    async def forward_videos(
        self, synapse: omega.protocol.Videos
    ) :
        # Scrape Youtube videos
        bt.logging.info(f"Received scraping request: {synapse.num_videos} videos for query '{synapse.query}'")
        
        start = time.time()
        
        synapse.video_metadata = search_and_embed_youtube_videos(
            self.augment(synapse.query), synapse.num_videos, self.imagebind
        )
            
        time_elapsed = time.time() - start
            
        if len(synapse.video_metadata) == synapse.num_videos and time_elapsed < VALIDATOR_TIMEOUT:
            bt.logging.info(f"–––––– SCRAPING SUCCEEDED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")
        else:
            bt.logging.error(f"–––––– SCRAPING FAILED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")


        return synapse
    
    async def forward_audios(
        self, synapse: omega.protocol.Audios
    ) -> omega.protocol.Audios:
        bt.logging.info(f"Received youtube audio scraping and diarization request: {synapse.num_audios} audios for query '{synapse.query}'")
        
        start = time.time()
        
        synapse.audio_metadata = search_and_diarize_youtube_videos(
            self.augment(synapse.query), synapse.num_audios, self.diarization_pipeline, self.imagebind
        )
        
        time_elapsed = time.time() - start
            
        if len(synapse.audio_metadata) == synapse.num_audios and time_elapsed < VALIDATOR_TIMEOUT_AUDIO:
            bt.logging.info(f"–––––– SCRAPING SUCCEEDED: Scraped {len(synapse.audio_metadata)}/{synapse.num_audios} audios in {time_elapsed} seconds.")
        else:
            bt.logging.error(f"–––––– SCRAPING FAILED: Scraped {len(synapse.audio_metadata)}/{synapse.num_audios} audios in {time_elapsed} seconds.")
        return synapse

    async def blacklist(
        self, synapse: bt.Synapse
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
    
    async def blacklist_videos(
        self, synapse: omega.protocol.Videos
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)
    
    async def blacklist_audios(
        self, synapse: omega.protocol.Audios
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority(self, synapse: bt) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Audios): The synapse object that contains metadata about the incoming request.

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

    async def priority_videos(
        self, synapse: omega.protocol.Videos
    ) -> float:
        return await self.priority(synapse)
    
    async def priority_audios(
        self, synapse: omega.protocol.Audios
    ) -> float:
        return await self.priority(synapse)
    
    def save_state(self):
        """
        We define this function to avoid printing out the log message in the BaseNeuron class
        that says `save_state() not implemented`.
        """
        pass

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
