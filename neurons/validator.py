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


from aiohttp import ClientSession, BasicAuth
import asyncio
from typing import List
import datetime as dt
import os
import random

# Bittensor
import bittensor as bt
import torch
from torch.nn import CosineSimilarity
import wandb

# Bittensor Validator Template:
from omega.utils.uids import get_random_uids
from omega.protocol import Videos, VideoMetadata
from omega.constants import (
    VALIDATOR_TIMEOUT, 
    VALIDATOR_TIMEOUT_MARGIN, 
    MAX_VIDEO_LENGTH, 
    MIN_VIDEO_LENGTH,
    CHECK_PROBABILITY,
    DIFFERENCE_THRESHOLD, 
    SIMILARITY_THRESHOLD, 
    VIDEO_DOWNLOAD_TIMEOUT, 
    MIN_SCORE, 
    FAKE_VIDEO_PUNISHMENT
)
from omega import video_utils
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async

# import base validator class which takes care of most of the boilerplate
from omega.base.validator import BaseValidatorNeuron

NO_RESPONSE_MINIMUM = 0.005
GPU_SEMAPHORE = asyncio.Semaphore(1)
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(5)

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        if not self.config.wandb.off:
            if os.getenv("WANDB_API_KEY"):
                self.new_wandb_run()
            else:
                bt.logging.exception("WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.of, but it is strongly recommended to run with W&B enabled.")
        else:
            bt.logging.warning("Running with --wandb.off. It is strongly recommended to run with W&B enabled.")

        api_root = (
            "https://dev-validator.api.omega-labs.ai"
            if self.config.subtensor.network == "test" else
            "https://validator.api.omega-labs.ai"
        )
        self.topics_endpoint = f"{api_root}/api/topic"
        self.validation_endpoint = f"{api_root}/api/validate"
        self.proxy_endpoint = f"{api_root}/api/get_proxy"
        self.novelty_scores_endpoint = f"{api_root}/api/get_novelty_scores"
        self.num_videos = 8
        self.client_timeout_seconds = VALIDATOR_TIMEOUT + VALIDATOR_TIMEOUT_MARGIN

    def new_wandb_run(self):
        # Shoutout SN13 for the wandb snippet!
        """Creates a new wandb run to save information to."""
        # Create a unique run id for this run.
        now = dt.datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project="omega-sn24-validator-logs",
            entity="omega-labs",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "validator",
            },
            allow_val_change=True,
            anonymous="allow",
        )

        bt.logging.debug(f"Started a new wandb run: {name}")

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        """
        The forward function is called by the validator every time step.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

        """
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        if len(miner_uids) == 0:
            bt.logging.info("No miners available")
            return

        try:
            async with ClientSession() as session:
                async with session.get(self.topics_endpoint) as response:
                    response.raise_for_status()
                    query = await response.json()
        except Exception as e:
            bt.logging.error(f"Error in get_topics: {e}")
            return

        # The dendrite client queries the network.
        bt.logging.info(f"Sending query '{query}' to miners {miner_uids}")
        input_synapse = Videos(query=query, num_videos=self.num_videos)
        axons = [self.metagraph.axons[uid] for uid in miner_uids]
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=axons,
            synapse=input_synapse,
            deserialize=False,
            timeout=self.client_timeout_seconds,
        )

        working_miner_uids = []
        finished_responses = []

        for response in responses:
            if response.video_metadata is None or not response.axon or not response.axon.hotkey:
                continue

            uid = [uid for uid, axon in zip(miner_uids, axons) if axon.hotkey == response.axon.hotkey][0]
            working_miner_uids.append(uid)
            finished_responses.append(response)

        if len(working_miner_uids) == 0:
            bt.logging.info("No miner responses available")
            return

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        # Adjust the scores based on responses from miners.
        try:
            #rewards_list = await self.get_rewards(input_synapse=input_synapse, responses=finished_responses)
            rewards_list = await self.check_videos_and_calculate_rewards(videos=input_synapse, responses=finished_responses)
        except Exception as e:
            bt.logging.error(f"Error in get_rewards: {e}")
            return

        # give reward to all miners who responded and had a non-null reward
        rewards = []
        reward_uids = []
        for r, r_uid in zip(rewards_list, working_miner_uids):
            if r is not None:
                rewards.append(r)
                reward_uids.append(r_uid)
        rewards = torch.FloatTensor(rewards).to(self.device)
        self.update_scores(rewards, reward_uids)
        
        # give min reward to miners who didn't respond
        bad_miner_uids = [uid for uid in miner_uids if uid not in working_miner_uids]
        penalty_tensor = torch.FloatTensor([NO_RESPONSE_MINIMUM] * len(bad_miner_uids)).to(self.device)
        self.update_scores(penalty_tensor, bad_miner_uids)

        for reward, miner_uid in zip(rewards, reward_uids):
            bt.logging.info(f"Rewarding miner={miner_uid} with reward={reward}")
        
        for penalty, miner_uid in zip(penalty_tensor, bad_miner_uids):
            bt.logging.info(f"Penalizing miner={miner_uid} with penalty={penalty}")


    def metadata_check(metadata: List[VideoMetadata]) -> List[VideoMetadata]:
        return [
            video_metadata for video_metadata in metadata
            if (
                video_metadata.end_time - video_metadata.start_time <= MAX_VIDEO_LENGTH and
                video_metadata.end_time - video_metadata.start_time >= MIN_VIDEO_LENGTH
            )
        ]
    
    async def deduplicate_videos(metadata: VideoMetadata) -> Videos:
        # return a list of booleans where True means the corresponding video is a duplicate i.e. is_similar
        embeddings = Embeddings(
            video=torch.stack([torch.tensor(v.video_emb) for v in metadata]),
            audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]),
            description=torch.stack([torch.tensor(v.description_emb) for v in metadata]),
        )

        video_tensor = embeddings.video
        num_videos = video_tensor.shape[0]
        cossim = CosineSimilarity(dim=1)
        is_similar = []
        for i in range(num_videos):
            similarity_score = cossim(video_tensor[[i]], video_tensor[i + 1:])
            has_duplicates = (similarity_score > SIMILARITY_THRESHOLD).any()
            is_similar.append(has_duplicates.item())
        
        return is_similar

    async def check_videos_and_calculate_rewards(
        self,
        videos: Videos,
        responses: List[Videos],
    ) -> torch.FloatTensor:
        
        # check video_ids for fake videos
        if any(not video_utils.is_valid_id(video.video_id) for video in videos.video_metadata):
            return {"score": FAKE_VIDEO_PUNISHMENT}

        # check and filter duplicate metadata
        metadata = self.metadata_check(videos.video_metadata)[:videos.num_videos]
        print(f"Filtered {len(videos.video_metadata)} videos down to {len(metadata)} videos")

        # check and deduplicate videos based on embedding similarity checks. We do this because we're not uploading to pinecone first.
        metadata_is_similar = self.deduplicate_videos(metadata)
        print(f"Deduplicated {len(videos.video_metadata)} videos down to {len([m for m in metadata_is_similar if not m])} videos")

        if len(metadata) == 0:
            return {"score": MIN_SCORE}

        check_video = CHECK_PROBABILITY > random.random()
        random_meta_and_vid = await get_random_video(metadata, check_video)
        if random_meta_and_vid is None:
            return {"score": FAKE_VIDEO_PUNISHMENT}

        async with GPU_SEMAPHORE:
            passed_check = await random_check(random_meta_and_vid, imagebind)
            if not passed_check:
                return {"score": FAKE_VIDEO_PUNISHMENT}
            query_emb = await imagebind.embed_text_async([videos.query])

        # first get the novelty_scores from the validator api
        novelty_scores = await asyncio.gather(*[
            self.get_novelty_scores(
                data,
                response,
            )
            for data in metadata
        ])

        rewards = []
        return rewards
        
    
    async def get_novelty_scores(self, input_synapse: Videos, response: Videos) -> List[float]:
        """
        Queries the validator api to get novelty scores for supplied videos. 
        Returns a list of float novelty scores for each video after deduplicating.

        Returns:
        - List[float: The novelty scores for the miner's videos.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                async with session.post(
                    self.novelty_scores_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=response.to_serializable_dict(input_synapse),
                ) as response:
                    response.raise_for_status()
                    novelty_scores = await response.json()
            return novelty_scores
        except Exception as e:
            bt.logging.debug(f"Error trying novelty_scores_endpoint: {e}")
            return None

    async def reward(self, input_synapse: Videos, response: Videos) -> float:
        """
        Reward the miner response to the query. This method returns a reward
        value for the miner, which is used to update the miner's score.

        Returns:
        - float: The reward value for the miner.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                async with session.post(
                    self.validation_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=response.to_serializable_dict(input_synapse),
                ) as response:
                    response.raise_for_status()
                    score = await response.json()
            return score
        except Exception as e:
            bt.logging.debug(f"Error in reward: {e}")
            return None

    async def get_rewards(
        self,
        input_synapse: Videos,
        responses: List[Videos],
    ) -> torch.FloatTensor:
        """
        Returns a tensor of rewards for the given query and responses.
        """
        # Get all the reward results by iteratively calling your reward() function.
        rewards = await asyncio.gather(*[
            self.reward(
                input_synapse,
                response,
            )
            for response in responses
        ])
        return rewards


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Validator().run()
