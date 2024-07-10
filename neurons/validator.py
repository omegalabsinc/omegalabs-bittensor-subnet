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
from typing import List, Tuple, Optional, BinaryIO
from pydantic import ValidationError
import datetime as dt
import os
import random
import json
import traceback
import requests

# Bittensor
import bittensor as bt
import torch
import torch.nn.functional as F
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
    FAKE_VIDEO_PUNISHMENT,
    RANDOM_DESCRIPTION_PENALTY
)
from omega import video_utils
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async
import omega.imagebind_desc_mlp as imagebind_desc_mlp

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
                bt.logging.exception("WANDB_API_KEY not found. Set it with `export WANDB_API_KEY=<your API key>`. Alternatively, you can disable W&B with --wandb.off, but it is strongly recommended to run with W&B enabled.")
        else:
            bt.logging.warning("Running with --wandb.off. It is strongly recommended to run with W&B enabled.")

        api_root = (
            "https://dev-validator.api.omega-labs.ai"
            if self.config.subtensor.network == "test" else
            "https://validator.api.omega-labs.ai"
        )
        self.validation_endpoint = f"{api_root}/api/validate"
        self.proxy_endpoint = f"{api_root}/api/get_proxy"
        self.novelty_scores_endpoint = f"{api_root}/api/get_pinecone_novelty"
        self.upload_video_metadata_endpoint = f"{api_root}/api/upload_video_metadata"
        self.num_videos = 8
        self.client_timeout_seconds = VALIDATOR_TIMEOUT + VALIDATOR_TIMEOUT_MARGIN

        # load topics from topics URL (CSV) or fallback to local topics file
        self.load_topics_start = dt.datetime.now()
        self.all_topics = self.load_topics()

        self.imagebind = None
        if not self.config.neuron.decentralization.off:
            if torch.cuda.is_available():
                bt.logging.info(f"Running with decentralization enabled, thank you Bittensor Validator!")
                self.decentralization = True
                self.imagebind = ImageBind()
            else:
                bt.logging.warning(f"Attempting to run decentralization, but no GPU found. Please see min_compute.yml for minimum resource requirements.")
                self.decentralization = False
        else:
            bt.logging.warning("Running with --decentralization.off. It is strongly recommended to run with decentralization enabled.")
            self.decentralization = False

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

    def load_topics(self):
        # get topics from CSV URL and load them into our topics list
        try:
            response = requests.get(self.config.topics_url)
            response.raise_for_status()
            # split the response text into a list of topics and trim any whitespace
            all_topics = [line.strip() for line in response.text.split("\n")]
            bt.logging.info(f"Loaded {len(all_topics)} topics from {self.config.topics_url}")
        except Exception as e:
            bt.logging.error(f"Error loading topics from URL {self.config.topics_url}: {e}")
            traceback.print_exc()
            bt.logging.info(f"Using fallback topics from {self.config.topics_path}")
            all_topics = [line.strip() for line in open(self.config.topics_path) if line.strip()]
            bt.logging.info(f"Loaded {len(all_topics)} topics from {self.config.topics_path}")
        return all_topics

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

        # The dendrite client queries the network.
        query = random.choice(self.all_topics)
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
            # Check if this validator is running decentralization
            if not self.decentralization:
                # if not, use validator API get_rewards system
                rewards_list = await self.get_rewards(input_synapse=input_synapse, responses=finished_responses)
            else:
                # if so, use decentralization logic with local GPU
                rewards_list = await self.handle_checks_and_rewards(input_synapse=input_synapse, responses=finished_responses)
        except Exception as e:
            bt.logging.error(f"Error in handle_checks_and_rewards: {e}")
            traceback.print_exc()
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


    def metadata_check(self, metadata: List[VideoMetadata]) -> List[VideoMetadata]:
        return [
            video_metadata for video_metadata in metadata
            if (
                video_metadata.end_time - video_metadata.start_time <= MAX_VIDEO_LENGTH and
                video_metadata.end_time - video_metadata.start_time >= MIN_VIDEO_LENGTH
            )
        ]
    
    def filter_embeddings(self, embeddings: Embeddings, is_too_similar: List[bool]) -> Embeddings:
        """Filter the embeddings based on whether they are too similar to the query."""
        is_too_similar = torch.tensor(is_too_similar)
        embeddings.video = embeddings.video[~is_too_similar]
        embeddings.audio = embeddings.audio[~is_too_similar]
        embeddings.description = embeddings.description[~is_too_similar]
        return embeddings

    async def deduplicate_videos(self, embeddings: Embeddings) -> Videos:
        # return a list of booleans where True means the corresponding video is a duplicate i.e. is_similar
        video_tensor = embeddings.video
        num_videos = video_tensor.shape[0]
        cossim = CosineSimilarity(dim=1)
        is_similar = []
        for i in range(num_videos):
            similarity_score = cossim(video_tensor[[i]], video_tensor[i + 1:])
            has_duplicates = (similarity_score > SIMILARITY_THRESHOLD).any()
            is_similar.append(has_duplicates.item())
        
        return is_similar
    
    def is_similar(self, emb_1: torch.Tensor, emb_2: List[float]) -> bool:
        return F.cosine_similarity(
            emb_1,
            torch.tensor(emb_2, device=emb_1.device).unsqueeze(0)
        ) > SIMILARITY_THRESHOLD

    def strict_is_similar(self, emb_1: torch.Tensor, emb_2: List[float]) -> bool:
        return torch.allclose(emb_1, torch.tensor(emb_2, device=emb_1.device), atol=1e-4)
    
    async def get_random_video(self, metadata: List[VideoMetadata], check_video: bool) -> Optional[Tuple[VideoMetadata, Optional[BinaryIO]]]:
        if not check_video:
            random_metadata = random.choice(metadata)
            return random_metadata, None

        random_video = None
        metadata_copy = [v for v in metadata]  # list shallow copy
        while random_video is None and len(metadata_copy) > 0:
            idx = random.randint(0, len(metadata_copy) - 1)
            random_metadata = metadata_copy.pop(idx)
            proxy_url = await self.get_proxy_url()
            if proxy_url is None:
                bt.logging.info("Issue getting proxy_url from API, not using proxy. Attempting download for random_video check")
            else:
                bt.logging.info("Got proxy_url from API. Attempting download for random_video check")
            try:
                async with DOWNLOAD_SEMAPHORE:
                    random_video = await asyncio.wait_for(run_async(
                        video_utils.download_video,
                        random_metadata.video_id,
                        random_metadata.start_time,
                        random_metadata.end_time,
                        proxy=proxy_url
                    ), timeout=VIDEO_DOWNLOAD_TIMEOUT)
            except video_utils.IPBlockedException:
                # IP is blocked, cannot download video, check description only
                bt.logging.warning("WARNING: IP is blocked, cannot download video, checking description only")
                return random_metadata, None
            except video_utils.FakeVideoException:
                bt.logging.warning(f"WARNING: Video {random_metadata.video_id} is fake, punishing miner")
                return None
            except asyncio.TimeoutError:
                continue

        # IP is not blocked, video is not fake, but video download failed for some reason. We don't
        # know why it failed so we won't punish the miner, but we will check the description only.
        if random_video is None:
            return random_metadata, None

        return random_metadata, random_video
    
    async def random_check(self, random_meta_and_vid: List[VideoMetadata]) -> bool:
        random_metadata, random_video = random_meta_and_vid

        if random_video is None:
            desc_embeddings = self.imagebind.embed_text([random_metadata.description])
            is_similar_ = self.is_similar(desc_embeddings, random_metadata.description_emb)
            strict_is_similar_ = self.strict_is_similar(desc_embeddings, random_metadata.description_emb)
            bt.logging.info(f"Description similarity: {is_similar_}, strict description similarity: {strict_is_similar_}")
            return is_similar_

        # Video downloaded, check all embeddings
        embeddings = self.imagebind.embed([random_metadata.description], [random_video])
        is_similar_ = (
            self.is_similar(embeddings.video, random_metadata.video_emb) and
            self.is_similar(embeddings.audio, random_metadata.audio_emb) and
            self.is_similar(embeddings.description, random_metadata.description_emb)
        )
        strict_is_similar_ = (
            self.strict_is_similar(embeddings.video, random_metadata.video_emb) and
            self.strict_is_similar(embeddings.audio, random_metadata.audio_emb) and
            self.strict_is_similar(embeddings.description, random_metadata.description_emb)
        )
        bt.logging.debug(f"Total similarity: {is_similar_}, strict total similarity: {strict_is_similar_}")
        return is_similar_
    
    def compute_novelty_score_among_batch(self, emb: Embeddings) -> List[float]:
        video_tensor = emb.video
        num_videos = video_tensor.shape[0]
        novelty_scores = []
        for i in range(num_videos - 1):
            similarity_score = F.cosine_similarity(video_tensor[[i]], video_tensor[i + 1:]).max()
            novelty_scores.append(1 - similarity_score.item())
        novelty_scores.append(1.0)  # last video is 100% novel
        return novelty_scores

    async def async_zero() -> None:
        return 0

    # algorithm for computing final novelty score
    def compute_final_novelty_score(self, base_novelty_scores: List[float]) -> float:
        is_too_similar = [score < DIFFERENCE_THRESHOLD for score in base_novelty_scores]
        novelty_score = sum([
            score for score, is_too_similar
            in zip(base_novelty_scores, is_too_similar) if not is_too_similar
        ])
        return novelty_score

    # Main function that handles checks and scoring for a single response (Videos) from a miner
    async def check_videos_and_calculate_rewards(
        self,
        input_synapse: Videos,
        videos: Videos,
    ) -> torch.FloatTensor:
        
        try:
            # return minimum score if no videos were found in video_metadata
            if len(videos.video_metadata) == 0:
                return MIN_SCORE

            # check video_ids for fake videos
            if any(not video_utils.is_valid_id(video.video_id) for video in videos.video_metadata):
                return FAKE_VIDEO_PUNISHMENT

            # check and filter duplicate metadata
            metadata = self.metadata_check(videos.video_metadata)[:input_synapse.num_videos]
            if len(metadata) < len(videos.video_metadata):
                bt.logging.info(f"Filtered {len(videos.video_metadata)} videos down to {len(metadata)} videos")

            # if randomly tripped, flag our random check to pull a video from miner's submissions
            check_video = CHECK_PROBABILITY > random.random()
            
            # pull a random video and/or description only
            random_meta_and_vid = await self.get_random_video(metadata, check_video)
            if random_meta_and_vid is None:
                return FAKE_VIDEO_PUNISHMENT

            # execute the random check on metadata and video
            async with GPU_SEMAPHORE:
                passed_check = await self.random_check(random_meta_and_vid)
                # punish miner if not passing
                if not passed_check:
                    return FAKE_VIDEO_PUNISHMENT
                # create query embeddings for relevance scoring
                query_emb = await self.imagebind.embed_text_async([videos.query])

            # generate embeddings
            embeddings = Embeddings(
                video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(self.imagebind.device),
                audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(self.imagebind.device),
                description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(self.imagebind.device),
            )

            # check and deduplicate videos based on embedding similarity checks. We do this because we're not uploading to pinecone first.
            metadata_is_similar = await self.deduplicate_videos(embeddings)
            metadata = [metadata for metadata, too_similar in zip(metadata, metadata_is_similar) if not too_similar]
            embeddings = self.filter_embeddings(embeddings, metadata_is_similar)
            if len(metadata) < len(videos.video_metadata):
                bt.logging.info(f"Deduplicated {len(videos.video_metadata)} videos down to {len(metadata)} videos")

            # return minimum score if no unique videos were found
            if len(metadata) == 0:
                return MIN_SCORE
            
            # first get local novelty scores
            local_novelty_scores = self.compute_novelty_score_among_batch(embeddings)
            #bt.logging.debug(f"local_novelty_scores: {local_novelty_scores}")
            # second get the novelty scores from the validator api if not already too similar
            embeddings_to_check = [
                (embedding, metadata)
                for embedding, local_score, metadata in zip(embeddings.video, local_novelty_scores, metadata)
                if local_score >= DIFFERENCE_THRESHOLD
            ]
            # If there are embeddings to check, call get_novelty_scores once
            if embeddings_to_check:
                embeddings_to_check, metadata_to_check = zip(*embeddings_to_check)
                global_novelty_scores = await self.get_novelty_scores(metadata_to_check)
            else:
                # If no embeddings to check, return an empty list or appropriate default value
                global_novelty_scores = []

            if global_novelty_scores is None or len(global_novelty_scores) == 0:
                bt.logging.error("Issue retrieving global novelty scores, returning None.")
                return None
            #bt.logging.debug(f"global_novelty_scores: {global_novelty_scores}")
            
            # calculate true novelty scores between local and global
            true_novelty_scores = [
                min(local_score, global_score) for local_score, global_score
                in zip(local_novelty_scores, global_novelty_scores)
            ]
            #bt.logging.debug(f"true_novelty_scores: {true_novelty_scores}")

            pre_filter_metadata_length = len(metadata)
            # check scores from index for being too similar
            is_too_similar = [score < DIFFERENCE_THRESHOLD for score in true_novelty_scores]
            # filter out metadata too similar
            metadata = [metadata for metadata, too_similar in zip(metadata, is_too_similar) if not too_similar]
            # filter out embeddings too similar
            embeddings = self.filter_embeddings(embeddings, is_too_similar)
            if len(metadata) < pre_filter_metadata_length:
                bt.logging.info(f"Filtering {pre_filter_metadata_length} videos down to {len(metadata)} videos that are too similar to videos in our index.")

            # return minimum score if no unique videos were found
            if len(metadata) == 0:
                return MIN_SCORE

            # compute our final novelty score - 6/3/24: NO LONGER USING NOVELTY SCORE IN SCORING
            #novelty_score = self.compute_final_novelty_score(true_novelty_scores)
            
            # Compute relevance scores
            description_relevance_scores = F.cosine_similarity(
                embeddings.video, embeddings.description
            ).tolist()
            query_relevance_scores = F.cosine_similarity(
                embeddings.video, query_emb
            ).tolist()

            # Compute if there are penalties for random descriptions
            description_mlp_results = [imagebind_desc_mlp.get_desc_embedding_score(embedding) for embedding in embeddings.description]
            description_mlp_scores = []
            # Apply penalties and store the penalized scores
            for desc_score, desc_mlp_score in zip(description_relevance_scores, description_mlp_results):
                if desc_mlp_score == 4 or desc_mlp_score == 5:
                    # score of 4 or 5 is "good", reward with 40% or 50% description relevance score boost, respectfully
                    rewarded_score = (desc_mlp_score * 0.1) * desc_score
                    description_mlp_scores.append(rewarded_score)
                elif desc_mlp_score == 2:
                    # score of 2 is "poor", penalize with 50% description relevance score penalty
                    description_mlp_scores.append(desc_score * -0.5)
                elif desc_mlp_score == 1:
                    # score of 1 is "bad", penalize full description relevance score penalty
                    description_mlp_scores.append(desc_score * -1)
                else:
                    # score of 3 is "OK", no reward or penalty
                    description_mlp_scores.append(0)

            # Aggregate scores
            score = (
                sum(description_relevance_scores) +
                sum(query_relevance_scores) +
                sum(description_mlp_scores)
            ) / 2 / videos.num_videos
            
            # Set final score, giving minimum if necessary
            score = max(score, MIN_SCORE)

            # Log all our scores
            bt.logging.info(f'''
                is_unique: {[not is_sim for is_sim in is_too_similar]},
                description_relevance_scores: {description_relevance_scores},
                query_relevance_scores: {query_relevance_scores},
                description_penalties: {description_penalties},
                score: {score}
            ''')

            # Upload our final results to API endpoint for index and dataset insertion. Include leaderboard statistics
            miner_hotkey = videos.axon.hotkey
            upload_result = await self.upload_video_metadata(metadata, description_relevance_scores, query_relevance_scores, videos.query, None, score, miner_hotkey)
            if upload_result:
                bt.logging.info("Uploading of video metadata successful.")
            else:
                bt.logging.error("Issue uploading video metadata.")

            return score

        except Exception as e:
            bt.logging.error(f"Error in check_videos_and_calculate_rewards: {e}")
            return None

    # Get all the reward results by iteratively calling your reward() function.
    async def handle_checks_and_rewards(
        self,
        input_synapse: Videos,
        responses: List[Videos],
    ) -> torch.FloatTensor:
        
        rewards = await asyncio.gather(*[
            self.check_videos_and_calculate_rewards(
                input_synapse,
                response.replace_with_input(input_synapse), # replace with input properties from input_synapse
            )
            for response in responses
        ])
        return rewards
        
    
    async def upload_video_metadata(
        self, 
        metadata: List[VideoMetadata], 
        description_relevance_scores: List[float], 
        query_relevance_scores: List[float], 
        query: str, 
        novelty_score: float, 
        score: float, 
        miner_hotkey: str
    ) -> bool:
        """
        Queries the validator api to get novelty scores for supplied videos. 
        Returns a list of float novelty scores for each video after deduplicating.

        Returns:
        - List[float]: The novelty scores for the miner's videos.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                # Serialize the list of VideoMetadata
                serialized_metadata = [item.dict() for item in metadata]
                # Construct the JSON payload
                payload = {
                    "metadata": serialized_metadata,
                    "description_relevance_scores": description_relevance_scores,
                    "query_relevance_scores": query_relevance_scores,
                    "topic_query": query,
                    "novelty_score": novelty_score,
                    "total_score": score,
                    "miner_hotkey": miner_hotkey
                }

                async with session.post(
                    self.upload_video_metadata_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            return True
        except Exception as e:
            bt.logging.debug(f"Error trying upload_video_metadata_endpoint: {e}")
            return False

    async def get_novelty_scores(self, metadata: List[VideoMetadata]) -> List[float]:
        """
        Queries the validator api to get novelty scores for supplied videos. 
        Returns a list of float novelty scores for each video after deduplicating.

        Returns:
        - List[float]: The novelty scores for the miner's videos.
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                # Serialize the list of VideoMetadata
                serialized_metadata = [item.dict() for item in metadata]

                async with session.post(
                    self.novelty_scores_endpoint,
                    auth=BasicAuth(hotkey, signature),
                    json=serialized_metadata,
                ) as response:
                    response.raise_for_status()
                    novelty_scores = await response.json()
            return novelty_scores
        
        except Exception as e:
            bt.logging.debug(f"Error trying novelty_scores_endpoint: {e}")
            return None
        
    async def get_proxy_url(self) -> str:
        """
        Queries the validator api to get a random proxy URL.

        Returns:
        - str: A proxy URL
        """
        keypair = self.dendrite.keypair
        hotkey = keypair.ss58_address
        signature = f"0x{keypair.sign(hotkey).hex()}"
        try:
            async with ClientSession() as session:
                async with session.post(
                    self.proxy_endpoint,
                    auth=BasicAuth(hotkey, signature),
                ) as response:
                    response.raise_for_status()
                    proxy_url = await response.json()
            return proxy_url
        except Exception as e:
            bt.logging.debug(f"Error trying proxy_endpoint: {e}")
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
