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


import copy
import torch
import asyncio
import argparse
import os
import threading
import datetime as dt
import bittensor as bt
from datetime import datetime
from subprocess import Popen, PIPE

from typing import List
from traceback import print_exception

from omega.base.neuron import BaseNeuron
from omega.mock import MockDendrite
from omega.utils.config import add_validator_args
from omega.constants import FOCUS_REWARDS_PERCENT, AUDIO_REWARDS_PERCENT


class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )
        self.focus_scores = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )

        self.audio_score_arr = torch.zeros(
            self.metagraph.n, dtype=torch.float32, device=self.device
        )
        
        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        if self.config.neuron.auto_update:
            bt.logging.info("Auto update enabled.")
        else:
            bt.logging.info("Auto update disabled.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.last_update_check = datetime.now()
        self.update_check_interval = 1800  # 30 minutes

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def is_git_latest(self) -> bool:
        p = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        current_commit = out.decode().strip()
        p = Popen(['git', 'ls-remote', 'origin', 'HEAD'], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if err:
            return False
        latest_commit = out.decode().split()[0]
        bt.logging.info(f'Current commit: {current_commit}, Latest commit: {latest_commit}')
        return current_commit == latest_commit

    def should_restart(self) -> bool:
        # Check if enough time has elapsed since the last update check, if not assume we are up to date.
        if (datetime.now() - self.last_update_check).seconds < self.update_check_interval:
            return False
        
        self.last_update_check = datetime.now()

        return not self.is_git_latest()

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                if self.config.neuron.auto_update and self.should_restart():
                    bt.logging.info(f'Validator is out of date, quitting to restart.')
                    raise KeyboardInterrupt

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

                # Check if we should start a new wandb run.
                if not self.config.wandb.off and self.successfully_started_wandb:
                    if (dt.datetime.now() - self.wandb_run_start) >= dt.timedelta(
                        days=1
                    ):
                        bt.logging.info(
                            "Current wandb run is more than 1 day old. Starting a new run."
                        )
                        self.wandb_run.finish()
                        self.new_wandb_run()

                # Check if we should reload the topics.
                if (dt.datetime.now() - self.load_topics_start) >= dt.timedelta(
                    hours=1
                ):
                    bt.logging.info("Reloading topics after 1 hour.")
                    self.all_topics = self.load_topics()
                    self.load_topics_start = dt.datetime.now()

                # Check if we should reload the focus videos rewards percentage.
                if (dt.datetime.now() - self.load_focus_rewards_start) >= dt.timedelta(
                    hours=1
                ):
                    bt.logging.info("Reloading focus videos rewards percent after 1 hour.")
                    self.FOCUS_REWARDS_PERCENT = self.load_focus_rewards_percent()
                    self.AUDIO_REWARDS_PERCENT = AUDIO_REWARDS_PERCENT
                    self.YOUTUBE_REWARDS_PERCENT = 1.0 - self.FOCUS_REWARDS_PERCENT - self.AUDIO_REWARDS_PERCENT
                    self.load_focus_rewards_start = dt.datetime.now()

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(
                print_exception(type(err), err, err.__traceback__)
            )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def pad_tensors(self, tensor_a, tensor_b, tensor_c):
        # Ensure both tensors are on the same device
        device = tensor_a.device
        tensor_b = tensor_b.to(device)
        tensor_c = tensor_c.to(device)
        max_size = max(tensor_a.size(0), tensor_b.size(0), tensor_c.size(0))
        if tensor_a.size(0) < max_size:
            padding = torch.zeros(max_size - tensor_a.size(0), device=device)
            tensor_a = torch.cat((tensor_a, padding))
            print("tensor a was padded")
        if tensor_b.size(0) < max_size:
            padding = torch.zeros(max_size - tensor_b.size(0), device=device)
            tensor_b = torch.cat((tensor_b, padding))
            print("tensor b was padded")
        if tensor_c.size(0) < max_size:
            padding = torch.zeros(max_size - tensor_c.size(0), device=device)
            tensor_c = torch.cat((tensor_c, padding))
            print("tensor c was padded")

        return tensor_a, tensor_b, tensor_c

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if torch.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        self.scores, self.focus_scores, self.audio_score_arr = self.pad_tensors(self.scores, self.focus_scores, self.audio_score_arr)

        bt.logging.debug(f"Normalizing scores with YOUTUBE_REWARDS_PERCENT: {self.YOUTUBE_REWARDS_PERCENT}, FOCUS_REWARDS_PERCENT: {self.FOCUS_REWARDS_PERCENT}, AUDIO_REWARDS_PERCENT: {self.AUDIO_REWARDS_PERCENT}")
        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Normalize the youtube rewards and scale by the percentage.
        raw_weights_youtube = torch.nn.functional.normalize(self.scores, p=1, dim=0) * self.YOUTUBE_REWARDS_PERCENT
        # Normalize the focus rewards and scale by the percentage.
        raw_weights_focus = torch.nn.functional.normalize(self.focus_scores, p=1, dim=0) * self.FOCUS_REWARDS_PERCENT
        # Normalize the audio rewards and scale by the percentage.
        raw_weights_audio = torch.nn.functional.normalize(self.audio_score_arr, p=1, dim=0) * self.AUDIO_REWARDS_PERCENT

        # Combine the youtube and focus rewards.
        raw_weights = raw_weights_youtube + raw_weights_focus + raw_weights_audio

        bt.logging.debug("raw_weights_youtube", raw_weights_youtube)
        bt.logging.debug("raw_weights_focus", raw_weights_focus)
        bt.logging.debug("raw_weights_audio", raw_weights_audio)
        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", self.metagraph.uids.to("cpu"))
        if raw_weights.shape[0] > self.metagraph.uids.shape[0]:
            bt.logging.warning("More raw_weights than metagraph uids, truncating raw_weights.")
        raw_weights = raw_weights[:self.metagraph.uids.shape[0]]
        # Process the raw weights to final_weights via subtensor limitations.
        try:
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids.to("cpu"),
                weights=raw_weights.to("cpu"),
                netuid=self.config.netuid,
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )
            bt.logging.debug("processed_weights", processed_weights)
            bt.logging.debug("processed_weight_uids", processed_weight_uids)
        except Exception as e:
            bt.logging.error(f"Failed to process weights with exception: {e}, skipping set_weights this time")
            return

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Set the weights on chain via our subtensor connection.
        result, result_msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error(f"set_weights failed with message: {result_msg}")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced
                self.focus_scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            # Update the size of the moving average scores.
            new_moving_average = torch.zeros((self.metagraph.n)).to(
                self.device
            )
            min_len = min(len(self.hotkeys), len(self.scores))
            new_moving_average[:min_len] = self.scores[:min_len]
            self.scores = new_moving_average
            self.focus_scores = new_moving_average
            self.audio_score_arr = new_moving_average

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        if len(rewards) == 0:
            bt.logging.debug("self.update_scores: Rewards are empty, returning early")
            return

        if len(uids) == 0:
            bt.logging.debug("self.update_scores: Miner UIDs list is empty, returning early")
            return

        if len(rewards) != len(uids):
            bt.logging.exception("self.update_scores: Rewards are not the same size as UIDs list (THIS SHOULD NEVER HAPPEN!)")
            return

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.scores.to(self.device).scatter(
            0, uids_tensor.to(self.device), rewards.to(self.device)
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.scores.to(self.device)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def update_focus_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the focus video scores based on the rewards received from the miners."""

        # Check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)

        # Check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: torch.FloatTensor = self.focus_scores.to(self.device).scatter(
            0, uids_tensor.to(self.device), rewards.to(self.device)
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        self.focus_scores: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.focus_scores.to(self.device)
        bt.logging.debug(f"Updated moving avg focus_scores: {self.focus_scores}")

    def update_audio_scores(self, rewards: torch.FloatTensor, uids: List[int]):
        """Performs exponential moving average on the audio scores based on the rewards received from the miners."""

        # check if rewards contains NaN values.
        if torch.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = torch.nan_to_num(rewards, 0)
        
        # check if `uids` is already a tensor and clone it to avoid the warning.
        if isinstance(uids, torch.Tensor):
            uids_tensor = uids.clone().detach()
        else:
            uids_tensor = torch.tensor(uids).to(self.device)
        
        # compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [metagraph.n]
        scattered_rewards: torch.FloatTensor = self.audio_score_arr.to(self.device).scatter(
            0, uids_tensor.to(self.device), rewards.to(self.device)
        ).to(self.device)
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # update scores with rewards produced by this step.
        # shape: [metagraph.n]
        alpha: float = self.config.neuron.moving_average_alpha
        self.audio_score_arr: torch.FloatTensor = alpha * scattered_rewards + (
            1 - alpha
        ) * self.audio_score_arr.to(self.device)
        bt.logging.debug(f"Updated moving avg audio_scores: {self.audio_score_arr}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        torch.save(
            {
                "step": self.step,
                "scores": self.scores,
                "focus_scores": self.focus_scores,
                "audio_score_arr": self.audio_score_arr,
                "hotkeys": self.hotkeys,
            },
            self.config.neuron.full_path + "/state.pt",
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        if not os.path.exists(self.config.neuron.full_path + "/state.pt"):
            bt.logging.warning("No saved state found")
            return

        # Load the state of the validator from file.
        state = torch.load(self.config.neuron.full_path + "/state.pt", map_location=self.device)
        self.step = state["step"]
        self.scores = state["scores"]
        if "focus_scores" in state:
            self.focus_scores = state["focus_scores"]
        else:
            state["focus_scores"] = torch.zeros(
                self.metagraph.n, dtype=torch.float32, device=self.device
            )
        
        if "audio_score_arr" in state:
            self.audio_score_arr = state["audio_score_arr"]
        else:
            state["audio_score_arr"] = torch.zeros(
                self.metagraph.n, dtype=torch.float32, device=self.device
            )
        self.hotkeys = state["hotkeys"]
