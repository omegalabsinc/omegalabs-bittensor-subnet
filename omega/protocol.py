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

import typing
import json

import bittensor as bt
from pydantic import BaseModel


class VideoMetadata(BaseModel):
    """
    A model class representing YouTube video metadata.
    """
    video_id: str
    description: str
    views: int
    start_time: int
    end_time: int
    video_emb: typing.List[float]
    audio_emb: typing.List[float]
    description_emb: typing.List[float]

    def __repr_args__(self):
        parent_args = super().__repr_args__()
        exclude_args = ['video_emb', 'audio_emb', 'description_emb']
        return (
            [(a, v) for a, v in parent_args if a not in exclude_args] +
            [(a, ["..."]) for a in exclude_args]
        )


class Videos(bt.Synapse):
    """
    A synapse class representing a video scraping request and response.

    Attributes:
    - query: the input query for which to find relevant videos
    - num_videos: the number of videos to return
    - video_metadata: a list of video metadata objects
    """

    query: str
    num_videos: int
    video_metadata: typing.Optional[typing.List[VideoMetadata]] = None

    def deserialize(self) -> typing.List[VideoMetadata]:
        assert self.video_metadata is not None
        return self.video_metadata

    def to_serializable_dict(self, input_synapse: "Videos") -> dict:
        """
        Dumps the Videos object to a serializable dict, but makes sure to use input properties from
        the input_synapse, while taking the non-null output property video_metadata from the
        response (self).
        """
        json_str = self.replace_with_input(input_synapse).json(
            include={"query", "num_videos", "video_metadata"})
        return json.loads(json_str)

    def replace_with_input(self, input_synapse: "Videos") -> "Videos":
        """
        Replaces the query and num_videos of current synapse with the given input synapse.
        """
        return Videos(
            query=input_synapse.query,
            num_videos=input_synapse.num_videos,
            video_metadata=self.video_metadata[:input_synapse.num_videos],
            axon=self.axon
        )
