import os
import time
from typing import List, Tuple

import bittensor as bt

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES
from omega import video_utils
import concurrent.futures
import moviepy.editor as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI

    OPENAI_CLIENT = OpenAI()
else:
    OPENAI_CLIENT = None


def get_description(yt: video_utils.YoutubeDL, video_path: str) -> str:
    """
    Get / generate the description of a video from the YouTube API.
    
    Miner TODO: Implement logic to get / generate the most relevant and information-rich
    description of a video from the YouTube API.
    """
    description = yt.title
    if yt.description:
        description += f"\n\n{yt.description}"
    return description


def get_relevant_timestamps(query: str, yt: video_utils.YoutubeDL, video_path: str) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = min(yt.length, MAX_VIDEO_LENGTH)
    return start_time, end_time


def search_and_embed_videos(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    # fetch more videos than we need
    results = video_utils.search_videos(query, max_results=int(num_videos * 1.5))
    video_metas = []
    try:
        # take the first N that we need
        for result in results:
            start = time.time()
            download_path = video_utils.download_video(
                result.video_id,
                start=5,
                end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
            )
            if download_path:
                clip_path = None
                try:
                    result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                    bt.logging.info(
                        f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    start, end = get_relevant_timestamps(query, result, download_path)
                    description = get_description(result, download_path)
                    clip_path = video_utils.clip_video(download_path.name, start, end)
                    embeddings = imagebind.embed([description], [clip_path])
                    video_metas.append(VideoMetadata(
                        video_id=result.video_id,
                        description=description,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                        video_emb=embeddings.video[0].tolist(),
                        audio_emb=embeddings.audio[0].tolist(),
                        description_emb=embeddings.description[0].tolist(),
                    ))
                finally:
                    download_path.close()
                    if clip_path:
                        clip_path.close()
                        os.remove(clip_path.name)
            if len(video_metas) == num_videos:
                break

    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")

    return video_metas


def search_and_embed_videos_parallel(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
    results = video_utils.search_videos(query, max_results=int(num_videos * 1.5))
    video_metas = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_video, result, query, imagebind) for result in results]
        for future in concurrent.futures.as_completed(futures):
            video_meta = future.result()
            if video_meta:
                video_metas.append(video_meta)
            if len(video_metas) == num_videos:
                break
    return video_metas


def process_video(result, query: str, imagebind: ImageBind):
    try:
        start = time.time()
        download_path = video_utils.download_video(
            result.video_id,
            start=5,
            end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
        )
        if download_path:
            clip_path = None
            try:
                result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                bt.logging.info(
                    f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                start, end = get_relevant_timestamps(query, result, download_path)
                description = get_description(result, download_path)
                clip_path = video_utils.clip_video(download_path.name, start, end)
                embeddings = imagebind.embed([description], [clip_path])
                return VideoMetadata(
                    video_id=result.video_id,
                    description=description,
                    views=result.views,
                    start_time=start,
                    end_time=end,
                    video_emb=embeddings.video[0].tolist(),
                    audio_emb=embeddings.audio[0].tolist(),
                    description_emb=embeddings.description[0].tolist(),
                )
            finally:
                download_path.close()
                if clip_path:
                    clip_path.close()
                    os.remove(clip_path.name)
                    bt.logging.error(f"Remove clip path {clip_path.name} with youtube id {result.video_id}")
    except Exception as e:
        bt.logging.error(f"Error processing video {result.video_id}: {e}")
        return None
