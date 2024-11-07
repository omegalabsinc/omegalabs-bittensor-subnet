import os
import time
from typing import List, Tuple

import bittensor as bt

from omega.protocol import VideoMetadata, AudioMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES, MAX_AUDIO_LENGTH_SECONDS, MIN_AUDIO_LENGTH_SECONDS
from omega import video_utils
from omega.diarization_pipeline import CustomDiarizationPipeline

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


def get_relevant_timestamps(query: str, yt: video_utils.YoutubeDL, video_path: str, max_length: int) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = min(yt.length, max_length)
    return start_time, end_time


def search_and_embed_youtube_videos(query: str, num_videos: int, imagebind: ImageBind) -> List[VideoMetadata]:
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
            download_path = video_utils.download_youtube_video(
                result.video_id,
                start=0,
                end=min(result.length, FIVE_MINUTES)  # download the first 5 minutes at most
            )
            if download_path:
                clip_path = None
                try:
                    result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                    bt.logging.info(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    start, end = get_relevant_timestamps(query, result, download_path, max_length=MAX_VIDEO_LENGTH)
                    description = get_description(result, download_path)
                    clip_path = video_utils.clip_video(download_path.name, start, end)
                    bt.logging.info(f"Clip video path: {clip_path}")
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
            if len(video_metas) == num_videos:
                break

    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")

    return video_metas




def search_and_diarize_youtube_videos(query: str, num_videos: int, diarization_pipeline: CustomDiarizationPipeline, imagebind: ImageBind) -> List[AudioMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of AudioMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[AudioMetadata]: A list of AudioMetadata objects representing the search results.
    """
    results = video_utils.search_videos(query, max_results=int(num_videos * 1.5))
    bt.logging.info(f"Audio Results: {results}")
    audio_metas = []
    try:
        # take the first N that we need
        for result in results:
            start_time_loop = time.time()
            download_path = video_utils.download_youtube_video(
                result.video_id,
                start=0,
                end=min(result.length, MAX_AUDIO_LENGTH_SECONDS)  # download the first 5 minutes at most
            )
            if download_path:
                clip_path = None
                try:
                    result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                    bt.logging.info(f"Downloaded audio {result.video_id} ({min(result.length, MAX_AUDIO_LENGTH_SECONDS)}) in {time.time() - start_time_loop} seconds")
                    start, end = get_relevant_timestamps(query, result, download_path, max_length=MAX_AUDIO_LENGTH_SECONDS)
                    # bt.logging.info(f"Audio Start: {start}, End: {end}")
                    description = get_description(result, download_path)
                    audio_array, sr = video_utils.get_audio_array(download_path.name)
                    dataframe = diarization_pipeline.process(audio_array, sr)
                    diar_timestamps_start = dataframe["start"]
                    diar_timestamps_end = dataframe["end"]
                    diar_speakers = dataframe["speakers"]
                    clip_path = video_utils.clip_video(download_path.name, start, end)
                    bt.logging.info(f"Clip video path: {clip_path}")
                    embeddings = imagebind.embed([description], [clip_path])
                    # bt.logging.info(f"Embeddings: {type(embeddings)}, audio_emb: {type(embeddings.audio[0])}, audio_array: {type(audio_array)} {audio_array.shape}, sr: {sr}, diar_timestamps_start: {type(diar_timestamps_start)}, diar_timestamps_end: {type(diar_timestamps_end)}, diar_speakers: {type(diar_speakers)}")
                    # bt.logging.info(f"Audio duration: {end - start}, actual length: {result.length}")
                    # bt.logging.info("Diarization Dataframe: ", dataframe)
                    audio_metas.append(AudioMetadata(
                        video_id=result.video_id,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                        audio_emb=embeddings.audio[0].tolist(),
                        sampling_rate=sr,
                        audio_array=audio_array.tolist(),
                        diar_timestamps_start=diar_timestamps_start,
                        diar_timestamps_end=diar_timestamps_end,
                        diar_speakers=diar_speakers,
                    ))
                finally:
                    download_path.close()
                    if clip_path:
                        clip_path.close()
            if len(audio_metas) == num_videos:
                break
            end_time_loop = time.time()
            bt.logging.info(f"Audio Time taken for loop: {end_time_loop - start_time_loop}")

    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")

    return audio_metas

