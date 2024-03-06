from typing import List, Tuple

from pytube import Search, YouTube

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH
from omega import video_utils


TWENTY_MINUTES = 1200


def get_description(yt: YouTube, video_path: str) -> str:
    """
    Get / generate the description of a video from the YouTube API.
    
    Miner TODO: Implement logic to get / generate the most relevant and information-rich
    description of a video from the YouTube API.
    """
    description = yt.title
    if yt.description:
        description += f"\n\n{yt.description}"
    if yt.keywords:
        description += f"\n\nKeywords: {', '.join(yt.keywords)}"
    return description


def get_relevant_timestamps(query: str, yt: YouTube, video_path: str) -> Tuple[int, int]:
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
    video_metas = []
    s = Search(query)
    try:
        while len(video_metas) < num_videos:
            for result in s.results:
                download_path = video_utils.download_video(
                    result.video_id,
                    start=0,
                    end=min(result.length, TWENTY_MINUTES)  # download the first 20 minutes at most
                )
                if download_path:
                    clip_path = None
                    try:
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
                if len(video_metas) == num_videos:
                    break
            s.get_next_results()

    except Exception as e:
        print(f"Error searching for videos: {e}")
    
    return video_metas
