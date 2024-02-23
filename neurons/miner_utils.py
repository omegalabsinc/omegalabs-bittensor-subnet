import os
from typing import List, Optional, BinaryIO, Tuple
import tempfile

from pytube import Search, YouTube

from omega.protocol import VideoMetadata


def download_video(yt: YouTube, tempfile: BinaryIO) -> bool:
    """
    Download a video from YouTube to a given file object.

    Args:
        yt (YouTube): The YouTube object representing the video to download.
        tempfile (BinaryIO): The file object to write the video to.

    Returns:
        bool: True if the video was successfully downloaded, False otherwise.
    """
    try:
        # Select the highest resolution available that is AVC codec (MPEG-4)
        streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc()
        avc_streams = [stream for stream in streams if stream.video_codec.startswith("avc")]
        video = avc_streams[0]

        # Download the video
        video.download(output_path=os.path.dirname(tempfile.name), filename=os.path.basename(tempfile.name))

        return True
    
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def get_description(video_path: str, yt: YouTube) -> str:
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


def get_relevant_timestamps(query: str, video_path: str, yt: YouTube) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = yt.length
    return start_time, end_time


def search_videos(query: str, num_videos: int) -> List[VideoMetadata]:
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
    while len(video_metas) < num_videos:
        for result in s.results[:(num_videos - len(video_metas))]:
            with tempfile.NamedTemporaryFile() as temp_fileobj:
                is_downloaded = download_video(result)
                if is_downloaded:
                    start, end = get_relevant_timestamps(temp_fileobj.name)
                    description = get_description(result, temp_fileobj.name)
                    video_metas.append(VideoMetadata(
                        id=result.id,
                        description=description,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                    ))
        s.get_next_results()
    return video_metas
