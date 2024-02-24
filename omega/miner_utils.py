import os
from typing import List, Optional, BinaryIO, Tuple
import tempfile

from pytube import Search, YouTube

from omega.protocol import VideoMetadata


def download_video_from_id(video_id: str, get_best_quality: bool = False) -> Optional[BinaryIO]:
    """
    Download a video from YouTube by its ID.

    Args:
        video_id (str): The ID of the video to download.

    Returns:
        Optional[BytesIO]: A BytesIO object containing the video data if the download was
        successful, otherwise None.
    """
    return download_video(YouTube.from_id(video_id, get_best_quality))


def download_video(yt: YouTube, get_best_quality: bool = False) -> Optional[BinaryIO]:
    """
    Download a video from YouTube to a given file object.

    get_best_quality is set to False by default, but can be set to True if you want to download
    the highest resolution video instead.
    """
    try:
        # Select the highest resolution available that is AVC codec (MPEG-4)
        streams = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
        if get_best_quality:
            streams = streams.desc()
        else:
            streams = streams.asc()
        avc_streams = [stream for stream in streams if stream.video_codec.startswith("avc")]
        video = avc_streams[0]

        # Download the video
        temp_fileobj = tempfile.NamedTemporaryFile()
        video.download(
            output_path=os.path.dirname(temp_fileobj.name),
            filename=os.path.basename(temp_fileobj.name)
        )
        return temp_fileobj
    
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


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
            download_path = download_video(result)
            if download_path:
                try:
                    start, end = get_relevant_timestamps(download_path)
                    description = get_description(result, download_path)
                    video_metas.append(VideoMetadata(
                        video_id=result.id,
                        description=description,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                    ))
                finally:
                    download_path.close()
        s.get_next_results()
    return video_metas
