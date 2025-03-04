import re
import json
import os
import tempfile
from typing import Optional, BinaryIO
import requests
import bittensor as bt
import ffmpeg
from pydantic import BaseModel
from yt_dlp import YoutubeDL
import librosa
import numpy as np

from omega.constants import FIVE_MINUTES


def seconds_to_str(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def clip_video(video_path: str, start: int, end: int) -> Optional[BinaryIO]:
    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")
    (
        ffmpeg.input(video_path, ss=seconds_to_str(start), to=seconds_to_str(end))
        .output(
            temp_fileobj.name, c="copy"
        )  # copy flag prevents decoding and re-encoding
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_fileobj


def skip_live(info_dict):
    """
    function to skip downloading if it's a live video (yt_dlp doesn't respect the 20 minute
    download limit for live videos), and we don't want to hang on an hour long stream
    """
    if info_dict.get("is_live"):
        return "Skipping live video"
    return None


class YoutubeResult(BaseModel):
    video_id: str
    title: str
    description: Optional[str]
    length: int
    views: int


def search_videos(query, max_results=8):
    videos = []
    ydl_opts = {
        "format": "worst",
        "dumpjson": True,
        "extract_flat": True,
        "quiet": True,
        "simulate": True,
        "match_filter": skip_live,
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_query = f"ytsearch{max_results}:{query}"
            result = ydl.extract_info(search_query, download=False)
            if "entries" in result and result["entries"]:
                videos = [
                    YoutubeResult(
                        video_id=entry["id"],
                        title=entry["title"],
                        description=entry.get("description"),
                        length=(
                            int(entry.get("duration"))
                            if entry.get("duration")
                            else FIVE_MINUTES
                        ),
                        views=(
                            entry.get("view_count") if entry.get("view_count") else 0
                        ),
                    )
                    for entry in result["entries"]
                ]
        except Exception as e:
            bt.logging.warning(f"Error searching for videos: {e}")
            return []
    return videos


def get_video_duration(filename: str) -> int:
    metadata = ffmpeg.probe(filename)
    video_stream = next(
        (stream for stream in metadata["streams"] if stream["codec_type"] == "video"),
        None,
    )
    duration = int(float(video_stream["duration"]))
    return duration


class IPBlockedException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class FakeVideoException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def is_valid_youtube_id(youtube_id: str) -> bool:
    return youtube_id is not None and len(youtube_id) == 11


def download_youtube_video(
    video_id: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
    proxy: Optional[str] = None,
) -> Optional[BinaryIO]:
    if not is_valid_youtube_id(video_id):
        raise FakeVideoException(f"Invalid Youtube video ID: {video_id}")

    video_url = f"https://www.youtube.com/watch?v={video_id}"

    temp_fileobj = tempfile.NamedTemporaryFile(suffix=".mp4")

    ydl_opts = {
        "format": "worst",  # Download the worst quality
        "outtmpl": temp_fileobj.name,  # Set the output template to the temporary file"s name
        "overwrites": True,
        "quiet": True,
        "noprogress": True,
        "match_filter": skip_live,
    }

    if start is not None and end is not None:
        ydl_opts["download_ranges"] = lambda _, __: [
            {"start_time": start, "end_time": end}
        ]

    if proxy is not None:
        ydl_opts["proxy"] = proxy

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if the file is empty (download failed)
        if os.stat(temp_fileobj.name).st_size == 0:
            print(f"Error downloading Youtube video: {temp_fileobj.name} is empty")
            temp_fileobj.close()
            return None

        return temp_fileobj
    except Exception as e:
        temp_fileobj.close()
        if "Your IP is likely being blocked by Youtube" in str(
            e
        ) or "Requested format is not available" in str(e):
            raise IPBlockedException(e)

        # Quick check to see if miner passed an "unplayable" (sign-in required, paid video, etc.).
        fake_video = False
        try:
            result = requests.get(video_url, proxies={"https": proxy})
            json_match = re.search(
                r"ytInitialPlayerResponse\s*=\s*(\{(?:.*?)\})\s*;\s*<", result.text
            )
            if json_match:
                player_info = json.loads(json_match.group(1))
                status = player_info.get("playabilityStatus", {}).get("status", "ok")
                unacceptable_statuses = ("UNPLAYABLE",)
                if status in unacceptable_statuses or (
                    status == "ERROR"
                    and player_info["playabilityStatus"].get("reason", "").lower()
                    == "video unavailable"
                ):
                    if "sign in to confirm you’re not a bot" not in result.text.lower():
                        if (
                            player_info["playabilityStatus"]["errorScreen"][
                                "playerErrorMessageRenderer"
                            ]["subreason"]["simpleText"]
                            != "This content isn’t available."
                        ):
                            fake_video = True
                            print(
                                f"Fake video submitted, youtube player status [{status}]: {player_info['playabilityStatus']}"
                            )
        except Exception as fake_check_exc:
            print(f"Error sanity checking playability: {fake_check_exc}")
        if fake_video:
            raise FakeVideoException("Unplayable video provided")
        if any(
            fake_vid_msg in str(e)
            for fake_vid_msg in [
                "Video unavailable",
                "is not a valid URL",
                "Incomplete YouTube ID",
                "Unsupported URL",
            ]
        ):
            if "Video unavailable. This content isn’t available." not in str(e):
                raise FakeVideoException(e)
        print(f"Error downloading video: {e}")
        return None


def copy_audio(video_path: str) -> BinaryIO:
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".aac")
    (
        ffmpeg.input(video_path)
        .output(temp_audiofile.name, vn=None, acodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )
    return temp_audiofile


def copy_audio_wav(video_path: str) -> BinaryIO:
    """
    Extract audio from video file to 16-bit PCM WAV format.

    Args:
        video_path: Path to input video

    Returns:
        BinaryIO: Temporary file containing WAV audio
    """
    temp_audiofile = tempfile.NamedTemporaryFile(suffix=".wav")

    (
        ffmpeg.input(video_path)
        .output(
            temp_audiofile.name,
            acodec="pcm_s16le",  # 16-bit PCM
            ac=1,  # mono
            ar=16000,  # 16kHz sample rate
            vn=None,  # no video
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return temp_audiofile


def get_audio_bytes(video_path: str) -> bytes:
    audio_file = copy_audio_wav(video_path)
    with open(audio_file.name, "rb") as f:
        wav_bytes = f.read()

    # Clean up temp file
    audio_file.close()

    # NOTE: MINERS, you cannot change the sample rate here or we will not be able to score your audio
    return wav_bytes
