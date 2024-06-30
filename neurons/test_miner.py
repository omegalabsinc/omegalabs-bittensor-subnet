from omega.miner_utils import search_and_embed_youtube_videos, ImageBind
from omega.constants import VALIDATOR_TIMEOUT
from omega.protocol import Videos
import time
import requests

imagebind = ImageBind()
start = time.time()
query = "wine and winemaking"
num_videos = 8
video_metadata_list = search_and_embed_youtube_videos(query, num_videos, imagebind)
time_elapsed = time.time() - start

if time_elapsed > VALIDATOR_TIMEOUT or len(video_metadata_list) < num_videos:
    if time_elapsed > VALIDATOR_TIMEOUT:
        print(f"Searching took {time_elapsed} seconds, which is longer than the validator timeout of {VALIDATOR_TIMEOUT} seconds")

    if len(video_metadata_list) < num_videos:
        print(f"Only got {len(video_metadata_list)} videos, which is less than the requested {num_videos} videos")
else:
    print(f"SUCCESS! Search and embed took {time_elapsed} seconds and got {len(video_metadata_list)} videos")


if len(video_metadata_list) == 0:
    print("No videos found")
else:
    videos = Videos(query=query, num_videos=num_videos, video_metadata=video_metadata_list)
    response = requests.get(
        "https://dev-validator.api.omega-labs.ai/api/count_unique",
        json=videos.to_serializable_dict(videos)
    )
    print(response.json())
