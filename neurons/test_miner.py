from omega.miner_utils import search_and_embed_videos, ImageBind
from omega.constants import VALIDATOR_TIMEOUT
import time

imagebind = ImageBind()
start = time.time()
query = "minecraft gameplay footage"
num_videos = 8
video_metadata_list = search_and_embed_videos(query, num_videos, imagebind)
time_elapsed = time.time() - start

if time_elapsed > VALIDATOR_TIMEOUT or len(video_metadata_list) < num_videos:
    if time_elapsed > VALIDATOR_TIMEOUT:
        print(f"Searching took {time_elapsed} seconds, which is longer than the validator timeout of {VALIDATOR_TIMEOUT} seconds")

    if len(video_metadata_list) < num_videos:
        print(f"Only got {len(video_metadata_list)} videos, which is less than the requested {num_videos} videos")
else:
    print(f"SUCCESS! Search and embed took {time_elapsed} seconds and got {len(video_metadata_list)} videos")
