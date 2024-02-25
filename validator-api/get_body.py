from omega.miner_utils import search_videos
import json
import time

start = time.time()
video_meta = search_videos("chefs cooking in a kitchen shorts", 2)
breakpoint()
print(time.time() - start)
