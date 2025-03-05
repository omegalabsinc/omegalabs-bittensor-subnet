from omega.miner_utils import search_and_embed_youtube_videos, ImageBind
from omega.protocol import Videos
import time

imagebind = ImageBind()
start = time.time()
query = "minecraft gameplay footage"
num_videos = 1
video_metadata_list = search_and_embed_youtube_videos(query, num_videos, imagebind)
print(f"Search and embed took {time.time() - start} seconds")

videos = Videos(
    query=query,
    num_videos=num_videos,
    video_metadata=video_metadata_list,
)

# dataset_uploader.min_batch_size = 2  # override to force upload
# dataset_uploader.desired_batch_size = 2  # override to force upload
# print(asyncio.run(score_and_upload_videos(videos, imagebind)))
