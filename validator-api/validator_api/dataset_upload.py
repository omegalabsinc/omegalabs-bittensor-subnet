from io import BytesIO
from typing import List

from datasets import Dataset
from huggingface_hub import HfApi
import ulid

from omega.protocol import VideoMetadata

from validator_api import config


HF_API = HfApi()


def get_data_path(batch_id: str) -> str:
    return f"partitions/{batch_id}.parquet"


class DatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.desired_batch_size = 1024

    def add_videos(self, metadata: List[VideoMetadata], video_ids: List[str]) -> None:
        self.current_batch.extend([
            {
                "video_id": vid_uuid,
                "youtube_id": video.video_id,
                "description": video.description,
                "views": video.views,
                "start_time": video.start_time,
                "end_time": video.end_time,
                "video_embed": video.video_emb,
                "audio_embed": video.audio_emb,
                "description_embed": video.description_emb,
            }
            for vid_uuid, video in
            zip(video_ids, metadata)
        ])
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()

    def submit(self) -> None:
        data = self.current_batch[:self.desired_batch_size]
        with BytesIO() as f:
            dataset = Dataset.from_list(data)
            num_bytes = dataset.to_parquet(f)
            try:
                HF_API.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=get_data_path(str(ulid.new())),
                    repo_id=config.HF_REPO,
                    repo_type=config.REPO_TYPE,
                )
            except Exception as e:
                print(f"Error uploading to Hugging Face: {e}")
        self.current_batch = self.current_batch[self.desired_batch_size:]


dataset_uploader = DatasetUploader()
