from io import BytesIO
from typing import List
from datetime import datetime

from datasets import Dataset
from huggingface_hub import HfApi
import ulid

from omega.protocol import VideoMetadata

from validator_api import config


HF_API = HfApi()


def get_data_path(batch_id: str) -> str:
    return f"default/train/{batch_id}.parquet"


class DatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.desired_batch_size = config.UPLOAD_BATCH_SIZE
        self.min_batch_size = 32

    def add_videos(
        self, metadata: List[VideoMetadata], video_ids: List[str],
        description_relevance_scores: List[float], query_relevance_scores: List[float],
        query: str,
    ) -> None:
        curr_time = datetime.now()
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
                "description_relevance_score": desc_score,
                "query_relevance_score": query_score,
                "query": query,
                "submitted_at": int(curr_time.timestamp()),
            }
            for vid_uuid, video, desc_score, query_score
            in zip(video_ids, metadata, description_relevance_scores, query_relevance_scores)
        ])
        print(f"Added {len(metadata)} videos to batch, now have {len(self.current_batch)}")
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()

    def submit(self) -> None:
        if len(self.current_batch) < self.min_batch_size:
            print(f"Need at least {self.min_batch_size} videos to submit, but have {len(self.current_batch)}")
            return
        data = self.current_batch[:self.desired_batch_size]
        print(f"Uploading batch of {len(self.current_batch)} videos")
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
                print(f"Uploaded {num_bytes} bytes to Hugging Face")
            except Exception as e:
                print(f"Error uploading to Hugging Face: {e}")
        self.current_batch = self.current_batch[self.desired_batch_size:]


dataset_uploader = DatasetUploader()
