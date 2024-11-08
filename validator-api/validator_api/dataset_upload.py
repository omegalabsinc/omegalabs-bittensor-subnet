from io import BytesIO
from typing import List
from datetime import datetime
import random

from datasets import Dataset
from huggingface_hub import HfApi
import ulid

from omega.protocol import VideoMetadata, AudioMetadata

from validator_api import config


HF_API = HfApi(token=config.HF_TOKEN)
NUM_BUCKETS = 1000


def get_data_path(batch_ulid_str: str) -> str:
    batch_ulid = ulid.from_str(batch_ulid_str)
    bucket = batch_ulid.int % NUM_BUCKETS
    return f"default/train/{bucket:03d}/{batch_ulid_str}.parquet"


def get_random_batch_size() -> int:
    return random.choice([
        config.UPLOAD_BATCH_SIZE // 2,
        config.UPLOAD_BATCH_SIZE,
        config.UPLOAD_BATCH_SIZE * 2,
    ])

def create_repo(name: str) -> None:
    try:
        HF_API.create_repo(
            repo_id=name,
            repo_type=config.REPO_TYPE,
            exist_ok=True,
            token=config.HF_TOKEN
        )
        print("Successfully created/verified repository")
    except Exception as e:
        print(f"Error creating repository: {e}")

class DatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.desired_batch_size = get_random_batch_size()
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
                    token=config.HF_TOKEN,
                )
                print(f"Uploaded {num_bytes} bytes to Hugging Face")
            except Exception as e:
                print(f"Error uploading to Hugging Face: {e}")
        self.current_batch = self.current_batch[self.desired_batch_size:]
        self.desired_batch_size = get_random_batch_size()

    

class AudioDatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.min_batch_size = 2
        self.desired_batch_size = get_random_batch_size()

    def add_audios(
        self, metadata: List[AudioMetadata], audio_ids: List[str],
        inverse_der: float, audio_length_score: float,
        audio_quality_total_score: float, audio_query_score: float,
        query: str, total_score: float
    ) -> None:
        curr_time = datetime.now()
        self.current_batch.extend([
            {
                "audio_id": audio_uuid,
                "youtube_id": audio.video_id,
                "start_time": audio.start_time,
                "end_time": audio.end_time,
                "audio_embed": audio.audio_emb,
                "diar_timestamps_start": audio.diar_timestamps_start,
                "diar_timestamps_end": audio.diar_timestamps_end,
                "diar_speakers": audio.diar_speakers,
                "inverse_der": inverse_der,
                "audio_length_score": audio_length_score,
                "audio_quality_score": audio_quality_total_score,
                "query_relevance_score": audio_query_score,
                "total_score": total_score,
                "query": query,
                "submitted_at": int(curr_time.timestamp()),
            }
            for audio_uuid, audio in zip(audio_ids, metadata)
        ])
        print(f"Added {len(metadata)} audios to batch, now have {len(self.current_batch)}")
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()

    def submit(self) -> None:
        if len(self.current_batch) < self.min_batch_size:
            print(f"Need at least {self.min_batch_size} audios to submit, but have {len(self.current_batch)}")
            return
        data = self.current_batch[:self.desired_batch_size]
        print(f"Uploading batch of {len(self.current_batch)} audios")
        with BytesIO() as f:
            dataset = Dataset.from_list(data)
            num_bytes = dataset.to_parquet(f)
            try:
                HF_API.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=get_data_path(str(ulid.new())),
                    repo_id=config.HF_AUDIO_REPO,
                    repo_type=config.REPO_TYPE,
                    token=config.HF_TOKEN,
                )
                print(f"Uploaded {num_bytes} bytes to Hugging Face")
            except Exception as e:
                print(f"Error uploading to Hugging Face: {e}")
        self.current_batch = self.current_batch[self.desired_batch_size:]
        self.desired_batch_size = get_random_batch_size()




audio_dataset_uploader = AudioDatasetUploader()
video_dataset_uploader = DatasetUploader()
