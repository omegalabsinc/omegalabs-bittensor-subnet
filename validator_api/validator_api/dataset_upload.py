from io import BytesIO
from typing import List
from datetime import datetime
import random
import tempfile
import gc

from datasets import Dataset, Audio
from huggingface_hub import HfApi
import ulid
import soundfile as sf
import base64
import numpy as np
import os

from omega.protocol import VideoMetadata, AudioMetadata

from validator_api.validator_api import config


HF_API = HfApi(token=config.HF_TOKEN)
NUM_BUCKETS = 1000


def get_data_path(batch_ulid_str: str) -> str:
    batch_ulid = ulid.from_str(batch_ulid_str)
    bucket = batch_ulid.int % NUM_BUCKETS
    return f"default/train/{bucket:03d}/{batch_ulid_str}.parquet"


def get_random_batch_size(batch_size: int) -> int:
    return random.choice(
        [
            batch_size // 2,
            batch_size,
        ]
    )


def create_repo(name: str) -> None:
    try:
        HF_API.create_repo(
            repo_id=name,
            repo_type=config.REPO_TYPE,
            exist_ok=True,
            token=config.HF_TOKEN,
        )
        print("Successfully created/verified repository")
    except Exception as e:
        print(f"Error creating repository: {e}")


class DatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.desired_batch_size = get_random_batch_size(config.UPLOAD_BATCH_SIZE)
        self.min_batch_size = 32

    def add_videos(
        self,
        metadata: List[VideoMetadata],
        video_ids: List[str],
        description_relevance_scores: List[float],
        query_relevance_scores: List[float],
        query: str,
    ) -> None:
        curr_time = datetime.now()
        self.current_batch.extend(
            [
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
                for vid_uuid, video, desc_score, query_score in zip(
                    video_ids,
                    metadata,
                    description_relevance_scores,
                    query_relevance_scores,
                )
            ]
        )
        print(
            f"Added {len(metadata)} videos to batch, now have {len(self.current_batch)}"
        )
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()

    def submit(self) -> None:
        if len(self.current_batch) < self.min_batch_size:
            print(
                f"Need at least {self.min_batch_size} videos to submit, but have {len(self.current_batch)}"
            )
            return

        # Take a copy of the data we need and immediately clear the original batch
        data_to_upload = self.current_batch[: self.desired_batch_size]
        self.current_batch = self.current_batch[self.desired_batch_size :]
        self.desired_batch_size = get_random_batch_size(config.UPLOAD_BATCH_SIZE)

        print(f"Uploading batch of {len(data_to_upload)} videos")
        with BytesIO() as f:
            try:
                # Create dataset and immediately upload it
                dataset = Dataset.from_list(data_to_upload)
                num_bytes = dataset.to_parquet(f)
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
            finally:
                # Explicitly delete large objects
                del data_to_upload
                gc.collect()  # force garbage collection after upload


class AudioDatasetUploader:
    def __init__(self):
        self.current_batch = []
        self.min_batch_size = 8
        self.desired_batch_size = get_random_batch_size(config.UPLOAD_AUDIO_BATCH_SIZE)
        self.temp_dir = tempfile.mkdtemp()
        self.max_batch_size = 128
    
    def save_audio_to_disk(self, audio_data, sample_rate, audio_id):
        """Save audio data to disk and return the file path"""
        try:
            filepath = os.path.join(self.temp_dir, f"{audio_id}.wav")
            sf.write(filepath, audio_data, sample_rate)
            return filepath
        except Exception as e:
            print(f"Error saving audio to disk: {e}")
            return None

    def cleanup_files(self, file_paths):
        """Delete temporary audio files after upload"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Failed to delete temporary file {path}: {e}")

    def add_audios(
        self,
        metadata: List[AudioMetadata],
        audio_ids: List[str],
        inverse_der: float,
        audio_length_score: float,
        audio_quality_total_score: float,
        audio_query_score: float,
        query: str,
        total_score: float,
    ) -> None:
        curr_time = datetime.now()
        
        for audio_uuid, audio in zip(audio_ids, metadata):
            # Decode base64 audio and save to disk
            audio_data = np.frombuffer(base64.b64decode(audio.audio_bytes), dtype=np.float32)
            audio_path = self.save_audio_to_disk(audio_data, 16000, audio_uuid)
            
            self.current_batch.append({
                "audio_id": audio_uuid,
                "youtube_id": audio.video_id,
                "audio": {
                    "path": audio_path,
                    "sampling_rate": 16000
                },
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
            })

        # If the batch exceeds max size, shuffle and trim it
        if len(self.current_batch) > self.max_batch_size:
            print(f"Batch exceeds max size of {self.max_batch_size}, shuffling and trimming")
            random.shuffle(self.current_batch)
            
            # Get the paths of files to delete
            excess_items = self.current_batch[self.max_batch_size:]
            paths_to_delete = [item["audio"]["path"] for item in excess_items if item["audio"]["path"]]
            
            # Delete the files
            self.cleanup_files(paths_to_delete)
            
            # Trim the batch
            self.current_batch = self.current_batch[:self.max_batch_size]

        print(f"Added {len(metadata)} audios to batch, now have {len(self.current_batch)}")
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()

    def submit(self) -> None:
        if len(self.current_batch) < self.min_batch_size:
            print(f"Need at least {self.min_batch_size} audios to submit, but have {len(self.current_batch)}")
            return

        # Store only what we need temporarily and clear original data
        temp_batch = self.current_batch[:self.desired_batch_size]
        self.current_batch = self.current_batch[self.desired_batch_size:]
        self.desired_batch_size = get_random_batch_size(config.UPLOAD_AUDIO_BATCH_SIZE)
        
        # Create a new list for processing
        data_to_upload = []
        file_paths_to_delete = []
        
        for item in temp_batch:
            audio_path = item["audio"]["path"]
            file_paths_to_delete.append(audio_path)
            
            # Create new dictionary with fresh references
            upload_item = {
                "audio_id": item["audio_id"],
                "youtube_id": item["youtube_id"],
                "audio": {
                    "path": audio_path,
                    "sampling_rate": 16000,
                    "array": sf.read(audio_path)[0]
                },
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "audio_embed": item["audio_embed"].copy() if isinstance(item["audio_embed"], list) else item["audio_embed"],
                "diar_timestamps_start": item["diar_timestamps_start"].copy() if hasattr(item["diar_timestamps_start"], "copy") else item["diar_timestamps_start"],
                "diar_timestamps_end": item["diar_timestamps_end"].copy() if hasattr(item["diar_timestamps_end"], "copy") else item["diar_timestamps_end"],
                "diar_speakers": item["diar_speakers"].copy() if hasattr(item["diar_speakers"], "copy") else item["diar_speakers"],
                "inverse_der": item["inverse_der"],
                "audio_length_score": item["audio_length_score"],
                "audio_quality_score": item["audio_quality_score"],
                "query_relevance_score": item["query_relevance_score"],
                "total_score": item["total_score"],
                "query": item["query"],
                "submitted_at": item["submitted_at"],
            }
            data_to_upload.append(upload_item)
        
        # Clear reference to temporary batch
        temp_batch.clear()
        del temp_batch
        
        # Force garbage collection
        gc.collect()
        
        print(f"Uploading batch of {len(data_to_upload)} audios")
        with tempfile.NamedTemporaryFile(suffix='.parquet') as f:
            try:
                # Create dataset and immediately work with it without storing reference in self
                dataset = Dataset.from_list(data_to_upload)
                dataset = dataset.cast_column("audio", Audio())
                num_bytes = dataset.to_parquet(f.name)
                
                # Use the global HF_API instead of creating a new one
                HF_API.upload_file(
                    path_or_fileobj=f.name,
                    path_in_repo=get_data_path(str(ulid.new())),
                    repo_id=config.HF_AUDIO_REPO,
                    repo_type=config.REPO_TYPE,
                    token=config.HF_TOKEN
                )
                print(f"Uploaded {num_bytes} bytes to Hugging Face")
            except Exception as e:
                print(f"Error uploading to Hugging Face: {e}")
            finally:
                # Clean up with proper indentation
                del data_to_upload
                del dataset
                
                # Delete temporary files
                self.cleanup_files(file_paths_to_delete)
                
                # Force garbage collection again
                gc.collect()
    
    def __del__(self):
        """Clean up temporary directory when uploader is destroyed"""
        try:
            if os.path.exists(self.temp_dir):
                # Remove any remaining files
                for filename in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")
                
                # Remove the directory
                os.rmdir(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

audio_dataset_uploader = AudioDatasetUploader()
video_dataset_uploader = DatasetUploader()


if __name__ == "__main__":
    audio_wav_file = "../example.wav"
    with open(audio_wav_file, "rb") as f:
        audio_bytes = base64.b64encode(f.read()).decode("utf-8")
    for _ in range(100):
        audio_dataset_uploader.add_audios(
            metadata=[
                AudioMetadata(
                    video_id="123",
                    start_time=0,
                    end_time=10,
                    audio_bytes=audio_bytes,
                    audio_emb=[],
                    views=0,
                    diar_timestamps_start=[],
                    diar_timestamps_end=[],
                    diar_speakers=[],
                )
            ]
            * 10,
            audio_ids=list(range(10)),
            inverse_der=0.0,
            audio_length_score=0.0,
            audio_quality_total_score=0.0,
            audio_query_score=0.0,
            query="",
            total_score=0.0,
        )
        # audio_dataset_uploader.submit()
        import psutil
        import os

        process = psutil.Process(os.getpid())
        print(f"Current RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
