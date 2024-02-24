import uuid
import ulid
from io import BytesIO
from typing import List

from datasets import Dataset
from huggingface_hub import HfApi
from pinecone import Pinecone
import torch.nn.functional as F

from omega.protocol import Videos, VideoMetadata
from omega.miner_utils import download_video_from_id
from validator_api.imagebind_wrapper import ImageBind, Embeddings
from validator_api import config


PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
HF_API = HfApi()


def compute_description_relevance_score(embeddings: Embeddings) -> float:
    return F.cosine_similarity(embeddings.video, embeddings.description).sum().item()


def compute_query_relevance_score(embeddings: Embeddings, query: str, imagebind: ImageBind) -> float:
    query_emb = imagebind.embed_text(query)
    return F.cosine_similarity(embeddings.video, query_emb).sum().item()


def compute_novelty_score(embeddings: Embeddings) -> float:
    """
    Take the top 2nd match from the Pinecone index (cause the 1st match is itself) and then
    take the complement of the score to be the novelty score.
    """
    novelty_scores = [
        1 - PINECONE_INDEX.query(
            vector=embedding.tolist(),
            top_k=2,
        )["matches"][1]["score"]
        for embedding in embeddings.video
    ]
    return sum(novelty_scores)


def upload_to_pinecone(embeddings: Embeddings, metadata: List[VideoMetadata], batch_id: str) -> None:
    video_ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
    PINECONE_INDEX.upsert(
        vectors=[
            {
                "id": video_uuid,
                "values": embedding.tolist(),
                "metadata": {
                    "batch_id": batch_id,
                    "youtube_id": video.video_id,
                }
            }
            for video_uuid, video, embedding in zip(video_ids, metadata, embeddings.video)
        ],
    )
    return video_ids


def get_data_path(batch_id: str) -> str:
    return f"partitions/{batch_id}.parquet"


def upload_to_hf(embeddings: Embeddings, metadata: List[VideoMetadata], batch_id: str, video_ids: List[str]) -> None:
    data = [
        {
            "batch_id": batch_id,
            "video_id": vid_uuid,
            "youtube_id": video.video_id,
            "description": video.description,
            "views": video.views,
            "start_time": video.start_time,
            "end_time": video.end_time,
            "video_embed": vid_emb.tolist(),
            "audio_embed": audio_emb.tolist(),
            "description_embed": text_emb.tolist(),
        }
        for vid_uuid, video, vid_emb, audio_emb, text_emb in
        zip(video_ids, metadata, embeddings.video, embeddings.audio, embeddings.description)
    ]
    with BytesIO() as f:
        dataset = Dataset.from_dict(data).to_parquet(f)
        f.seek(0)
        HF_API.upload_file(
            dataset,
            path_in_repo=get_data_path(batch_id),
            repo_id=config.HF_REPO,
            repo_type=config.REPO_TYPE,
        )


def score_and_upload_videos(videos: Videos, imagebind: ImageBind) -> float:
    """
    1. Scores the videos
    2. Uploads video metadata to huggingface
    """
    batch_id = str(ulid.new())
    video_files = [
        download_video_from_id(video.video_id, get_best_quality=False)
        for video in videos.video_metadata
    ]
    metadata = [metadata for metadata, file in zip(videos.video_metadata, video_files) if file is not None]
    video_files = [file for file in video_files if file is not None]
    try:
        embeddings = imagebind.embed(metadata, video_files)
        video_ids = upload_to_pinecone(embeddings, videos, batch_id)
        description_relevance_score = compute_description_relevance_score(embeddings)
        query_relevance_score = compute_query_relevance_score(embeddings, videos.query, imagebind)
        novelty_score = compute_novelty_score(embeddings)
        score = (description_relevance_score + query_relevance_score + novelty_score) / 3 / videos.num_videos
        upload_to_hf(embeddings, metadata, batch_id, video_ids)
        return score
    finally:
        for video_file in video_files:
            video_file.close()
