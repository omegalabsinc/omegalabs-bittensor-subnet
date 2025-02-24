import asyncio
import uuid
from typing import List, Tuple

from pinecone import Pinecone
import torch
import torch.nn.functional as F

from omega.protocol import VideoMetadata, AudioMetadata
from omega.constants import DIFFERENCE_THRESHOLD
from omega.imagebind_wrapper import Embeddings, run_async
from validator_api import config
from validator_api.dataset_upload import video_dataset_uploader, audio_dataset_uploader


PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
PINECONE_AUDIO_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_AUDIO_INDEX)
GPU_SEMAPHORE = asyncio.Semaphore(1)
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(5)
VIDEO_TYPE = "video"
AUDIO_TYPE = "audio"
DESCRIPTION_TYPE = "description"


async def query_pinecone(vector: List[float]) -> float:
    response = await run_async(
        PINECONE_INDEX.query,
        vector=vector,
        top_k=1,
        filter={
            "modality_type": {"$eq": VIDEO_TYPE},
        },
    )
    if len(response["matches"]) > 0:
        return 1 - response["matches"][0]["score"] 
    else:
        print("No pinecone matches, returning 0")
        return 0

async def get_pinecone_novelty(metadata: List[VideoMetadata]) -> List[float]:
    """
    Take the top match from the Pinecone index.
    """
    novelty_scores = await asyncio.gather(*[
        query_pinecone(
            vector=mdata.video_emb
        )
        for mdata in metadata
    ])    
    return novelty_scores

def compute_novelty_score_among_batch(emb: Embeddings) -> List[float]:
    video_tensor = emb.video
    num_videos = video_tensor.shape[0]
    novelty_scores = []
    for i in range(num_videos - 1):
        similarity_score = F.cosine_similarity(video_tensor[[i]], video_tensor[i + 1:]).max()
        novelty_scores.append(1 - similarity_score.item())
    novelty_scores.append(1.0)  # last video is 100% novel
    return novelty_scores

async def async_zero() -> None:
    return 0

async def compute_novelty_score(embeddings: Embeddings) -> Tuple[float, List[bool]]:
    local_novelty_scores = compute_novelty_score_among_batch(embeddings)
    global_novelty_scores = await asyncio.gather(*[
        async_zero() if local_score < DIFFERENCE_THRESHOLD else  # don't even query Pinecone if it's already too similar
        query_pinecone(vector=embedding.tolist())
        for embedding, local_score in zip(embeddings.video, local_novelty_scores)
    ])
    true_novelty_scores = [
        min(local_score, global_score) for local_score, global_score
        in zip(local_novelty_scores, global_novelty_scores)
    ]
    is_too_similar = [score < DIFFERENCE_THRESHOLD for score in true_novelty_scores]
    novelty_score = sum([
        score for score, is_too_similar
        in zip(true_novelty_scores, is_too_similar)
        if not is_too_similar
    ])
    return novelty_score, is_too_similar

def upload_to_pinecone(embeddings: Embeddings, metadata: List[VideoMetadata]) -> None:
    video_ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
    try:
        PINECONE_INDEX.upsert(
            vectors=[
                {
                    "id": f"{VIDEO_TYPE[:3]}{video_uuid}",
                    "values": embedding_vid.tolist(),
                    "metadata": {
                        "youtube_id": video.video_id,
                        "modality_type": VIDEO_TYPE,
                    }
                }
                for video_uuid, video, embedding_vid
                in zip(video_ids, metadata, embeddings.video)
            ],
        )
    except Exception as e:
        print(f"Failed to upload to Pinecone: {e}")
    return video_ids

def upload_to_pinecone_audio(embeddings: Embeddings, metadata: List[AudioMetadata]) -> None:
    audio_ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
    try:
        PINECONE_AUDIO_INDEX.upsert(
            vectors=[
                {
                    "id": f"{audio_uuid}",
                    "values": embedding_aud.tolist(),
                    "metadata": {
                        "youtube_id": audio.video_id,
                    }
                }
                for audio_uuid, audio, embedding_aud
                in zip(audio_ids, metadata, embeddings.audio)
            ],
        )
    except Exception as e:
        print(f"Failed to upload to Pinecone: {e}")
    return audio_ids

async def upload_video_metadata(
    metadata: List[VideoMetadata], 
    description_relevance_scores: List[float], 
    query_relevance_scores: List[float], 
    query: str, 
) -> None:
    # generate embeddings from our metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]),
    )
    # upload embeddings and metadata to pinecone
    video_ids = await run_async(upload_to_pinecone, embeddings, metadata)
    # Schedule upload to HuggingFace
    video_dataset_uploader.add_videos(
        metadata,
        video_ids,
        description_relevance_scores,
        query_relevance_scores,
        query,
    )
    return video_ids

async def upload_audio_metadata(
    metadata: List[AudioMetadata], 
    inverse_der: float, audio_length_score: float,
    audio_quality_total_score: float, 
    audio_query_score: float,
    query: str, 
    total_score: float 
) -> List[str]:
    embeddings = Embeddings(
        video=None,
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]),
        description=None,
    )
    audio_ids = await asyncio.to_thread(upload_to_pinecone_audio, embeddings, metadata)
    def _add_audios():
        audio_dataset_uploader.add_audios(
            metadata,
            audio_ids,
            inverse_der,
            audio_length_score,
            audio_quality_total_score,
            audio_query_score,
            query,
            total_score
        )
    async with audio_dataset_uploader.add_audios_mutex:
        await asyncio.to_thread(_add_audios)
    return audio_ids
