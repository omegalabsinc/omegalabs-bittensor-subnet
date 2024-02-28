import random
import uuid
from typing import List, Tuple

from pinecone import Pinecone
import torch
import torch.nn.functional as F

from omega.protocol import Videos, VideoMetadata
from omega import video_utils
from omega.constants import MAX_VIDEO_LENGTH
from omega.imagebind_wrapper import ImageBind, Embeddings

from validator_api import config
from validator_api.dataset_upload import dataset_uploader


PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
DIFFERENCE_THRESHOLD = 0.05
SIMILARITY_THRESHOLD = 1 - DIFFERENCE_THRESHOLD


def compute_description_relevance_score(embeddings: Embeddings) -> float:
    return F.cosine_similarity(embeddings.video, embeddings.description).sum().item()


def compute_query_relevance_score(embeddings: Embeddings, query: str, imagebind: ImageBind) -> float:
    query_emb = imagebind.embed_text([query])
    return F.cosine_similarity(embeddings.video, query_emb).sum().item()


def compute_novelty_score(embeddings: Embeddings) -> Tuple[float, List[bool]]:
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
    is_too_similar = [score < DIFFERENCE_THRESHOLD for score in novelty_scores]
    novelty_score = sum([
        score for score, is_too_similar
        in zip(novelty_scores, is_too_similar) if not is_too_similar
    ])
    return novelty_score, is_too_similar


def upload_to_pinecone(embeddings: Embeddings, metadata: List[VideoMetadata]) -> None:
    video_ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
    PINECONE_INDEX.upsert(
        vectors=[
            {
                "id": video_uuid,
                "values": embedding.tolist(),
                "metadata": {
                    "youtube_id": video.video_id,
                }
            }
            for video_uuid, video, embedding in zip(video_ids, metadata, embeddings.video)
        ],
    )
    return video_ids


def filter_embeddings(embeddings: Embeddings, is_too_similar: List[bool]) -> Embeddings:
    """Filter the embeddings based on whether they are too similar to the query."""
    is_too_similar = torch.tensor(is_too_similar)
    embeddings.video = embeddings.video[~is_too_similar]
    embeddings.audio = embeddings.audio[~is_too_similar]
    embeddings.description = embeddings.description[~is_too_similar]
    return embeddings


def is_similar(emb_1: torch.Tensor, emb_2: List[float]) -> bool:
    return F.cosine_similarity(emb_1, torch.tensor(emb_2, device=emb_1.device)) > SIMILARITY_THRESHOLD


def random_check(videos: Videos, imagebind: ImageBind) -> bool:
    random_video = None
    metadata = [v for v in videos.video_metadata]  # list shallow copy
    while random_video is None and len(metadata) > 0:
        idx = random.randint(0, len(metadata) - 1)
        random_video_metadata = metadata.pop(idx)
        if random_video_metadata.end_time - random_video_metadata.start_time > MAX_VIDEO_LENGTH:
            return False
        random_video = video_utils.download_video(
            random_video_metadata.video_id,
            random_video_metadata.start_time,
            random_video_metadata.end_time,
        )
    if random_video is None:
        return False
    embeddings = imagebind.embed([random_video_metadata.description], [random_video])
    if not (
        is_similar(embeddings.video, random_video_metadata.video_emb) and
        is_similar(embeddings.audio, random_video_metadata.audio_emb) and
        is_similar(embeddings.description, random_video_metadata.description_emb)
    ):
        return False
    return True


async def score_and_upload_videos(videos: Videos, imagebind: ImageBind) -> float:
    # Randomly check 1 video embedding
    passed_check = random_check(videos, imagebind)
    if not passed_check:
        return -1.0

    # Upload the videos to Pinecone and deduplicate
    metadata = videos.video_metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    video_ids = upload_to_pinecone(embeddings, metadata)
    novelty_score, is_too_similar = compute_novelty_score(embeddings)
    embeddings = filter_embeddings(embeddings, is_too_similar)
    metadata = [metadata for metadata, too_similar in zip(metadata, is_too_similar) if not too_similar]
    video_ids = [video_id for video_id, too_similar in zip(video_ids, is_too_similar) if not too_similar]

    # Compute relevance scores
    description_relevance_score = F.cosine_similarity(
        embeddings.video, embeddings.description
    ).sum().item()
    query_relevance_score = F.cosine_similarity(
        embeddings.video, imagebind.embed_text([videos.query])
    ).sum().item()
    
    # Aggregate scores
    score = (description_relevance_score + query_relevance_score + novelty_score) / 3 / videos.num_videos

    # Schedule upload to HuggingFace
    dataset_uploader.add_videos(metadata, video_ids)
    
    return score
