import asyncio
import random
import uuid
from typing import List, Tuple, Optional, BinaryIO

from pinecone import Pinecone
import torch
import torch.nn.functional as F

from omega.protocol import Videos, VideoMetadata
from omega import video_utils
from omega.constants import MAX_VIDEO_LENGTH, MIN_VIDEO_LENGTH
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async

from validator_api import config
from validator_api.dataset_upload import dataset_uploader


PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
DIFFERENCE_THRESHOLD = 0.05
SIMILARITY_THRESHOLD = 1 - DIFFERENCE_THRESHOLD
GPU_SEMAPHORE = asyncio.Semaphore(1)
DOWNLOAD_SEMAPHORE = asyncio.Semaphore(5)
VIDEO_DOWNLOAD_TIMEOUT = 10


def compute_novelty_score(embeddings: Embeddings, already_uploaded: bool) -> Tuple[float, List[bool]]:
    """
    Take the top 2nd match from the Pinecone index (cause the 1st match is itself) and then
    take the complement of the score to be the novelty score.
    """
    top_k = 2 if already_uploaded else 1
    select_idx = 1 if already_uploaded else 0
    novelty_scores = [
        1 - PINECONE_INDEX.query(
            vector=embedding.tolist(),
            top_k=top_k,
        )["matches"][select_idx]["score"]
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
    try:
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
    except Exception as e:
        print(f"Failed to upload to Pinecone: {e}")
    return video_ids


def filter_embeddings(embeddings: Embeddings, is_too_similar: List[bool]) -> Embeddings:
    """Filter the embeddings based on whether they are too similar to the query."""
    is_too_similar = torch.tensor(is_too_similar)
    embeddings.video = embeddings.video[~is_too_similar]
    embeddings.audio = embeddings.audio[~is_too_similar]
    embeddings.description = embeddings.description[~is_too_similar]
    return embeddings


def is_similar(emb_1: torch.Tensor, emb_2: List[float]) -> bool:
    return F.cosine_similarity(
        emb_1,
        torch.tensor(emb_2, device=emb_1.device).unsqueeze(0)
    ) > SIMILARITY_THRESHOLD


def metadata_check(metadata: List[VideoMetadata]) -> List[VideoMetadata]:
    return [
        video_metadata for video_metadata in metadata
        if (
            video_metadata.end_time - video_metadata.start_time <= MAX_VIDEO_LENGTH and
            video_metadata.end_time - video_metadata.start_time >= MIN_VIDEO_LENGTH
        )
    ]


def get_proxy_url() -> str:
    return random.choice(config.PROXY_LIST + [None])


async def get_random_video(metadata: List[VideoMetadata]) -> Optional[Tuple[VideoMetadata, Optional[BinaryIO]]]:
    random_video = None
    metadata_copy = [v for v in metadata]  # list shallow copy
    while random_video is None and len(metadata_copy) > 0:
        idx = random.randint(0, len(metadata_copy) - 1)
        random_metadata = metadata_copy.pop(idx)
        try:
            async with DOWNLOAD_SEMAPHORE:
                random_video = await asyncio.wait_for(run_async(
                    video_utils.download_video,
                    random_metadata.video_id,
                    random_metadata.start_time,
                    random_metadata.end_time,
                    proxy=get_proxy_url(),
                ), timeout=VIDEO_DOWNLOAD_TIMEOUT)
        except video_utils.IPBlockedException:
            # IP is blocked, cannot download video, check description only
            print("WARNING: IP is blocked, cannot download video, checking description only")
            return random_metadata, None
        except asyncio.TimeoutError:
            continue

    # IP not blocked, but video download failed regardless, bad video submitted
    if random_video is None:
        return None

    return random_metadata, random_video


async def random_check(random_meta_and_vid: List[VideoMetadata], imagebind: ImageBind) -> bool:
    random_metadata, random_video = random_meta_and_vid

    if random_video is None:
        desc_embeddings = await imagebind.embed_text_async([random_metadata.description])
        return is_similar(desc_embeddings, random_metadata.description_emb)

    # Video downloaded, check all embeddings
    embeddings = await imagebind.embed_async([random_metadata.description], [random_video])
    return (
        is_similar(embeddings.video, random_metadata.video_emb) and
        is_similar(embeddings.audio, random_metadata.audio_emb) and
        is_similar(embeddings.description, random_metadata.description_emb)
    )


async def get_num_unique_videos(videos: Videos) -> int:
    metadata = videos.video_metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]),
    )
    novelty_score, is_too_similar = compute_novelty_score(embeddings, already_uploaded=False)
    return sum([not is_sim for is_sim in is_too_similar])


async def score_videos_for_testing(videos: Videos, imagebind: ImageBind) -> float:
    metadata = metadata_check(videos.video_metadata)

    async with GPU_SEMAPHORE:
        query_emb = await imagebind.embed_text_async([videos.query])

    # Upload the videos to Pinecone and deduplicate
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    novelty_score, is_too_similar = compute_novelty_score(embeddings, already_uploaded=False)
    embeddings = filter_embeddings(embeddings, is_too_similar)
    metadata = [metadata for metadata, too_similar in zip(metadata, is_too_similar) if not too_similar]

    # Compute relevance scores
    description_relevance_scores = F.cosine_similarity(
        embeddings.video, embeddings.description
    ).tolist()
    query_relevance_scores = F.cosine_similarity(
        embeddings.video, query_emb
    ).tolist()

    # Aggregate scores
    score = (
        sum(description_relevance_scores) +
        sum(query_relevance_scores) +
        novelty_score
    ) / 3 / videos.num_videos

    score = max(score, 0.005)

    return {
        "is_unique": [not is_sim for is_sim in is_too_similar],
        "description_relevance_scores": description_relevance_scores,
        "query_relevance_scores": query_relevance_scores,
        "novelty_score": novelty_score,
        "score": score,
    }


async def score_and_upload_videos(videos: Videos, imagebind: ImageBind) -> float:
    # Randomly check 1 video embedding
    metadata = metadata_check(videos.video_metadata)
    check_video = config.CHECK_PROBABILITY > random.random()
    if check_video:
        random_meta_and_vid = await get_random_video(metadata)
        if random_meta_and_vid is None:
            return -0.4

    async with GPU_SEMAPHORE:
        if check_video:
            passed_check = await random_check(random_meta_and_vid, imagebind)
            if not passed_check:
                return -1.0
        query_emb = await imagebind.embed_text_async([videos.query])

    # Upload the videos to Pinecone and deduplicate
    print(f"Received {len(metadata)} videos")
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    video_ids = upload_to_pinecone(embeddings, metadata)
    novelty_score, is_too_similar = compute_novelty_score(embeddings, already_uploaded=True)
    embeddings = filter_embeddings(embeddings, is_too_similar)
    metadata = [metadata for metadata, too_similar in zip(metadata, is_too_similar) if not too_similar]
    video_ids = [video_id for video_id, too_similar in zip(video_ids, is_too_similar) if not too_similar]
    print(f"Filtered {len(videos.video_metadata)} videos down to {len(metadata)} videos")

    # Compute relevance scores
    description_relevance_scores = F.cosine_similarity(
        embeddings.video, embeddings.description
    ).tolist()
    query_relevance_scores = F.cosine_similarity(
        embeddings.video, query_emb
    ).tolist()

    # Aggregate scores
    score = (
        sum(description_relevance_scores) +
        sum(query_relevance_scores) +
        novelty_score
    ) / 3 / videos.num_videos

    # Schedule upload to HuggingFace
    dataset_uploader.add_videos(
        metadata,
        video_ids,
        description_relevance_scores,
        query_relevance_scores,
        videos.query,
    )
    score = max(score, 0.005)

    return score
