import asyncio
import random
import uuid
from typing import List, Tuple, Optional, BinaryIO

from pinecone import Pinecone
import torch
import torch.nn.functional as F

from omega.protocol import Videos, VideoMetadata, FocusVideoMetadata
from omega import video_utils
from omega.constants import (
    MAX_VIDEO_LENGTH, 
    MIN_VIDEO_LENGTH,
    DIFFERENCE_THRESHOLD, 
    SIMILARITY_THRESHOLD, 
    VIDEO_DOWNLOAD_TIMEOUT, 
    MIN_SCORE, 
    FAKE_VIDEO_PUNISHMENT,
    QUERY_RELEVANCE_SCALING_FACTOR,
    DESCRIPTION_RELEVANCE_SCALING_FACTOR
)
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async
import omega.imagebind_desc_mlp as imagebind_desc_mlp

from validator_api import config
from validator_api.dataset_upload import dataset_uploader


PINECONE_INDEX = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
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
            vectors=sum([
                [
                    {
                        "id": f"{modality_type[:3]}{video_uuid}",
                        "values": emb.tolist(),
                        "metadata": {
                            "youtube_id": video.video_id,
                            "modality_type": modality_type,
                        }
                    }
                    for emb, modality_type
                    in zip(
                        [embedding_vid, embedding_aud, embedding_des],
                        [VIDEO_TYPE, AUDIO_TYPE, DESCRIPTION_TYPE]
                    )
                ]
                for video_uuid, video, embedding_vid, embedding_aud, embedding_des
                in zip(video_ids, metadata, embeddings.video, embeddings.audio, embeddings.description)
            ], []),
        )
    except Exception as e:
        print(f"Failed to upload to Pinecone: {e}")
    return video_ids

def upload_focus_to_pinecone(embeddings: Embeddings, metadata: List[FocusVideoMetadata]) -> None:
    video_ids = [str(uuid.uuid4()) for _ in range(len(metadata))]
    try:
        PINECONE_INDEX.upsert(
            vectors=sum([
                [
                    {
                        "id": f"{modality_type[:3]}{video_uuid}",
                        "values": emb.tolist(),
                        "metadata": {
                            "focus_id": video.video_id,
                            "modality_type": modality_type,
                        }
                    }
                    for emb, modality_type
                    in zip(
                        [embedding_vid, embedding_des],
                        [VIDEO_TYPE, DESCRIPTION_TYPE]
                    )
                ]
                for video_uuid, video, embedding_vid, embedding_des 
                in zip(video_ids, metadata, embeddings.video, embeddings.description)
            ], []),
        )
    except Exception as e:
        print(f"Failed to upload to Pinecone: {e}")
    return video_ids

async def upload_video_metadata(
    metadata: List[VideoMetadata], 
    description_relevance_scores: List[float], 
    query_relevance_scores: List[float], 
    query: str, 
    imagebind: ImageBind
) -> None:
    # generate embeddings from our metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    # upload embeddings and metadata to pinecone
    video_ids = await run_async(upload_to_pinecone, embeddings, metadata)
    # Schedule upload to HuggingFace
    dataset_uploader.add_videos(
        metadata,
        video_ids,
        description_relevance_scores,
        query_relevance_scores,
        query,
    )
    return video_ids

async def upload_focus_metadata(
    metadata: List[FocusVideoMetadata], 
    imagebind: ImageBind
) -> None:
    # generate embeddings from our metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    print(f"before uploading to pinecone {metadata}")
    # upload embeddings and metadata to pinecone
    video_ids = await run_async(upload_focus_to_pinecone, embeddings, metadata)
    print(f"uploaded to pinecone {video_ids}")
    # Schedule upload to HuggingFace
    dataset_uploader.add_focus_videos(
        metadata,
        video_ids,
    )
    return video_ids

def filter_embeddings(embeddings: Embeddings, is_too_similar: List[bool]) -> Embeddings:
    """Filter the embeddings based on whether they are too similar to the query."""
    is_too_similar = torch.tensor(is_too_similar)
    if embeddings.video is not None:
        embeddings.video = embeddings.video[~is_too_similar]
    if embeddings.audio is not None:
        embeddings.audio = embeddings.audio[~is_too_similar]
    if embeddings.description is not None:
        embeddings.description = embeddings.description[~is_too_similar]
    return embeddings

def filter_embeddings_by_mlp_results(embeddings: Embeddings, description_mlp_results: List[int]) -> Embeddings:
    """Filter the embeddings based on the description MLP results."""
    valid_indices = [i for i, result in enumerate(description_mlp_results) if result > 1]
    valid_indices = torch.tensor(valid_indices, dtype=torch.long)
    if embeddings.video is not None:
        embeddings.video = embeddings.video[valid_indices]
    if embeddings.audio is not None:
        embeddings.audio = embeddings.audio[valid_indices]
    if embeddings.description is not None:
        embeddings.description = embeddings.description[valid_indices]
    return embeddings

def is_similar(emb_1: torch.Tensor, emb_2: List[float]) -> bool:
    return F.cosine_similarity(
        emb_1,
        torch.tensor(emb_2, device=emb_1.device).unsqueeze(0)
    ) > SIMILARITY_THRESHOLD


def strict_is_similar(emb_1: torch.Tensor, emb_2: List[float]) -> bool:
    return torch.allclose(emb_1, torch.tensor(emb_2, device=emb_1.device), atol=1e-4)


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


async def get_random_video(metadata: List[VideoMetadata], check_video: bool) -> Optional[Tuple[VideoMetadata, Optional[BinaryIO]]]:
    if not check_video:
        random_metadata = random.choice(metadata)
        return random_metadata, None

    random_video = None
    metadata_copy = [v for v in metadata]  # list shallow copy
    while random_video is None and len(metadata_copy) > 0:
        idx = random.randint(0, len(metadata_copy) - 1)
        random_metadata = metadata_copy.pop(idx)
        try:
            async with DOWNLOAD_SEMAPHORE:
                random_video = await asyncio.wait_for(run_async(
                    video_utils.download_youtube_video,
                    random_metadata.video_id,
                    random_metadata.start_time,
                    random_metadata.end_time,
                    proxy=get_proxy_url(),
                ), timeout=VIDEO_DOWNLOAD_TIMEOUT)
        except video_utils.IPBlockedException:
            # IP is blocked, cannot download video, check description only
            print("WARNING: IP is blocked, cannot download video, checking description only")
            return random_metadata, None
        except video_utils.FakeVideoException:
            print(f"WARNING: Video {random_metadata.video_id} is fake, punishing miner")
            return None
        except asyncio.TimeoutError:
            continue

    # IP is not blocked, video is not fake, but video download failed for some reason. We don't
    # know why it failed so we won't punish the miner, but we will check the description only.
    if random_video is None:
        return random_metadata, None

    return random_metadata, random_video


async def random_check(random_meta_and_vid: List[VideoMetadata], imagebind: ImageBind) -> bool:
    random_metadata, random_video = random_meta_and_vid

    if random_video is None:
        desc_embeddings = await imagebind.embed_text_async([random_metadata.description])
        is_similar_ = is_similar(desc_embeddings, random_metadata.description_emb)
        strict_is_similar_ = strict_is_similar(desc_embeddings, random_metadata.description_emb)
        print(f"Description similarity: {is_similar_}, strict description similarity: {strict_is_similar_}")
        return is_similar_

    # Video downloaded, check all embeddings
    embeddings = await imagebind.embed_async([random_metadata.description], [random_video])
    is_similar_ = (
        is_similar(embeddings.video, random_metadata.video_emb) and
        is_similar(embeddings.audio, random_metadata.audio_emb) and
        is_similar(embeddings.description, random_metadata.description_emb)
    )
    strict_is_similar_ = (
        strict_is_similar(embeddings.video, random_metadata.video_emb) and
        strict_is_similar(embeddings.audio, random_metadata.audio_emb) and
        strict_is_similar(embeddings.description, random_metadata.description_emb)
    )
    print(f"Total similarity: {is_similar_}, strict total similarity: {strict_is_similar_}")
    return is_similar_


async def get_num_unique_videos(videos: Videos) -> int:
    metadata = videos.video_metadata
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]),
    )
    novelty_score, is_too_similar = await compute_novelty_score(embeddings)
    return sum([not is_sim for is_sim in is_too_similar])


async def _run_video_scoring(videos: Videos, imagebind: ImageBind, is_check_only: bool) -> float:
    if any(not video_utils.is_valid_youtube_id(video.video_id) for video in videos.video_metadata):
        return {"score": FAKE_VIDEO_PUNISHMENT}

    metadata = metadata_check(videos.video_metadata)[:videos.num_videos]
    print(f"Filtered {len(videos.video_metadata)} videos down to {len(metadata)} videos")

    if len(metadata) == 0:
        return {"score": MIN_SCORE}

    check_video = config.CHECK_PROBABILITY > random.random()
    random_meta_and_vid = await get_random_video(metadata, check_video)
    if random_meta_and_vid is None:
        return {"score": FAKE_VIDEO_PUNISHMENT}

    async with GPU_SEMAPHORE:
        passed_check = await random_check(random_meta_and_vid, imagebind)
        if not passed_check:
            return {"score": FAKE_VIDEO_PUNISHMENT}
        query_emb = await imagebind.embed_text_async([videos.query])

    # Upload the videos to Pinecone and deduplicate
    original_length = len(metadata)
    embeddings = Embeddings(
        video=torch.stack([torch.tensor(v.video_emb) for v in metadata]).to(imagebind.device),
        audio=torch.stack([torch.tensor(v.audio_emb) for v in metadata]).to(imagebind.device),
        description=torch.stack([torch.tensor(v.description_emb) for v in metadata]).to(imagebind.device),
    )
    novelty_score, is_too_similar = await compute_novelty_score(embeddings)
    embeddings = filter_embeddings(embeddings, is_too_similar)
    metadata = [metadata for metadata, too_similar in zip(metadata, is_too_similar) if not too_similar]
    print(f"Deduplicated {original_length} videos down to {len(metadata)} videos")

    original_length = len(metadata)
    # Compute description scores using the imagebind_desc_mlp model
    description_mlp_results = [imagebind_desc_mlp.get_desc_embedding_score(embedding) for embedding in embeddings.description]
    print(f"description_mlp_results: {description_mlp_results}")
    # filter out metadata that have description scores of 1
    metadata = [metadata for metadata, desc_mlp_result in zip(metadata, description_mlp_results) if desc_mlp_result > 1]
    # filter out embeddings that have description scores of 1
    embeddings = filter_embeddings_by_mlp_results(embeddings, description_mlp_results)
    # filter out description scores that are 1
    filtered_description_mlp_results = [desc_mlp_result for desc_mlp_result in description_mlp_results if desc_mlp_result > 1]
    if len(metadata) < original_length:
        print(f"Filtering {original_length} videos down to {len(metadata)} videos that had poor descriptions.")

    # Compute relevance scores
    description_relevance_scores = F.cosine_similarity(
        embeddings.video, embeddings.description
    ).tolist()
    query_relevance_scores = F.cosine_similarity(
        embeddings.video, query_emb
    ).tolist()

    description_mlp_scores = []
    # Apply penalties and store the penalized scores
    for desc_score, desc_mlp_score in zip(description_relevance_scores, filtered_description_mlp_results):
        if desc_mlp_score == 4 or desc_mlp_score == 5:
            # score of 4 or 5 is "good", reward with 40% or 50% description relevance score boost, respectfully
            print("Good description detected, thank you honest miner.")
            rewarded_score = (desc_mlp_score * 0.1) * desc_score
            description_mlp_scores.append(rewarded_score)
        elif desc_mlp_score == 2:
            # score of 2 is "poor", penalize with 50% description relevance score penalty
            print("Poor description detected, please do better.")
            description_mlp_scores.append(desc_score * -0.5)
        elif desc_mlp_score == 1:
            # score of 1 is "bad", penalize full description relevance score penalty
            print("Bad description detected, omitting submission.")
            description_mlp_scores.append(desc_score * -1.0)
        else:
            # score of 3 is "OK", no reward or penalty
            print("This description is OK, but please improve.")
            description_mlp_scores.append(0)
    print(f"description_mlp_scores: {description_mlp_scores}")

    # Aggregate scores
    score = (
        ((sum(description_relevance_scores) + sum(description_mlp_scores)) * DESCRIPTION_RELEVANCE_SCALING_FACTOR) +
        (sum(query_relevance_scores) * QUERY_RELEVANCE_SCALING_FACTOR)
    ) / 2 / videos.num_videos

    if not is_check_only and len(metadata) > 0:
        video_ids = await run_async(upload_to_pinecone, embeddings, metadata)
        # Schedule upload to HuggingFace
        dataset_uploader.add_videos(
            metadata,
            video_ids,
            description_relevance_scores,
            query_relevance_scores,
            videos.query,
        )
    score = max(score, MIN_SCORE)

    if score > 0.4:
        print(f"Videos with score > 0.4: {metadata}")

    return {
        "is_unique": [not is_sim for is_sim in is_too_similar],
        "description_relevance_scores": description_relevance_scores,
        "query_relevance_scores": query_relevance_scores,
        "score": score,
    }


async def score_videos_for_testing(videos: Videos, imagebind: ImageBind) -> float:
    return await _run_video_scoring(videos, imagebind, is_check_only=True)


async def score_and_upload_videos(videos: Videos, imagebind: ImageBind) -> float:
    scores_dict = await _run_video_scoring(videos, imagebind, is_check_only=False)
    return scores_dict["score"]
