import asyncio
import random
import uuid
from typing import List, Tuple, Optional, BinaryIO
import math

from pinecone import Pinecone
import torch
import torch.nn.functional as F

from omega.protocol import Videos, VideoMetadata, FocusVideoMetadata
from omega import video_utils, unstuff
from omega.constants import (
    MAX_VIDEO_LENGTH, 
    MIN_VIDEO_LENGTH,
    DIFFERENCE_THRESHOLD, 
    SIMILARITY_THRESHOLD, 
    VIDEO_DOWNLOAD_TIMEOUT, 
    MIN_SCORE, 
    FAKE_VIDEO_PUNISHMENT,
    QUERY_RELEVANCE_SCALING_FACTOR,
    DESCRIPTION_RELEVANCE_SCALING_FACTOR,
    VIDEO_RELEVANCE_WEIGHT,
    DESCRIPTION_LENGTH_WEIGHT,
    MIN_LENGTH_BOOST_TOKEN_COUNT,
    MAX_LENGTH_BOOST_TOKEN_COUNT,
    STUFFED_DESCRIPTION_PUNISHMENT,
)
from omega.imagebind_wrapper import ImageBind, Embeddings, run_async, LENGTH_TOKENIZER, IMAGEBIND_VERSION
from omega.text_similarity import get_text_similarity_score
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
    dataset_uploader.add_videos(
        metadata,
        video_ids,
        description_relevance_scores,
        query_relevance_scores,
        query,
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


def filter_stuffed_embeddings(embeddings: Embeddings, stuffed: List[Tuple[bool, float]]) -> Embeddings:
    """Filter the embeddings based on whether they are too similar to the query."""
    stuffed = torch.tensor([s for s, _ in stuffed])
    if embeddings.video is not None:
        embeddings.video = embeddings.video[~stuffed]
    if embeddings.audio is not None:
        embeddings.audio = embeddings.audio[~stuffed]
    if embeddings.description is not None:
        embeddings.description = embeddings.description[~stuffed]
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
    
    # check video_ids for fake videos
    if any(not video_utils.is_valid_youtube_id(video.video_id) for video in videos.video_metadata):
        return {"score": FAKE_VIDEO_PUNISHMENT}
    
    metadata = metadata_check(videos.video_metadata)[:videos.num_videos]
    print(f"Filtered {len(videos.video_metadata)} videos down to {len(metadata)} videos")

    # return minimum score if no videos were found in video_metadata
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

    # Filter out "stuffed" descriptions.
    pre_filter_metadata_length = len(metadata)
    stuffed = [
        unstuff.is_stuffed(meta.description)
        for meta in metadata
    ]
    if any([garbage and confidence > 0.75 for garbage, confidence in stuffed]):
        print("Stuffed description found with high confidence, penalizing the miner.")
        return {"score": STUFFED_DESCRIPTION_PUNISHMENT}

    metadata = [
        metadata[idx]
        for idx in range(len(metadata))
        if not stuffed[idx][0]
    ]
    if len(metadata) < pre_filter_metadata_length:
        print(f"Filtering {pre_filter_metadata_length} videos down to {len(metadata)} videos to remove token-stuffed descriptions.")
    if len(metadata) == 0:
        return {"score": MIN_SCORE}

    embeddings = filter_stuffed_embeddings(embeddings, stuffed)

    # Compute relevance scores
    video_description_relevance_scores = F.cosine_similarity(
        embeddings.video, embeddings.description
    ).tolist()
    audio_description_relevance_scores = F.cosine_similarity(
        embeddings.audio, embeddings.description
    ).tolist()
    video_query_relevance_scores = F.cosine_similarity(
        embeddings.video, query_emb
    ).tolist()
    audio_query_relevance_scores = F.cosine_similarity(
        embeddings.audio, query_emb
    ).tolist()

    # Query relevance score now includes video cosim, audio cosim, and text cosim using higher quality text-only model.
    query_relevance_scores = [
        sum([
            video_query_relevance_scores[idx],
            audio_query_relevance_scores[idx],
            get_text_similarity_score(metadata[idx].description, videos.query),
        ]) / 3
    ]

    # Combine audio & visual description scores, weighted towards visual.
    description_relevance_scores = [
        sum([
            video_description_relevance_scores[idx] * VIDEO_RELEVANCE_WEIGHT,
            audio_description_relevance_scores[idx] * (1.0 - VIDEO_RELEVANCE_WEIGHT),
        ])
        for idx in range(len(video_description_relevance_scores))
    ]

    # Scale description scores by number of unique tokens.
    length_scalers = []
    for idx in range(len(description_relevance_scores)):
        unique_token_count = len(set(LENGTH_TOKENIZER(metadata[idx].description).nonzero()))
        if unique_token_count <= MIN_LENGTH_BOOST_TOKEN_COUNT:
            print(f"Very few tokens, applying {DESCRIPTION_LENGTH_WEIGHT} penalty.")
            description_relevance_scores[idx] *= (1.0 - DESCRIPTION_LENGTH_WEIGHT)
            length_scalers.append(0)
            continue
        length_scaler = min(math.log(MAX_LENGTH_BOOST_TOKEN_COUNT, 2), math.log(unique_token_count, 2)) - math.log(MIN_LENGTH_BOOST_TOKEN_COUNT, 2)
        length_scaler /= (math.log(MAX_LENGTH_BOOST_TOKEN_COUNT, 2) - math.log(MIN_LENGTH_BOOST_TOKEN_COUNT, 2))
        length_scalers.append(length_scaler)
        print(f"Description length scaling factor = {length_scaler}")
        description_relevance_scores[idx] -= description_relevance_scores[idx] * DESCRIPTION_LENGTH_WEIGHT * (1.0 - length_scaler)

    # Aggregate scores
    score = (
        (sum(description_relevance_scores) * DESCRIPTION_RELEVANCE_SCALING_FACTOR) +
        (sum(query_relevance_scores) * QUERY_RELEVANCE_SCALING_FACTOR)
    ) / 2 / videos.num_videos

    print(f'''
        is_unique: {[not is_sim for is_sim in is_too_similar]},
        video cosine sim: {video_description_relevance_scores},
        audio cosine sim: {audio_description_relevance_scores},
        description relevance scores: {description_relevance_scores},
        query relevance scores: {query_relevance_scores},
        length scalers: {length_scalers},
        total score: {score}
    ''')

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
