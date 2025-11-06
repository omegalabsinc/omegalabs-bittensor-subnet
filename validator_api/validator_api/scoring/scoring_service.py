"""
Description of scoring system:
- Phase 0: generate detailed annotation for video;
- Phase 1: spam detection + rejection (can order from least to greatest cost)
        - Working:
        - length of video (too long or short)
                - uniqueness detection (video embedding vector similarity)
                - chat-only detection (openai o1 + text description)
        - Not working:
                - YouTube/movie video-watching detection (gemini + first and last video chunks)
                - exploit/screen recording video watching detection (gemini + first and last video chunks)
                        - prompt can be found in old subnet commits
                - automation detection (??) (I don't think this is reliably working yet)
- Phase 2: actual scoring
        - can be gemini evaluation on the whole video, but I think it's probably more cost-efficient to use a reasoning model with the task descriptions
"""

import asyncio
import json
import random
import time
from typing import List, Optional, Tuple

import vertexai
from openai import AsyncOpenAI
from pinecone import Pinecone
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from validator_api.validator_api.config import (
    GOOGLE_CLOUD_BUCKET_NAME,
    GOOGLE_LOCATION,
    GOOGLE_PROJECT_ID,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
)
from validator_api.validator_api.database import get_db_context
from validator_api.validator_api.database.models.boosted_task import BoostedTask
from validator_api.validator_api.database.models.focus_video_record import (
    FocusVideoInternal,
    FocusVideoRecord,
)
from validator_api.validator_api.database.models.real_focused_time import (
    RealFocusedTime,
)
from validator_api.validator_api.database.models.scoring import (
    CompletionScore,
    CompletionScoreWithoutRange,
    DetailedVideoDescription,
    FocusVideoEmbeddings,
    LegitimacyCheckError,
    VideoScore,
    VideoTooLongError,
    VideoTooShortError,
    VideoUniquenessError,
)
from validator_api.validator_api.database.models.task import TaskRecordPG
from validator_api.validator_api.scoring import focus_scoring_prompts
from validator_api.validator_api.scoring.legitimacy_checks import ChatOnlyCheck
from validator_api.validator_api.scoring.query_llm import query_llm
from validator_api.validator_api.utils import run_async, run_with_retries
from vertexai.generative_models import Part
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.vision_models import MultiModalEmbeddingModel, Video, VideoSegmentConfig

ONE_MINUTE = 60  # in seconds
NINETY_MINUTES = 5400  # in seconds
FOCUS_VIDEO_MIN_SCORE = 0.05
FOCUS_VIDEO_MAX_SCORE = 1.0
MIN_VIDEO_UNIQUENESS_SCORE = 0.02


async def get_video_metadata(
    db: AsyncSession, video_id: str
) -> Optional[FocusVideoInternal]:
    query = select(FocusVideoRecord).filter(FocusVideoRecord.video_id == video_id)
    result = await db.execute(query)
    video = result.scalar_one_or_none()

    if video and video.deleted_at is not None:
        print(f"Video {video_id} has been deleted")
        return None

    return video


async def _get_details_if_boosted(video_id: str) -> Optional[BoostedTask]:
    """
    Retrieves the details of a boosted task from the database for a given video.

    This function prevents task manipulation by using the stored boosted task information
    directly from the database instead of user-provided values. It first looks up the
    task associated with the video, then checks if that task is boosted.

    Args:
        video_id (str): The ID of the video to check for a boosted task.

    Returns:
        Optional[BoostedTask]: The boosted task details if the video's task is boosted,
            including multiplier, title, and description. Returns None if the video
            is not associated with a boosted task.
    """
    async with get_db_context() as db:
        video_metadata = await get_video_metadata(db, video_id)
        if video_metadata and video_metadata.task_id:
            query = select(TaskRecordPG).filter(
                TaskRecordPG.id == video_metadata.task_id,
            )
            result = await db.execute(query)
            task = result.scalar_one_or_none()

            if task and task.boosted_id:
                query = select(BoostedTask).filter(
                    BoostedTask.id == task.boosted_id,
                )
                result = await db.execute(query)
                return result.scalar_one_or_none()
    return None


async def get_video_duration_seconds(video_id: str) -> int:
    async with get_db_context() as db:
        # First verify video exists and is not deleted
        video_metadata = await get_video_metadata(db, video_id)
        if video_metadata is None:
            raise ValueError(f"Focus video is deleted or doesn't exist: {video_id}")

        # Query real_focused_time table for actual focused duration
        query = select(RealFocusedTime).filter(RealFocusedTime.video_id == video_id)
        result = await db.execute(query)
        real_focused_time = result.scalar_one_or_none()

        if real_focused_time is None or real_focused_time.real_focused_duration is None:
            print(f"Video duration not found in real_focused_time for video: {video_id}")
            video_duration_seconds = 120  # Default fallback
        else:
            # Convert from milliseconds to seconds
            video_duration_seconds = int(real_focused_time.real_focused_duration / 1000)

        # print(f" Video Duration ")
        return video_duration_seconds


def get_s3_path(video_id: str) -> str:
    return f"clips/{video_id}.webm"


def get_gcs_uri(video_id: str) -> str:
    return f"gs://{GOOGLE_CLOUD_BUCKET_NAME}/{get_s3_path(video_id)}"


async def _make_gemini_request(
    system_prompt: str, user_prompt: str, video_id: str, OutputClassSchema: BaseModel
) -> GenerativeModel:
    """
    Makes a request to the Gemini model with specified prompts and video content.

    Args:
        system_prompt (str): The system instruction for the model
        user_prompt (str): The user prompt/query for the model
        video_id (str): The ID of the video to analyze, if any
        OutputClassSchema (BaseModel): Pydantic model class defining the expected response structure

    Returns:
        An instance of OutputClassSchema containing the parsed model response
    """
    model_name = "gemini-2.5-flash"
    # print(f"Video text annotation is using model: {model_name}")
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    model = GenerativeModel(
        model_name,
        system_instruction=system_prompt.strip(),
        safety_settings=safety_settings,
        generation_config=GenerationConfig(
            temperature=1,
            response_mime_type="application/json",
            response_schema=OutputClassSchema.model_json_schema(),
        ),
    )

    parts = []
    if video_id:
        gcs_uri = get_gcs_uri(video_id)
        print(f"Making Gemini request with GCS URI: {gcs_uri}")
        parts.append(Part.from_uri(gcs_uri, mime_type="video/webm"))
    parts.append(user_prompt.strip())

    response = await model.generate_content_async(parts)
    return OutputClassSchema(**json.loads(response.text))


async def _make_gemini_request_with_retries(
    system_prompt: str, user_prompt: str, video_id: str, OutputClassSchema: BaseModel
) -> str:
    """
    Makes a request to Gemini with automatic retries on failure.

    Handles JSON parsing errors, validation errors, and general exceptions with 3 retry attempts.
    Delays between retries.

    Args:
        system_prompt (str): The system instruction for the model
        user_prompt (str): The user prompt/query for the model
        video_id (str): The ID of the video to analyze, if any
        OutputClassSchema (BaseModel): Pydantic model class defining the expected response structure

    Returns:
        An instance of OutputClassSchema containing the parsed model response

    Raises:
        Exception: If all retry attempts fail
    """
    num_retries = 3
    for retry_idx in range(num_retries):
        try:
            start = time.time()
            output = await _make_gemini_request(
                system_prompt, user_prompt, video_id, OutputClassSchema
            )
            print(
                f"Got gemini output in {time.time() - start} seconds for {OutputClassSchema.__name__}"
            )
            return output
        except json.JSONDecodeError as e:
            print(
                f"Error parsing JSON from Gemini response for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})"
            )
            await asyncio.sleep(1)
        except ValidationError as e:
            print(
                f"Error turning parsed JSON into Pydantic object for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})"
            )
            await asyncio.sleep(1)
        except Exception as e:
            print(
                f"Error making Gemini request for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})"
            )
            await asyncio.sleep(6)
    raise Exception(
        f"Failed to turn Gemini response into JSON and then into Pydantic object for {OutputClassSchema.__name__} after {num_retries} attempts"
    )


async def get_detailed_video_description(
    video_id: str, task_overview: str, recompute: bool = False
) -> DetailedVideoDescription:
    print(f"get_detailed_video_description called for video_id: {video_id} (length: {len(video_id)})")

    if not recompute:
        async with (
            get_db_context() as db
        ):  # get already computed description from db if it exists
            query = select(FocusVideoRecord).filter(
                FocusVideoRecord.video_id == video_id,
                FocusVideoRecord.deleted_at.is_(None),
            )
            result = await db.execute(query)
            video_record = result.scalar_one_or_none()

            if video_record is None:
                raise ValueError(f"Video not found: {video_id}")

            if (
                video_record.video_details
                and "detailed_video_description" in video_record.video_details
            ):
                print(f"Using cached detailed_video_description for video_id: {video_id}")
                return DetailedVideoDescription.model_validate(
                    video_record.video_details["detailed_video_description"]
                )

    print(f"Generating new detailed_video_description for video_id: {video_id}")
    description = await _make_gemini_request_with_retries(
        system_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_SYSTEM_PROMPT,
        user_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_USER_PROMPT.format(
            task_overview=task_overview
        ),
        video_id=video_id,
        OutputClassSchema=DetailedVideoDescription,
    )

    return description


# async def get_task_score_from_gemini(self, task_overview: str) -> TaskScoreBreakdown:
#     return await _make_gemini_request_with_retries(
#         system_prompt=focus_scoring_prompts.TASK_SCORE_SYSTEM_PROMPT,
#         user_prompt=focus_scoring_prompts.TASK_SCORE_USER_PROMPT.format(task_overview=task_overview),
#         video_id=None,
#         OutputClassSchema=TaskScoreBreakdown,
#     )


async def _get_completion_score_breakdown(
    task_overview: str,
    detailed_video_description: Optional[DetailedVideoDescription] = None,
    system_prompt: str = focus_scoring_prompts.DESC_ONLY_TASK_COMPLETION_SYSTEM_PROMPT.format(
        EXPLOITED_TASK_CASES=focus_scoring_prompts.EXPLOITED_TASK_CASES
    ),
    user_prompt: str = focus_scoring_prompts.DESC_ONLY_TASK_COMPLETION_USER_PROMPT,
) -> CompletionScore:
    """
    This function generates a completion score breakdown using the DeepSeek model via Chutes API.

    Args:
        task_overview (str): An overview of the task associated with the video.
        openai_client (AsyncOpenAI): Kept for compatibility but no longer used
        detailed_video_description (Optional[DetailedVideoDescription], optional): A detailed description of the video content.
        system_prompt (str, optional): The system prompt to be used for generating the completion score.
        user_prompt (str, optional): The user prompt to be used for generating the completion score.

    Returns:
        CompletionScore: The completion score breakdown for the video.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_prompt.format(
                task_overview=task_overview,
                applications_used=detailed_video_description.applications_used,
                completion_sequence_steps=detailed_video_description.completion_sequence_steps,
            ),
        },
    ]

    try:
        completion_score_without_range = await query_llm(
            messages=messages,
            # OpenAI API doesn't like it when there's a range in the Pydantic model
            output_model=CompletionScoreWithoutRange,
        )
        return CompletionScore(
            rationale=completion_score_without_range.rationale,
            completion_score=max(
                0.0, min(1.0, completion_score_without_range.completion_score)
            ),
        )

    except Exception as e:
        print(f"Error getting completion score: {str(e)}")
        raise


async def get_video_embedding(
    video_id: str, video_duration_seconds: int
) -> List[float]:
    """
    Generates an embedding vector for a video segment using Google's multimodal embedding model.

    Takes a random 120-second segment from the video if the video is longer than 2 minutes.

    Args:
        video_id (str): The ID of the video to embed
        video_duration_seconds (int): The total duration of the video in seconds

    Returns:
        List[float]: The embedding vector for the video segment
    """

    async def _internal_async():
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
        start_offset_sec = random.randint(0, max(0, video_duration_seconds - 120))
        end_offset_sec = min(video_duration_seconds, start_offset_sec + 120)
        embeddings = await run_async(
            model.get_embeddings,
            video=Video.load_from_file(get_gcs_uri(video_id)),
            video_segment_config=VideoSegmentConfig(
                start_offset_sec=start_offset_sec,
                end_offset_sec=end_offset_sec,
                interval_sec=end_offset_sec - start_offset_sec,
            ),
        )
        return embeddings.video_embeddings[0].embedding

    return await run_with_retries(_internal_async)


async def query_pinecone(pinecone_index: Pinecone, vector: List[float]) -> float:
    """
    Queries a Pinecone index with a vector to find the most similar existing vector.

    Returns a uniqueness score based on the inverse of the similarity score (1 - similarity).
    Ensures the returned score is between 0 and 1.

    Args:
        pinecone_index (Pinecone): The Pinecone index to query
        vector (List[float]): The embedding vector to search for

    Returns:
        float: The uniqueness score (1 - similarity score)
    """

    async def _internal_async():
        response = await run_async(
            pinecone_index.query,
            vector=vector,
            top_k=1,
        )
        if len(response["matches"]) > 0:
            matches = response["matches"]
            similarity_score = matches[0]["score"]
            # for match in matches:
            #     print(f"Match:")
            #     print(f"  - Score: {match['score']}")
            #     print(f"  - ID: {match.get('id', 'N/A')}")
            #     print(f"  - Metadata: {match.get('metadata', {})}")
        else:
            # print("No pinecone matches, returning 0")
            similarity_score = 0
        similarity_score = max(0.0, min(similarity_score, 1.0))
        return 1.0 - similarity_score

    return await run_with_retries(_internal_async)


class FocusScoringService:
    def __init__(self):
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        self.task_overview_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-task-overview-index"
        )
        self.video_description_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-video-description-index"
        )
        self.completion_video_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-completion-video-index"
        )
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.legitimacy_checks = [ChatOnlyCheck()]

    # Pinecone is used for similarity search and scoring (uniqueness check)
    async def _get_task_uniqueness_score(
        self, task_overview_embedding: List[float]
    ) -> float:
        return await query_pinecone(self.task_overview_index, task_overview_embedding)

    async def get_description_uniqueness_score(
        self, detailed_video_description_embedding: List[float]
    ) -> float:
        return await query_pinecone(
            self.video_description_index, detailed_video_description_embedding
        )

    async def get_video_uniqueness_score(self, video_embedding: List[float]) -> float:
        return await query_pinecone(self.completion_video_index, video_embedding)

    # Embedding related functions
    async def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding vector for text using OpenAI's embedding model.

        Implements timeout and retry logic for reliability.

        Args:
            text (str): The text to generate an embedding for

        Returns:
            Optional[List[float]]: The embedding vector, or None if the request fails
        """

        async def _internal_async():
            response = await asyncio.wait_for(
                self.openai_client.embeddings.create(
                    input=text, model="text-embedding-3-large"
                ),
                timeout=10,
            )
            return response.data[0].embedding

        try:
            return await run_with_retries(_internal_async)
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return None

    async def embed_and_get_task_uniqueness_score(
        self, task_overview: str
    ) -> Tuple[Optional[List[float]], Optional[float]]:
        embedding = await self.get_text_embedding(task_overview)
        if embedding is None:
            return None, None
        return embedding, await self._get_task_uniqueness_score(embedding)

    async def embed_and_get_video_uniqueness_score(
        self, video_id: str, video_duration_seconds: int
    ):
        try:
            embedding = await get_video_embedding(video_id, video_duration_seconds)
            return embedding, await self.get_video_uniqueness_score(embedding)
        except Exception as e:
            print(f"Failed to create video embedding for {video_id}: {str(e)}")
            return None, 0.1  # Assumes unique if we can't check

    async def get_detailed_video_description_embedding_score(
        self, video_id, task_overview
    ):
        detailed_video_description = await get_detailed_video_description(
            video_id, task_overview
        )
        embedding = await self.get_text_embedding(
            detailed_video_description.model_dump_json()
        )
        if embedding is None:
            return detailed_video_description, None, None
        return (
            detailed_video_description,
            embedding,
            await self.get_description_uniqueness_score(embedding),
        )

    async def score_video(
        self,
        video_id: str,
        focusing_task: str,
        focusing_description: str,
        bypass_checks: bool = False,
    ) -> Tuple[VideoScore, FocusVideoEmbeddings]:
        """
        Generates a comprehensive score for a video submission based on multiple factors.

        The scoring process includes:
        1. Checking video duration constraints
        2. Computing task, description, and video uniqueness scores
        3. Running legitimacy checks (marks as REJECTED in DB if fails, returns score of 0)
        4. Generating a completion score
        5. Applying any boost multipliers

        Args:
            video_id (str): The ID of the video to score
            focusing_task (str): The title of the task
            focusing_description (str): The detailed description of the task
            bypass_checks (bool): If True, skip duration and legitimacy checks

        Returns:
            Tuple[VideoScore, FocusVideoEmbeddings]: The complete scoring details
            and computed embeddings. If video fails legitimacy checks, returns with
            final_score=0 and rejection reason in completion_score_breakdown.rationale

        Raises:
            ValueError: If video duration is outside acceptable range
            VideoUniquenessError: If video uniqueness score is too low
        """
        boosted_task = await _get_details_if_boosted(video_id)
        if boosted_task:
            boosted_multiplier = boosted_task.multiplier
            focusing_task = boosted_task.title
            focusing_description = boosted_task.description
        else:
            boosted_multiplier = 1.0

        video_duration_seconds = await get_video_duration_seconds(video_id)
        video_minutes = video_duration_seconds / 1000;
        print(f"video minutes {video_minutes}")
        # if not bypass_checks:
        if video_duration_seconds < ONE_MINUTE:
                raise VideoTooShortError(
                    f"Video duration is too short: {video_duration_seconds} seconds"
                )

        if video_duration_seconds > NINETY_MINUTES:
                raise VideoTooLongError(
                    f"Video duration is too long: {video_duration_seconds} seconds"
                )

        task_overview = f"# {focusing_task}\n\n{focusing_description}"

        (
            (task_overview_embedding, task_uniqueness_score),
            # task_score_breakdown,
            (
                video_description,
                video_description_embedding,
                video_description_uniqueness_score,
            ),
            (video_embedding, video_uniqueness_score),
        ) = await asyncio.gather(
            self.embed_and_get_task_uniqueness_score(
                task_overview
            ),  # uses openai to get embedding
            # self.get_task_score_from_gemini(task_overview),  # uses gemini to score task
            self.get_detailed_video_description_embedding_score(
                video_id, task_overview
            ),  # uses gemini to get detailed description
            self.embed_and_get_video_uniqueness_score(video_id, video_duration_seconds),
        )

        if not bypass_checks:
            if video_uniqueness_score < MIN_VIDEO_UNIQUENESS_SCORE:
                raise VideoUniquenessError("Video uniqueness score is too low.")

            if self.legitimacy_checks:
                check_results = await asyncio.gather(
                    *(
                        check.passes_check(video_id, video_description)
                        for check in self.legitimacy_checks
                    )
                )

                for passed, failure_reason in check_results:
                    if not passed:
                        # Return VideoScore with 0 final_score to indicate legitimacy check failure
                        # The actual database update will be handled by app.py
                        print(f"Video {video_id} failed legitimacy check: {failure_reason}")
                        return VideoScore(
                            task_uniqueness_score=task_uniqueness_score,
                            video_completion_score=0.0,
                            description_uniqueness_score=video_description_uniqueness_score,
                            video_uniqueness_score=video_uniqueness_score,
                            boosted_multiplier=boosted_multiplier,
                            final_score=0.0,
                            task_overview=task_overview,
                            completion_score_breakdown=CompletionScore(
                                rationale=failure_reason,
                                completion_score=0.0,
                            ),
                            detailed_video_description=video_description,
                        ), FocusVideoEmbeddings(
                            task_overview_embedding=task_overview_embedding,
                            detailed_video_description_embedding=video_description_embedding,
                            video_embedding=video_embedding,
                        )

        completion_score_breakdown = await _get_completion_score_breakdown(
            task_overview,
            detailed_video_description=video_description,
        )

        completion_gemini_score = completion_score_breakdown.completion_score
        final_score = completion_gemini_score * boosted_multiplier

        print(f"Final score: {final_score}")
        print(f"completion score breakdown: {completion_score_breakdown}")

        return VideoScore(
            task_uniqueness_score=task_uniqueness_score,
            video_completion_score=completion_gemini_score,
            description_uniqueness_score=video_description_uniqueness_score,
            video_uniqueness_score=video_uniqueness_score,
            boosted_multiplier=boosted_multiplier,
            final_score=final_score,
            task_overview=task_overview,
            completion_score_breakdown=completion_score_breakdown,
            detailed_video_description=video_description,
        ), FocusVideoEmbeddings(
            task_overview_embedding=task_overview_embedding,
            detailed_video_description_embedding=video_description_embedding,
            video_embedding=video_embedding,
        )


def main():
    service = FocusScoringService()
    import asyncio

    async def main():
        video_id = "29f91a6f-1393-4765-ba00-263b4cff28b6"
        task_overview = """
# Multimodal tokenization research

Read the Show-O peper to understand how they have trained a unified diffusion and autoregressive model for multimodal tokenization.
""".strip()

        score_details = await service.score_video(
            video_id, task_overview, "description"
        )
        print(score_details)

        # task_overview_embedding = await service.get_text_embedding(task_overview)
        # print(len(task_overview_embedding))

        # detailed_video_description = DetailedVideoDescription(
        #     applications_used=[],
        #     completion_sequence_steps=[],
        #     user_feedback="",
        #     description=""
        # )

        # video_embedding = await service.get_video_embedding(video_id, 1740)
        # print(f"Sum: {sum(video_embedding)}, min: {min(video_embedding)}, max: {max(video_embedding)}")

        # task_score_breakdown = await service.get_task_score_from_gemini(task_overview)
        # print(task_score_breakdown)

        # completion_score_breakdown = await service.get_completion_score_breakdown(video_id, task_overview, detailed_video_description=None)
        # print(completion_score_breakdown)

        # start = time.time()
        # model = service.get_model_cached_on_video(video_id)
        # print(f"Got model in {time.time() - start} seconds")
        # for _ in range(4):
        #     start = time.time()
        #     video_description = await service.get_detailed_video_description_from_cache(model)
        #     print(f"Got detailed video description ({video_description}) in {time.time() - start} seconds")

        # for _ in range(4):
        #     start = time.time()
        #     video_description = await service.get_detailed_video_description(video_id)
        #     print(f"Got detailed video description ({video_description}) in {time.time() - start} seconds")

    asyncio.run(main())


if __name__ == "__main__":
    main()
