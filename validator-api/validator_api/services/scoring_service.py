import asyncio
from typing import List, Optional
import json
import random
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session
import vertexai
from vertexai.generative_models import Part
from vertexai.preview import caching
from vertexai.preview.generative_models import (
    GenerativeModel, HarmCategory, HarmBlockThreshold, GenerationConfig,
)
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from vertexai.vision_models import VideoSegmentConfig
from pinecone import Pinecone

from validator_api.config import GOOGLE_PROJECT_ID, GOOGLE_LOCATION, OPENAI_API_KEY, GOOGLE_CLOUD_BUCKET_NAME, PINECONE_API_KEY
from validator_api.services import focus_scoring_prompts
from validator_api.utils import run_async, run_with_retries
from validator_api.database import get_db_context
from validator_api.database.models.focus_video_record import FocusVideoRecord, FocusVideoInternal
from validator_api.database.models.boosted_task import BoostedTask
from validator_api.database.models.task import TaskRecordPG

from typing import Tuple, Optional

TWO_MINUTES = 120  # in seconds
NINETY_MINUTES = 5400  # in seconds
FOCUS_VIDEO_MIN_SCORE = 0.05
FOCUS_VIDEO_MAX_SCORE = 1.0
MIN_VIDEO_UNIQUENESS_SCORE = 0.02

def get_video_metadata(db: Session, video_id: str) -> Optional[FocusVideoInternal]:
    return db.query(FocusVideoRecord).filter(
        FocusVideoRecord.video_id == video_id,
        FocusVideoRecord.deleted_at.is_(None)
    ).first()

async def query_pinecone(pinecone_index: Pinecone, vector: List[float]) -> float:
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
            print(f"No pinecone matches, returning 0")
            similarity_score = 0
        similarity_score = max(0.0, min(similarity_score, 1.0))
        return 1.0 - similarity_score
    return await run_with_retries(_internal_async)

class VideoUniquenessError(Exception):
    pass

class TaskScoreBreakdown(BaseModel):
    reasoning_steps: List[str] = Field(description="Steps of reasoning used to arrive at the final score. Before each step, write the text 'Step X: '")
    final_score: float = Field(ge=0, le=1, description="Final score for the task, between 0.0 and 1.0")
    rationale: str = Field(description="Compendious user-facing explanation for the given score")

class DetailedVideoDescription(BaseModel):
    applications_used: List[str] = Field(description="List of applications used in the video for completing the task")
    completion_sequence_steps: List[str] = Field(description="Highly detailed step-by-step breakdown of the sequence of steps taken to complete the task")
    user_feedback: str = Field(description="Feedback for the user to improve their task completion skills in the future")
    description: str = Field(description="High-level summary description of the video content")

class CompletionScore(BaseModel):
    rationale: str = Field(description="Concise description of how well the user completed the task")
    completion_score: float = Field(ge=0, le=1, description="Final completion score, between 0.0 and 1.0")

class VideoScore(BaseModel):
    # task and video scores
    # task_score: float
    task_uniqueness_score: Optional[float]
    video_completion_score: float
    description_uniqueness_score: Optional[float]
    video_uniqueness_score: float
    boosted_multiplier: Optional[float]
    final_score: float

    # metadata
    task_overview: str
    # task_score_breakdown: TaskScoreBreakdown
    completion_score_breakdown: CompletionScore
    detailed_video_description: DetailedVideoDescription

class FocusVideoEmbeddings(BaseModel):
    # embeddings
    task_overview_embedding: Optional[List[float]]
    detailed_video_description_embedding: Optional[List[float]]
    video_embedding: List[float]

class BoostedTaskIndex(BaseModel):
    index: int
    
class BoostedTaskData(BaseModel):
    title: str
    description: str
    multiplier: float

def get_s3_path(video_id: str) -> str:
    return f"clips/{video_id}.webm"

def get_gcs_uri(video_id: str) -> str:
    return f"gs://{GOOGLE_CLOUD_BUCKET_NAME}/{get_s3_path(video_id)}"

class FocusScoringService:
    def __init__(self):
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        self.model_name = "gemini-1.5-pro-001"
        print(f"Using model: {self.model_name}")
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.temperature = 1.3
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.task_overview_index = Pinecone(api_key=PINECONE_API_KEY).Index("focus-task-overview-index")
        self.video_description_index = Pinecone(api_key=PINECONE_API_KEY).Index("focus-video-description-index")
        self.completion_video_index = Pinecone(api_key=PINECONE_API_KEY).Index("focus-completion-video-index")
        # [gemini task score, task uniqueness score, completion score, description uniqueness score, video uniqueness score]
        self.coefficients = [0.23, 0.16, 0.29, 0.14, 0.18]

    # Gemini API call related functions

    async def make_gemini_request_with_retries(self, system_prompt: str, user_prompt: str, video_id: str, OutputClassSchema: BaseModel) -> str:
        num_retries = 3
        for retry_idx in range(num_retries):
            try:
                start = time.time()
                output = await self.make_gemini_request(system_prompt, user_prompt, video_id, OutputClassSchema)
                print(f"Got gemini output in {time.time() - start} seconds for {OutputClassSchema.__name__}")
                return output
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from Gemini response for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})")
                await asyncio.sleep(1)
            except ValidationError as e:
                print(f"Error turning parsed JSON into Pydantic object for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Error making Gemini request for {OutputClassSchema.__name__}, trying again: {e} ({retry_idx + 1}/{num_retries})")
                await asyncio.sleep(6)
        raise Exception(f"Failed to turn Gemini response into JSON and then into Pydantic object for {OutputClassSchema.__name__} after {num_retries} attempts")

    async def make_gemini_request(self, system_prompt: str, user_prompt: str, video_id: str, OutputClassSchema: BaseModel) -> GenerativeModel:
        model = GenerativeModel(
            self.model_name,
            system_instruction=system_prompt.strip(),
            safety_settings=self.safety_settings,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=OutputClassSchema.model_json_schema(),
            ),
        )

        parts = []
        if video_id:
            parts.append(Part.from_uri(get_gcs_uri(video_id), mime_type="video/webm"))
        parts.append(user_prompt.strip())

        response = await model.generate_content_async(parts)
        return OutputClassSchema(**json.loads(response.text))

    async def get_task_score_from_gemini(self, task_overview: str) -> TaskScoreBreakdown:
        return await self.make_gemini_request_with_retries(
            system_prompt=focus_scoring_prompts.TASK_SCORE_SYSTEM_PROMPT,
            user_prompt=focus_scoring_prompts.TASK_SCORE_USER_PROMPT.format(task_overview=task_overview),
            video_id=None,
            OutputClassSchema=TaskScoreBreakdown,
        )

    async def get_detailed_video_description(self, video_id: str, task_overview: str) -> DetailedVideoDescription:
        return await self.make_gemini_request_with_retries(
            system_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_SYSTEM_PROMPT,
            user_prompt=focus_scoring_prompts.DETAILED_DESCRIPTION_USER_PROMPT.format(task_overview=task_overview),
            video_id=video_id,
            OutputClassSchema=DetailedVideoDescription,
        )

    async def get_completion_score_breakdown(self, video_id: str, task_overview: str, detailed_video_description: Optional[DetailedVideoDescription] = None) -> CompletionScore:
        detailed_video_description_string = f"""\n\n
Additionally, here is a detailed description of the video content:

<detailed_video_description>
{detailed_video_description.model_dump_json(indent=2)}
</detailed_video_description>
""" if detailed_video_description else ""

        return await self.make_gemini_request_with_retries(
            system_prompt=focus_scoring_prompts.TASK_COMPLETION_SYSTEM_PROMPT,
            user_prompt=focus_scoring_prompts.TASK_COMPLETION_USER_PROMPT.format(
                task_overview=task_overview,
                detailed_video_description_string=detailed_video_description_string,
            ),
            video_id=video_id,
            OutputClassSchema=CompletionScore,
        )

    # Pinecone related functions

    async def get_task_uniqueness_score(self, task_overview_embedding: List[float]) -> float:
        return await query_pinecone(self.task_overview_index, task_overview_embedding)

    async def get_description_uniqueness_score(self, detailed_video_description_embedding: List[float]) -> float:
        return await query_pinecone(self.video_description_index, detailed_video_description_embedding)

    async def get_video_uniqueness_score(self, video_embedding: List[float]) -> float:
        return await query_pinecone(self.completion_video_index, video_embedding)

    # Embedding related functions

    def get_video_duration_seconds(self, video_id: str) -> int:
        with get_db_context() as db:
            video_metadata = get_video_metadata(db, video_id)

            if video_metadata is None:
                raise ValueError(f"Focus video not found: {video_id}")

            video_duration_seconds = video_metadata.video_details.get("duration")
            if video_duration_seconds is None:
                print(f"Video duration not found for video: {video_id}")
                video_duration_seconds = 120

            return video_duration_seconds

    async def get_video_embedding(self, video_id: str, video_duration_seconds: int) -> List[float]:
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
                    interval_sec=end_offset_sec - start_offset_sec
                )
            )
            return embeddings.video_embeddings[0].embedding
        return await run_with_retries(_internal_async)

    async def get_text_embedding(self, text: str) -> Optional[List[float]]:
        async def _internal_async():
            response = await asyncio.wait_for(self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            ), timeout=10)
            return response.data[0].embedding

        try:
            return await run_with_retries(_internal_async)
        except Exception as e:
            print(f"Error getting text embedding: {e}")
            return None

    async def embed_and_get_task_uniqueness_score(self, task_overview: str) -> Tuple[Optional[List[float]], Optional[float]]:
        embedding = await self.get_text_embedding(task_overview)
        if embedding is None:
            return None, None
        return embedding, await self.get_task_uniqueness_score(embedding)

    async def embed_and_get_video_uniqueness_score(self, video_id: str, video_duration_seconds: int):
        embedding = await self.get_video_embedding(video_id, video_duration_seconds)
        return embedding, await self.get_video_uniqueness_score(embedding)

    async def get_detailed_video_description_embedding_score(self, video_id, task_overview):
        detailed_video_description = await self.get_detailed_video_description(video_id, task_overview)
        embedding = await self.get_text_embedding(detailed_video_description.model_dump_json())
        if embedding is None:
            return detailed_video_description, None, None
        return detailed_video_description, embedding, await self.get_description_uniqueness_score(embedding)

    async def score_video(self, video_id: str, focusing_task: str, focusing_description: str):
        """
        The video score is a score of how well the user completed the task, based on the task overview and the detailed video description.
        If the video is too similar to other videos, it will be rejected.
        Errors raised should make the video rejected.
        """
        boosted_multiplier = 1.0
        with get_db_context() as db:
            """
            if the task is boosted, use the boosted task info directly
            """
            video_metadata = get_video_metadata(db, video_id)
            if video_metadata and video_metadata.task_id:
                task = db.query(TaskRecordPG).filter(
                    TaskRecordPG.id == video_metadata.task_id,
                ).first()
                if task:
                    boosted_task = db.query(BoostedTask).filter(
                        BoostedTask.id == task.boosted_id
                    ).first()
                    if boosted_task:
                        boosted_multiplier = boosted_task.multiplier
                        focusing_task = boosted_task.title
                        focusing_description = boosted_task.description
                        # print(f"Scoring boosted task index {boosted_task.id} multiplier {boosted_multiplier}\n\n{boosted_task.title}\n\n{boosted_task.description}")
                        
            
        video_duration_seconds = self.get_video_duration_seconds(video_id)

        if video_duration_seconds < TWO_MINUTES:
            raise ValueError(f"Video duration is too short: {video_duration_seconds} seconds")

        if video_duration_seconds > NINETY_MINUTES:
            raise ValueError(f"Video duration is too long: {video_duration_seconds} seconds")

        task_overview = f"# {focusing_task}\n\n{focusing_description}"

        (
            (task_overview_embedding, task_uniqueness_score),
            # task_score_breakdown,
            (video_description, video_description_embedding, video_description_uniqueness_score),
            (video_embedding, video_uniqueness_score),
        ) = await asyncio.gather(
            self.embed_and_get_task_uniqueness_score(task_overview),  # uses openai to get embedding
            # self.get_task_score_from_gemini(task_overview),  # uses gemini to score task
            self.get_detailed_video_description_embedding_score(video_id, task_overview),  # uses gemini to get detailed description
            self.embed_and_get_video_uniqueness_score(video_id, video_duration_seconds),
        )
        
        if video_uniqueness_score < MIN_VIDEO_UNIQUENESS_SCORE:
            raise VideoUniquenessError("Video uniqueness score is too low.")
        
        completion_score_breakdown = await self.get_completion_score_breakdown(
            video_id,
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

    # async def get_model_cached_on_video(self, video_id: str) -> GenerativeModel:
    #     video_part = Part.from_uri(get_gcs_uri(video_id), mime_type="video/webm")
    #     cached_content = caching.CachedContent.create(
    #         model_name=self.model_name,
    #         system_instruction="You are an expert video description generator. You are given a video and a task and you need to generate a detailed description of the video.",
    #         contents=[video_part],
    #         ttl=datetime.timedelta(minutes=5),
    #     )
    #     return GenerativeModel.from_cached_content(cached_content=cached_content)


def main():
    service = FocusScoringService()
    import asyncio
    import time

    async def main():
        video_id = "29f91a6f-1393-4765-ba00-263b4cff28b6"
        task_overview = """
# Multimodal tokenization research

Read the Show-O peper to understand how they have trained a unified diffusion and autoregressive model for multimodal tokenization.
""".strip()

        score_details = await service.score_video(video_id, task_overview, "description")
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
