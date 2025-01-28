from abc import ABC, abstractmethod
from typing import Tuple
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os
import json
from validator_api.database import get_db_context
from validator_api.database.models.focus_video_record import FocusVideoRecord
from validator_api.database.models.scoring import DetailedVideoDescription

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

def get_task_overview(video_id: str) -> str:
    with get_db_context() as db:
        video_record = db.query(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None)
        ).first()

        if video_record is None:
            raise ValueError(f"Video not found: {video_id}")

        focusing_task = video_record.video_details.get("focusing_task", "")
        focusing_description = video_record.video_details.get(
            "focusing_description", "")

    task_overview = f"# Task Title: {focusing_task}\n\n Task Description:\n{focusing_description}"
    return task_overview

async def get_detailed_video_description(video_id: str) -> DetailedVideoDescription:
    with get_db_context() as db:
        video_record = db.query(FocusVideoRecord).filter(
            FocusVideoRecord.video_id == video_id,
            FocusVideoRecord.deleted_at.is_(None)
        ).first()
        
        if video_record is None:
            raise ValueError(f"Video not found: {video_id}")
        
        if video_record.video_details and "detailed_video_description" in video_record.video_details:
            return DetailedVideoDescription.model_validate(
                video_record.video_details["detailed_video_description"]
            )
    
    raise ValueError(f"Detailed video description not found for video: {video_id}")

class LegitimacyCheck(ABC):
    @abstractmethod
    async def passes_check(self, video_id: str) -> Tuple[bool, str]:
        """
        Check if the video passes this legitimacy check.
        
        Args:
            video_id: The ID of the video to check
            
        Returns:
            Tuple[bool, str]: (passed, failure_reason)
            - passed: True if check passed, False if failed
            - failure_reason: Description of why check failed (empty if passed)
        """
        pass

class ChatOnlyDetectionModel(BaseModel):
    rationale: str = Field(description="Detailed rationale for the score")
    legitimate: bool = Field(description="False if the user is cheating by talking about completing a task, but not actually completing it, True otherwise")

class ChatOnlyCheck(LegitimacyCheck):
    """
    Fails if a user is talking about completing a task (e.g. in a notepad or AI chat), but not actually completing it.
    """
    async def passes_check(self, video_id: str) -> Tuple[bool, str]:
        chat_only_check_prompt = """You are an expert in analyzing task performance videos.
Your current task is to determine if the user is cheating by talking about completing a task, but not actually completing it.
Verify that the video shows actual evidence of task completion, not just chat interactions claiming completion.

Key verification points:
- Visual evidence matching the task requirements (e.g., code execution, file manipulation, system interactions)
- Presence of relevant tools and interfaces required for the task
- Active interaction with necessary applications or systems
- Actual task outputs visible in the recording

Red flags for chat-only submissions:
- Video shows only chat interface interactions
- Claims of completion without supporting visual evidence
- Missing technical elements required by the task
- Absence of expected task artifacts or outputs
- Timeline inconsistency between chat claims and visible work"""
        
        completion_sequence_steps = await get_detailed_video_description(video_id)
        
        messages = [
            {"role": "system", "content": chat_only_check_prompt},
            {"role": "user", "content": f"Please analyze the following annotated transcript and determine if the user is cheating by talking about completing a task, but not actually completing it: {completion_sequence_steps}"}
        ]

        try:
            response = await openai_client.beta.chat.completions.parse(
                model="o1-2024-12-17",
                messages=messages,
                response_format=ChatOnlyDetectionModel,
            )
            if not response.choices[0].message.content:
                raise Exception("Empty response from API")

            chat_only_detection_data = json.loads(response.choices[0].message.content)
            model = ChatOnlyDetectionModel(
                rationale=chat_only_detection_data["rationale"],
                legitimate=chat_only_detection_data["legitimate"]
            )
            
            print(f"[{video_id}] ChatOnlyCheck result: {model}")
            return model.legitimate, model.rationale

        except Exception as e:
            print(f"[{video_id}] ❌ Error during chat-only check: {str(e)}")
            return True, f"Error during chat-only check (allowing to pass): {str(e)}"
