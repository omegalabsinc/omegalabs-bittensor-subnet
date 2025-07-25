from abc import ABC, abstractmethod
from typing import Tuple, Optional
from pydantic import BaseModel, Field
from sqlalchemy import select
from validator_api.validator_api.database import get_db_context
from validator_api.validator_api.database.models.focus_video_record import (
    FocusVideoRecord,
)
from validator_api.validator_api.scoring import focus_scoring_prompts
from validator_api.validator_api.scoring.scoring_service import DetailedVideoDescription
from validator_api.validator_api.scoring.deepseek_chat import query_llm


class LegitimacyCheck(ABC):
    @abstractmethod
    async def passes_check(
        self,
        video_id: str,
        detailed_video_description: Optional[DetailedVideoDescription] = None,
    ) -> Tuple[bool, str]:
        """
        Check if the video passes this legitimacy check.

        Args:
            video_id: The ID of the video to check
            detailed_video_description: Optional pre-computed video description

        Returns:
            Tuple[bool, str]: (passed, failure_reason)
            - passed: True if check passed, False if failed
            - failure_reason: Description of why check failed (empty if passed)
        """
        pass


class ChatOnlyDetectionModel(BaseModel):
    rationale: str = Field(description="Detailed rationale for the score")
    legitimate: bool = Field(
        description="False if the user is cheating by talking about completing a task, but not actually completing it, True otherwise"
    )


class ChatOnlyCheck(LegitimacyCheck):
    """
    Fails if a user is talking about completing a task (e.g. in a notepad or AI chat), but not actually completing it.
    """

    async def passes_check(
        self,
        video_id: str,
        detailed_video_description: Optional[DetailedVideoDescription] = None,
    ) -> Tuple[bool, str]:
        chat_only_check_prompt = """You are an expert in analyzing task performance videos.
Your current task is to determine if the user is cheating by talking about completing a task, but not actually completing it.
Verify that the video shows actual evidence of task completion, not just chat interactions claiming completion.


Key verification points:
- Visual evidence matching the task requirements (e.g., code execution, file manipulation, system interactions)
- Presence of relevant tools and interfaces required for the task
- Active interaction with necessary applications or systems
- Actual task outputs visible in the recording
- user opens pdfs and not taking notes, stays on same page for more than 5 mins, etc.
- practice coding and not testing it.
- watches youtube videos /movies/ tv shows and not taking notes or not doing anything else.
- Check for omega focus app, if it is not open, then it is not legitimate.
- Check the exploited examples very carefully, there are users who just do that and not actually completing the task.
- If the task belongs to the below list of exploited examples where user just tries to complete the task but does not actually complete it,
    give legitimacy false even if it is similar to the steps done in the annotated transcript:
{EXPLOITED_TASK_CASES}

Red flags for chat-only submissions:
- Video shows only chat interface interactions
- Claims of completion without supporting visual evidence
- Missing technical elements required by the task
- Absence of expected task artifacts or outputs
- Timeline inconsistency between chat claims and visible work

Limit your critique to the existence of chat-only submissions; the full video scoring and rating will be done in another step.

OUTPUT JSON FORMAT:
{{
    "rationale": "Detailed explanation of the analysis",
    "legitimate": true/false; False if the user is cheating by talking about completing a task, but not actually completing it, True otherwise
}}
"""
        # important: above, we need to provide an example of the output JSON format

        # Use provided description if available, otherwise fetch from DB
        if detailed_video_description is None:
            async with get_db_context() as db:
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
                    detailed_video_description = (
                        DetailedVideoDescription.model_validate(
                            video_record.video_details["detailed_video_description"]
                        )
                    )
                else:
                    raise ValueError(
                        f"Detailed video description not found for video: {video_id}"
                    )

        messages = [
            {"role": "system", "content": chat_only_check_prompt.format(EXPLOITED_TASK_CASES=focus_scoring_prompts.EXPLOITED_TASK_CASES)},
            {
                "role": "user",
                "content": f"Please analyze the following annotated transcript and determine if the user is cheating by talking about completing a task, but not actually completing it: {detailed_video_description}",
            },
        ]

        try:
            chat_only_detection_data = await query_llm(messages, ChatOnlyDetectionModel)

            print(f"[{video_id}] ChatOnlyCheck result: {chat_only_detection_data}")

            return (
                chat_only_detection_data.legitimate,
                chat_only_detection_data.rationale,
            )

        except Exception as e:
            print(f"[{video_id}] ❌ Error during chat-only check: {str(e)}")
            return True, f"Error during chat-only check (allowing to pass): {str(e)}"
