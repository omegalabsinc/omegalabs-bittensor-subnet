from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class VideoTooShortError(Exception):
    pass


class VideoTooLongError(Exception):
    pass


class VideoUniquenessError(Exception):
    pass


class LegitimacyCheckError(Exception):
    pass


class TaskScoreBreakdown(BaseModel):
    reasoning_steps: List[str] = Field(
        description="Steps of reasoning used to arrive at the final score. Before each step, write the text 'Step X: '"
    )
    final_score: float = Field(
        ge=0, le=1, description="Final score for the task, between 0.0 and 1.0"
    )
    rationale: str = Field(
        description="Compendious user-facing explanation for the given score"
    )


class ActionDetails(BaseModel):
    type: Literal["click", "type", "scroll", "navigate", "press", "drag", "select", "switch"] = Field(
        description="Type of action performed (click, type, scroll, navigate, press, drag, select, switch)"
    )
    coordinates: Optional[List[int]] = Field(
        default=None,
        description="Absolute pixel coordinates [x, y] for click/drag actions"
    )
    button: Optional[Literal["left", "right", "middle"]] = Field(
        default=None,
        description="Mouse button used for click actions"
    )
    target: str = Field(
        description="UI element description (e.g., 'Submit button', 'Cell E2', 'Search bar')"
    )
    text: Optional[str] = Field(
        default=None,
        description="Text that was typed (for type actions)"
    )


class StepMetadata(BaseModel):
    window_title: Optional[str] = Field(
        default=None,
        description="Title of the active window"
    )
    process_name: Optional[str] = Field(
        default=None,
        description="Name of the active application process"
    )
    absolute_timestamp: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp from trajectories data"
    )


class CompletionStep(BaseModel):
    step: int = Field(
        description="Sequential step number starting from 0"
    )
    timestamp: float = Field(
        description="Seconds from video start (e.g., 0.0, 1.5, 10.2)"
    )
    action: ActionDetails = Field(
        description="Details of the action performed"
    )
    observation: str = Field(
        description="What the user saw or what happened as a result of this action"
    )
    metadata: StepMetadata = Field(
        description="Additional context from trajectories data and window information"
    )


class DetailedVideoDescription(BaseModel):
    applications_used: List[str] = Field(
        description="List of applications used in the video for completing the task"
    )
    completion_sequence_steps: List[str] = Field(
        description="Highly detailed step-by-step breakdown of the sequence of steps taken to complete the task"
    )
    user_feedback: str = Field(
        description="Feedback for the user to improve their task completion skills in the future"
    )
    description: str = Field(
        description="High-level summary description of the video content"
    )

    @field_validator('completion_sequence_steps', mode='before')
    @classmethod
    def convert_old_structured_to_strings(cls, v):
        """
        Convert old structured CompletionStep format (dicts) to simple strings.
        This handles migration of cached data from previous schema versions.
        """
        if not isinstance(v, list):
            return v

        converted_steps = []
        for step in v:
            if isinstance(step, dict):
                # Old structured format - extract the observation or target as the string
                observation = step.get('observation', '')
                target = step.get('action', {}).get('target', '') if isinstance(step.get('action'), dict) else ''
                step_str = observation or target or str(step)
                converted_steps.append(step_str)
            elif isinstance(step, str):
                # Already a string
                converted_steps.append(step)
            else:
                # Unknown format, convert to string
                converted_steps.append(str(step))

        return converted_steps


class StructuredVideoDescription(BaseModel):
    """Video description with structured completion steps integrated with trajectories data"""
    applications_used: List[str] = Field(
        description="List of applications used in the video for completing the task"
    )
    completion_sequence_steps: List[CompletionStep] = Field(
        description="Structured step-by-step breakdown with trajectories data integrated"
    )
    user_feedback: str = Field(
        description="Feedback for the user to improve their task completion skills in the future"
    )
    description: str = Field(
        description="High-level summary description of the video content"
    )


class CompletionScore(BaseModel):
    rationale: str = Field(
        description="Concise description of how well the user completed the task"
    )
    completion_score: float = Field(
        ge=0, le=1, description="Final completion score, between 0.0 and 1.0"
    )


class CompletionScoreWithoutRange(BaseModel):
    rationale: str = Field(
        description="Concise description of how well the user completed the task"
    )
    completion_score: float = Field(
        description="Final completion score, between 0.0 and 1.0"
    )


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
    video_embedding: Optional[List[float]]


class BoostedTaskIndex(BaseModel):
    index: int


class BoostedTaskData(BaseModel):
    title: str
    description: str
    multiplier: float
