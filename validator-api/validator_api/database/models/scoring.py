from typing import List, Optional
from pydantic import BaseModel, Field

class VideoTooShortError(Exception):
    pass

class VideoTooLongError(Exception):
    pass

class VideoUniquenessError(Exception):
    pass

class LegitimacyCheckError(Exception):
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

class CompletionScoreWithoutRange(BaseModel):
    rationale: str = Field(description="Concise description of how well the user completed the task")
    completion_score: float = Field(description="Final completion score, between 0.0 and 1.0")

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
