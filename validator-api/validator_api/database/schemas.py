from datetime import datetime
import enum
from typing import List, Optional
from pydantic import BaseModel, Field

class TaskStatusEnum(enum.Enum):
    Ready = 'Ready'
    Running = 'Running'
    Stopped = 'Stopped'
    Completed = 'Completed'

class FocusVideoEnum(enum.Enum):
    Uploaded = 'Uploaded'
    Available = 'Available'
    Pending = 'Pending'
    Purchased = 'Purchased'
    Submitted = 'Submitted'
    Consumed = 'Consumed'

class TaskSchema(BaseModel):
    focusing_task: str = Field(...)
    duration: float | None = None
    description: str | None = None
    checked: bool | None = None
    date: str | None = None
    clip_link: str | None = None
    status: str | None = None
    score: float | None = None
    event: dict | None = None

class UserSchema(BaseModel):
    email: str = Field(...)
    password: str = Field(...)
    nick_name: str = Field(...)

class UserLoginSchema(BaseModel):
    email: str = Field(...)
    password: str = Field(...)
    
class IpfsUrlSchema(BaseModel):
    url: str = Field(...)
    miner_hotkey: str = Field(...)

class TimeSlot(BaseModel):
    start: str
    end: str

class FocusTask(BaseModel):
    id: str
    name: str
    priority: str
    timeSlot: TimeSlot
    description: str
    steps: List[str]
    resources: List[str]
    challenges: List[str]
    focusTips: List[str]
    isCompleted: bool
    totalDuration: str
    category: Optional[str] = None

class Metadata(BaseModel):
    date: str
    day: str
    lastUpdated: datetime

class DailySchedule(BaseModel):
    metadata: Metadata
    tasks: List[FocusTask]
    tools: List[str]


class Link(BaseModel):
    url: str = Field(..., description="URL of the website")
    name: str = Field(..., description="Name of the website")

class Step(BaseModel):
    title: str = Field(..., description="Title of the step")
    content: List[str] = Field(..., description="Content of the step in paragraphs")
    links: Optional[List[Link]] = Field(None, description="Relevant links for the step")

class KeyPoint(BaseModel):
    title: str = Field(..., description="Title of the key point")
    details: List[str] = Field(..., description="Details of the key point")
    links: Optional[List[Link]] = Field(None, description="Relevant links for the key point")

class Analysis(BaseModel):
    summary: str = Field(..., description="Summary of the analysis")
    points: List[str] = Field(..., description="Key points or recommendations")
    links: Optional[List[Link]] = Field(None, description="Relevant links for the analysis")

class TextAnalysisReport(BaseModel):
    title: str = Field(..., description="Title of the report")
    introduction: str = Field(..., description="Introduction or overview of the report")
    steps: List[Step] = Field(..., description="Main steps of the report")
    keypoints: List[KeyPoint] = Field(..., description="Key points or findings")
    analysis: Analysis = Field(..., description="Overall analysis or conclusion")
    metadata: List[str] = Field(..., description="Additional metadata about the report")
    timestamp: str = Field(..., description="Timestamp of the report generation (ISO 8601 date string YYYY-MM-DDTHH:MM:SS-UTC)")
    links: Optional[List[Link]] = Field(None, description="General links for the entire report")

class FocusTask(BaseModel):
    id: str
    name: str
    priority: str
    timeSlot: TimeSlot
    description: str
    steps: List[str]
    resources: List[str]
    challenges: List[str]
    focusTips: List[str]
    isCompleted: bool
    totalDuration: str
    category: Optional[str] = None
    