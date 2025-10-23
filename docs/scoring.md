# Focus Video Scoring System Documentation

## Overview

The Focus Video Scoring System evaluates task completion recordings submitted through the OMEGA Focus application. It uses a multi-phase approach combining AI models (Gemini, OpenAI, DeepSeek) to analyze videos, generate detailed annotations, perform legitimacy checks, and calculate completion scores.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Scoring Workflow](#scoring-workflow)
3. [Scoring Phases](#scoring-phases)
4. [Video State Machine](#video-state-machine)
5. [API Endpoints](#api-endpoints)
6. [Scoring Components](#scoring-components)
7. [Error Handling](#error-handling)
8. [Code References](#code-references)

---

## System Architecture

### Key Components

- **API Layer** (`validator_api/app.py`): FastAPI endpoints for video submission and scoring requests
- **Scoring Service** (`validator_api/validator_api/scoring/scoring_service.py`): Core scoring logic and orchestration
- **Legitimacy Checks** (`validator_api/validator_api/scoring/legitimacy_checks.py`): Anti-fraud detection
- **Database Models** (`validator_api/validator_api/database/models/`): Data models for videos, scores, and embeddings

### External Services

- **Google Vertex AI (Gemini 2.5 Flash)**: Video annotation and detailed description generation
- **OpenAI**: Text embeddings for uniqueness checks
- **DeepSeek (via Chutes API)**: Task completion scoring
- **Pinecone**: Vector similarity search for uniqueness detection

---

## Scoring Workflow

### High-Level Flow

```
1. User completes task in OMEGA Focus app
2. Video uploaded → State: PROCESSING
3. /api/focus/get_focus_score endpoint triggered
4. run_focus_scoring() executed in background
5. Scoring phases executed (see below)
6. Video state updated based on score:
   - Marketplace videos → PENDING_HUMAN_REVIEW
   - Score < 0.1 → REJECTED
   - Score >= 0.1 → READY
7. User can submit READY videos to marketplace
```

### Detailed Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Video Submission                              │
│                         ↓                                        │
│              State: IN_PROGRESS/PROCESSING                       │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│         API: POST /api/focus/get_focus_score                    │
│         Parameters:                                              │
│         - video_id: str                                          │
│         - focusing_task: str (task title)                        │
│         - focusing_description: str (task description)           │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│            Background Task: run_focus_scoring()                 │
│                                                                  │
│  Step 1: Fetch video record from database                       │
│         - Get video metadata and task_type                       │
│         - Check if video exists and not deleted                  │
│                                                                  │
│  Step 2: Score video using FocusScoringService                  │
│         - Run all scoring phases (see below)                     │
│         - Returns: score_details, embeddings                     │
│                                                                  │
│  Step 3: Generate AI task feedback                              │
│         - Create actionable feedback for user improvement        │
│                                                                  │
│  Step 4: Update database based on results                       │
│         - If marketplace video: PENDING_HUMAN_REVIEW             │
│         - If score < 0.1: REJECTED                               │
│         - If score >= 0.1: READY                                 │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Final State                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐     │
│  │PENDING_HUMAN   │  │    REJECTED     │  │     READY     │     │
│  │    REVIEW      │  │                 │  │               │     │
│  │(marketplace)   │  │  (score < 0.1)  │  │(score >= 0.1) │     │
│  └────────────────┘  └────────────────┘  └───────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scoring Phases

The scoring system operates in three distinct phases as documented in `scoring_service.py:1-16`:

### Phase 0: Video Annotation Generation

**Purpose**: Generate a detailed, step-by-step annotation of the video that could allow replication by a Computer Use Agent.

**Process**:
1. Gemini 2.5 Flash analyzes the full video
2. Generates `DetailedVideoDescription` containing:
   - `applications_used`: List of software/tools used
   - `completion_sequence_steps`: Detailed step-by-step breakdown
   - `user_feedback`: Constructive feedback for improvement
   - `description`: High-level summary

**Implementation**: `get_detailed_video_description()` at `scoring_service.py:242-276`

**Prompts**:
- System: `DETAILED_DESCRIPTION_SYSTEM_PROMPT` at `focus_scoring_prompts.py:53-69`
- User: `DETAILED_DESCRIPTION_USER_PROMPT` at `focus_scoring_prompts.py:71-85`

### Phase 1: Legitimacy Checks (Spam Detection & Rejection)

**Purpose**: Detect and reject fraudulent or low-quality submissions before expensive scoring operations.

**Order**: Least to greatest computational cost

#### Check 1: Video Length Validation
- **Min Duration**: 120 seconds (2 minutes)
- **Max Duration**: 5400 seconds (90 minutes)
- **Raises**: `VideoTooShortError` or `VideoTooLongError`
- **Code**: `scoring_service.py:547-556`

#### Check 2: Video Uniqueness Detection
- **Method**: Video embedding similarity via Pinecone
- **Process**:
  1. Generate video embedding using Google's `multimodalembedding` model
  2. Random 120-second segment extracted
  3. Query Pinecone `focus-completion-video-index`
  4. Calculate uniqueness score: `1.0 - similarity_score`
- **Threshold**: `MIN_VIDEO_UNIQUENESS_SCORE = 0.02`
- **Raises**: `VideoUniquenessError` if below threshold
- **Code**: `scoring_service.py:439-440`, `scoring_service.py:580-582`

#### Check 3: Chat-Only Detection
- **Purpose**: Detect users who just talk about completing tasks instead of actually completing them
- **Method**: DeepSeek LLM analysis of video annotation
- **Check Class**: `ChatOnlyCheck` at `legitimacy_checks.py:43-137`
- **Key Verification Points**:
  - Visual evidence matching task requirements
  - Presence of relevant tools and interfaces
  - Active interaction with necessary applications
  - Actual task outputs visible
  - Detection of exploited patterns (watching YouTube, reading PDFs without notes, etc.)
- **Output**: `(legitimate: bool, rationale: str)`
- **Code**: `scoring_service.py:584-614`

#### Check 4: YouTube/Movie Detection (Not Currently Active)
- **Status**: Commented out in code
- **Purpose**: Detect screen recordings of watching videos
- **Method**: Would use Gemini on first and last video chunks

#### Check 5: Automation Detection (Not Currently Active)
- **Status**: Not reliably working yet
- **Purpose**: Detect automated/scripted task completions

**Exploited Task Cases** (defined in `focus_scoring_prompts.py:263-270`):
- Users listening to music
- Watching YouTube videos/movies/TV shows
- Writing code without testing it
- Reading PDFs without taking notes
- Reading a book without engagement

### Phase 2: Completion Scoring

**Purpose**: Evaluate how well the user completed the assigned task.

**Process**:
1. Task overview formatted: `# {focusing_task}\n\n{focusing_description}`
2. Three parallel operations:
   - **Task Embedding & Uniqueness**: OpenAI `text-embedding-3-large` → Pinecone `focus-task-overview-index`
   - **Video Description Embedding & Uniqueness**: Annotation → OpenAI embedding → Pinecone `focus-video-description-index`
   - **Video Embedding & Uniqueness**: Google multimodal embedding → Pinecone `focus-completion-video-index`
3. Completion score generated via DeepSeek LLM
4. Boosted multiplier applied (if applicable)
5. Final score calculated: `completion_score * boosted_multiplier`

**Completion Score Calculation**:
- **Model**: DeepSeek (via `query_llm()`)
- **Input**: Task overview + detailed video description
- **Output**: `CompletionScore` with:
  - `rationale`: Explanation of score
  - `completion_score`: Float [0.0, 1.0]
- **Scoring Rubric**:
  - 0.0-0.2: Poor task completion, largely irrelevant or counterproductive
  - 0.2-0.4: Weak task completion, minimal completion towards goal
  - 0.4-0.6: Moderate task completion, somewhat helpful but not ideal
  - 0.6-0.8: Good task completion, diligently completed
  - 0.8-1.0: Excellent task completion, high quality and efficiency

**Code**: `_get_completion_score_breakdown()` at `scoring_service.py:288-337`

**Prompts**:
- System: `DESC_ONLY_TASK_COMPLETION_SYSTEM_PROMPT` at `focus_scoring_prompts.py:208-239`
- User: `DESC_ONLY_TASK_COMPLETION_USER_PROMPT` at `focus_scoring_prompts.py:241-260`

---

## Video State Machine

### Internal States (FocusVideoStateInternal)

Defined in `focus_video_record.py:30-42`:

```
IN_PROGRESS → PROCESSING → {PENDING_HUMAN_REVIEW, READY, REJECTED}
                                    ↓
                                SUBMITTED → PURCHASE_PENDING → PURCHASED
```

| State | Description |
|-------|-------------|
| `IN_PROGRESS` | User is actively recording task |
| `PROCESSING` | Video uploaded, scoring in progress |
| `PENDING_HUMAN_REVIEW` | Marketplace video awaiting human approval |
| `READY` | Scored and eligible for submission to marketplace |
| `REJECTED` | Failed scoring or legitimacy checks, lifecycle ended |
| `SUBMITTED` | Listed on marketplace for miner purchase |
| `PURCHASE_PENDING` | Miner requested purchase, awaiting payment |
| `PURCHASED` | Miner payment confirmed |

### External States (User-Facing)

Mapping defined in `focus_video_record.py:44-64`:

| Internal State | External State |
|----------------|----------------|
| `IN_PROGRESS` | `PROCESSING` |
| `PROCESSING` | `PROCESSING` |
| `PENDING_HUMAN_REVIEW` | `PENDING_HUMAN_REVIEW` |
| `READY` | `READY` |
| `REJECTED` | `REJECTED` |
| `SUBMITTED` | `SUBMITTED` |
| `PURCHASE_PENDING` | `SUBMITTED` |
| `PURCHASED` | `REWARDED` |

### Task Types

Defined in `focus_video_record.py:15-18`:

- **USER**: Regular user-created tasks
- **BOOSTED**: Tasks with reward multipliers
- **MARKETPLACE**: Platform-created tasks requiring human review

---

## API Endpoints

### POST /api/focus/get_focus_score

**Purpose**: Initiate video scoring process

**Authentication**: Focus API Key (in header: `FOCUS_API_KEY`)

**Request Body**:
```json
{
  "video_id": "string",
  "focusing_task": "string",
  "focusing_description": "string"
}
```

**Response**:
```json
{
  "success": true
}
```

**Process**:
1. Validates API key
2. Adds `run_focus_scoring()` to background tasks
3. Returns immediately (non-blocking)
4. Scoring happens asynchronously

**Code**: `app.py:777-802`

### GET /api/focus/get_list

**Purpose**: Get list of available videos for purchase

**Rate Limit**: 5 requests/minute

**Response**: List of available focus videos

**Code**: `app.py:804-810`

### POST /api/focus/purchase

**Purpose**: Purchase a focus video

**Authentication**: Bittensor hotkey signature

**Rate Limit**: 2 requests/minute

**Code**: `app.py:812-860`

---

## Scoring Components

### Data Models

#### VideoScore (`scoring.py:66-80`)
```python
VideoScore(
    task_uniqueness_score: Optional[float],
    video_completion_score: float,
    description_uniqueness_score: Optional[float],
    video_uniqueness_score: float,
    boosted_multiplier: Optional[float],
    final_score: float,
    task_overview: str,
    completion_score_breakdown: CompletionScore,
    detailed_video_description: DetailedVideoDescription
)
```

#### FocusVideoEmbeddings (`scoring.py:83-87`)
```python
FocusVideoEmbeddings(
    task_overview_embedding: Optional[List[float]],
    detailed_video_description_embedding: Optional[List[float]],
    video_embedding: Optional[List[float]]
)
```

#### DetailedVideoDescription (`scoring.py:33-45`)
```python
DetailedVideoDescription(
    applications_used: List[str],
    completion_sequence_steps: List[str],
    user_feedback: str,
    description: str
)
```

#### CompletionScore (`scoring.py:48-54`)
```python
CompletionScore(
    rationale: str,
    completion_score: float  # [0.0, 1.0]
)
```

### FocusScoringService Class

**Location**: `scoring_service.py:411-641`

**Initialization**:
- Vertex AI client
- Three Pinecone indexes:
  - `focus-task-overview-index`
  - `focus-video-description-index`
  - `focus-completion-video-index`
- OpenAI client for embeddings
- Legitimacy checks list

**Key Methods**:

#### `score_video(video_id, focusing_task, focusing_description, bypass_checks=False)`
Main scoring orchestration method.

**Parameters**:
- `video_id`: Video identifier
- `focusing_task`: Task title
- `focusing_description`: Detailed task description
- `bypass_checks`: Skip duration and legitimacy checks (for marketplace videos)

**Returns**: `Tuple[VideoScore, FocusVideoEmbeddings]`

**Process**:
1. Check for boosted task multiplier
2. Validate video duration (if not bypassed)
3. Run parallel operations:
   - Generate task embedding and uniqueness score
   - Generate detailed video description, embedding, and uniqueness score
   - Generate video embedding and uniqueness score
4. Run legitimacy checks (if not bypassed)
5. Calculate completion score
6. Apply boosted multiplier
7. Return final scores and embeddings

**Code**: `scoring_service.py:506-641`

#### `get_text_embedding(text)`
Generate OpenAI embeddings for text.

**Model**: `text-embedding-3-large`

**Code**: `scoring_service.py:443-469`

#### `get_video_uniqueness_score(video_embedding)`
Query Pinecone for video similarity.

**Returns**: Uniqueness score (1.0 - similarity)

**Code**: `scoring_service.py:439-440`

---

## Error Handling

### Custom Exceptions

Defined in `scoring.py:5-18`:

| Exception | Raised When | Rejection Message |
|-----------|-------------|-------------------|
| `VideoTooShortError` | Duration < 120s | "Video is too short. Please ensure the video is at least 10 seconds long." |
| `VideoTooLongError` | Duration > 5400s | "Video is too long. Please ensure the video is less than 10 minutes long." |
| `VideoUniquenessError` | Uniqueness < 0.02 | "Task recording is not unique. If you believe this is an error, please contact a team member." |
| `LegitimacyCheckError` | Fraud detected | "An anomaly was detected in the video. If you believe this is an error, please contact a team member via the OMEGA Focus Discord channel." |

### Error Flow in run_focus_scoring()

**Code**: `app.py:382-413`

```python
try:
    # Main scoring logic
except Exception as e:
    # Determine rejection reason based on error type
    if isinstance(e, VideoTooShortError):
        rejection_reason = "Video is too short..."
    elif isinstance(e, VideoTooLongError):
        rejection_reason = "Video is too long..."
    elif isinstance(e, VideoUniquenessError):
        rejection_reason = "Task recording is not unique..."
    elif isinstance(e, LegitimacyCheckError):
        rejection_reason = "An anomaly was detected..."
    else:
        rejection_reason = "Error scoring video"

    # Mark video as rejected in database
    await mark_video_rejected(db, video_id, rejection_reason, ...)
```

### Score-Based Rejection

**Threshold**: `MIN_FINAL_SCORE = 0.1` (defined at `app.py:338`)

**Logic** (`app.py:357-373`):
- If `final_score < 0.1`:
  - If `final_score == 0.0` (legitimacy failure):
    - Use actual failure reason from legitimacy check
  - Else (low completion score):
    - Provide feedback including AI rationale
  - Mark as `REJECTED`

---

## Code References

### Key Files and Line Numbers

| Component | File | Lines |
|-----------|------|-------|
| Main API endpoint | `validator_api/app.py` | 777-802 |
| Background scoring function | `validator_api/app.py` | 295-413 |
| Scoring service class | `validator_api/validator_api/scoring/scoring_service.py` | 411-641 |
| Main scoring method | `validator_api/validator_api/scoring/scoring_service.py` | 506-641 |
| Legitimacy checks | `validator_api/validator_api/scoring/legitimacy_checks.py` | 1-137 |
| Chat-only detection | `validator_api/validator_api/scoring/legitimacy_checks.py` | 43-137 |
| Scoring prompts | `validator_api/validator_api/scoring/focus_scoring_prompts.py` | 1-270 |
| Data models | `validator_api/validator_api/database/models/scoring.py` | 1-98 |
| Video record model | `validator_api/validator_api/database/models/focus_video_record.py` | 1-133 |
| State machine | `validator_api/validator_api/database/models/focus_video_record.py` | 30-64 |

### Configuration Constants

Located in `scoring_service.py:68-72`:

```python
TWO_MINUTES = 120  # Minimum video duration
NINETY_MINUTES = 5400  # Maximum video duration
FOCUS_VIDEO_MIN_SCORE = 0.05  # Not currently enforced
FOCUS_VIDEO_MAX_SCORE = 1.0
MIN_VIDEO_UNIQUENESS_SCORE = 0.02  # Uniqueness threshold
```

---

## Summary

The Focus Video Scoring System is a sophisticated multi-phase pipeline that:

1. **Generates detailed annotations** of task completion videos using Gemini 2.5 Flash
2. **Performs legitimacy checks** to detect fraud and low-quality submissions
3. **Calculates completion scores** using DeepSeek LLM with detailed rubrics
4. **Manages video lifecycle** through a state machine from submission to marketplace purchase
5. **Provides actionable feedback** to users for improvement

The system is designed to be cost-efficient (checks ordered by expense), fraud-resistant (multiple legitimacy checks), and scalable (async operations, background tasks, parallel processing).

---

## Future Enhancements

As noted in the codebase comments, potential future improvements include:

1. Re-enabling YouTube/movie video-watching detection
2. Re-enabling exploit/screen recording detection
3. Implementing reliable automation detection
4. Adjusting `FOCUS_VIDEO_MIN_SCORE` threshold if needed
5. Exploring more cost-efficient scoring methods (reasoning models vs. full video analysis)
