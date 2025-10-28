# Focus Video Scoring and Embeddings: Complete Technical Analysis

## Overview

Focus video scoring uses **three separate Pinecone indexes** for uniqueness detection and **DeepSeek LLM** for completion scoring. The system generates embeddings from multiple sources (task, video, description) and checks uniqueness against historical submissions.

---

## 1. Three-Dimensional Uniqueness Scoring

**File**: `validator_api/validator_api/scoring/scoring_service.py:428-441`

The system maintains **3 separate Pinecone indexes**:

```python
class FocusScoringService:
    def __init__(self):
        # 3 Pinecone indexes for different uniqueness checks
        self.task_overview_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-task-overview-index"          # Task uniqueness
        )
        self.video_description_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-video-description-index"      # Description uniqueness
        )
        self.completion_video_index = Pinecone(api_key=PINECONE_API_KEY).Index(
            "focus-completion-video-index"       # Video visual uniqueness
        )
```

**Purpose of each index**:
1. **task_overview_index**: Prevents users from submitting the same task repeatedly
2. **video_description_index**: Ensures video content/actions are unique
3. **completion_video_index**: Detects visually similar videos

---

## 2. Complete Scoring Flow

**File**: `validator_api/validator_api/scoring/scoring_service.py:523-659`

### Phase 1: Validation Checks (Lines 563-576)

```python
# Duration constraints
if video_duration_seconds < TWO_MINUTES (120 sec):
    raise VideoTooShortError

if video_duration_seconds > NINETY_MINUTES (5400 sec):
    raise VideoTooLongError
```

### Phase 2: Parallel Embedding Generation (Lines 579-596)

Three operations run in parallel using `asyncio.gather`:

```python
(
    (task_overview_embedding, task_uniqueness_score),
    (video_description, video_description_embedding, video_description_uniqueness_score),
    (video_embedding, video_uniqueness_score),
) = await asyncio.gather(
    # 1. Task Embedding + Uniqueness
    self.embed_and_get_task_uniqueness_score(task_overview),

    # 2. Detailed Video Description + Embedding + Uniqueness
    self.get_detailed_video_description_embedding_score(video_id, task_overview),

    # 3. Video Embedding + Uniqueness
    self.embed_and_get_video_uniqueness_score(video_id, video_duration_seconds),
)
```

**Breakdown of each operation**:

#### 2a. Task Embedding + Uniqueness (Lines 488-494)

```python
async def embed_and_get_task_uniqueness_score(self, task_overview: str):
    # Step 1: Generate text embedding using OpenAI
    embedding = await self.get_text_embedding(task_overview)  # OpenAI text-embedding-3-large

    # Step 2: Query Pinecone for similar tasks
    return embedding, await self._get_task_uniqueness_score(embedding)
```

**Text Embedding Details** (Lines 460-486):
- **Model**: OpenAI `text-embedding-3-large`
- **Input**: Task title + description formatted as markdown
- **Output**: Embedding vector (dimension: ~3072 for text-embedding-3-large)

**Task Uniqueness Query** (Lines 444-447):
```python
async def _get_task_uniqueness_score(self, task_overview_embedding: List[float]):
    return await query_pinecone(self.task_overview_index, task_overview_embedding)
```

#### 2b. Video Description Embedding + Uniqueness (Lines 506-521)

```python
async def get_detailed_video_description_embedding_score(self, video_id, task_overview):
    # Step 1: Generate detailed video description using Gemini 2.5 Flash
    detailed_video_description = await get_detailed_video_description(
        video_id, task_overview
    )  # Uses Gemini to watch video and annotate

    # Step 2: Embed the description JSON
    embedding = await self.get_text_embedding(
        detailed_video_description.model_dump_json()  # Convert Pydantic model to JSON
    )

    # Step 3: Query Pinecone for similar descriptions
    return (
        detailed_video_description,
        embedding,
        await self.get_description_uniqueness_score(embedding),
    )
```

**Detailed Video Description Process** (`video_description.py:35-80`):
- **Step 1**: Check if description cached in database
- **Step 2**: If not cached, call Gemini 2.5 Flash to watch video and generate:
  - `applications_used`: List of applications/tools used
  - `completion_sequence_steps`: Step-by-step actions taken
- **Step 3**: Cache result in `FocusVideoRecord.video_details`
- **Step 4**: Embed the full JSON description using OpenAI

**Description Uniqueness Query** (Lines 449-454):
```python
async def get_description_uniqueness_score(self, detailed_video_description_embedding):
    return await query_pinecone(
        self.video_description_index, detailed_video_description_embedding
    )
```

#### 2c. Video Visual Embedding + Uniqueness (Lines 496-504)

```python
async def embed_and_get_video_uniqueness_score(self, video_id, video_duration_seconds):
    try:
        # Step 1: Generate video embedding using Google Multimodal Embedding
        embedding = await get_video_embedding(video_id, video_duration_seconds)

        # Step 2: Query Pinecone for visually similar videos
        return embedding, await self.get_video_uniqueness_score(embedding)
    except Exception as e:
        print(f"Failed to create video embedding for {video_id}: {str(e)}")
        return None, 0.1  # Assume unique if embedding fails
```

**Video Embedding Process** (Lines 356-387):
```python
async def get_video_embedding(video_id: str, video_duration_seconds: int):
    """
    Uses Google's multimodalembedding model to generate video embeddings.
    Takes a random 120-second segment from videos longer than 2 minutes.
    """
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")

    # Random 120-second window
    start_offset_sec = random.randint(0, max(0, video_duration_seconds - 120))
    end_offset_sec = min(video_duration_seconds, start_offset_sec + 120)

    embeddings = await model.get_embeddings(
        video=Video.load_from_file(get_gcs_uri(video_id)),  # Load from GCS bucket
        video_segment_config=VideoSegmentConfig(
            start_offset_sec=start_offset_sec,
            end_offset_sec=end_offset_sec,
            interval_sec=end_offset_sec - start_offset_sec,
        ),
    )
    return embeddings.video_embeddings[0].embedding
```

**Video Uniqueness Query** (Lines 456-457):
```python
async def get_video_uniqueness_score(self, video_embedding: List[float]):
    return await query_pinecone(self.completion_video_index, video_embedding)
```

---

## 3. Pinecone Query Logic (Uniqueness Calculation)

**File**: `validator_api/validator_api/scoring/scoring_service.py:390-425`

```python
async def query_pinecone(pinecone_index: Pinecone, vector: List[float]) -> float:
    """
    Queries Pinecone to find the most similar existing vector.
    Returns uniqueness score = 1 - similarity_score
    """
    response = await pinecone_index.query(
        vector=vector,
        top_k=1,  # Only get single most similar match
    )

    if len(response["matches"]) > 0:
        similarity_score = response["matches"][0]["score"]  # Cosine similarity
    else:
        similarity_score = 0  # No matches = completely unique

    # Clamp to [0, 1] range
    similarity_score = max(0.0, min(similarity_score, 1.0))

    # Return uniqueness = inverse of similarity
    return 1.0 - similarity_score
```

**Uniqueness Score Interpretation**:
- **1.0**: Completely unique (no similar submissions exist)
- **0.5**: Moderately similar to existing submission
- **0.02**: Very similar (at rejection threshold)
- **0.0**: Identical to existing submission

---

## 4. Video Uniqueness Threshold Check

**File**: `validator_api/validator_api/scoring/scoring_service.py:75, 599-600`

```python
MIN_VIDEO_UNIQUENESS_SCORE = 0.02  # 2% uniqueness required

if video_uniqueness_score < MIN_VIDEO_UNIQUENESS_SCORE:
    raise VideoUniquenessError("Video uniqueness score is too low.")
```

**Critical Note**: Only **video visual uniqueness** is checked against threshold. Task and description uniqueness are computed but **not enforced** (stored for analytics/future use).

---

## 5. Legitimacy Checks

**File**: `validator_api/validator_api/scoring/scoring_service.py:602-632`

```python
if self.legitimacy_checks:
    check_results = await asyncio.gather(
        *(check.passes_check(video_id, video_description)
          for check in self.legitimacy_checks)
    )

    for passed, failure_reason in check_results:
        if not passed:
            return VideoScore(
                final_score=0.0,  # REJECTED
                completion_score_breakdown=CompletionScore(
                    rationale=failure_reason,
                    completion_score=0.0,
                ),
                ...
            )
```

**Active Legitimacy Checks** (Line 441):
- **ChatOnlyCheck**: Detects if video shows only chat/messaging (no actual work)

**Inactive Checks** (from comments at top of file):
- YouTube/movie watching detection
- Screen recording exploit detection
- Automation detection

---

## 6. Task Completion Scoring

**File**: `validator_api/validator_api/scoring/scoring_service.py:305-353`

```python
async def _get_completion_score_breakdown(
    task_overview: str,
    detailed_video_description: DetailedVideoDescription,
) -> CompletionScore:
    """
    Uses DeepSeek model via Chutes API to evaluate task completion quality.
    """
    messages = [
        {"role": "system", "content": DESC_ONLY_TASK_COMPLETION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DESC_ONLY_TASK_COMPLETION_USER_PROMPT.format(
                task_overview=task_overview,
                applications_used=detailed_video_description.applications_used,
                completion_sequence_steps=detailed_video_description.completion_sequence_steps,
            ),
        },
    ]

    completion_score_without_range = await query_llm(
        messages=messages,
        output_model=CompletionScoreWithoutRange,  # Uses DeepSeek
    )

    return CompletionScore(
        rationale=completion_score_without_range.rationale,
        completion_score=max(0.0, min(1.0, completion_score_without_range.completion_score)),
    )
```

**Completion Score Components**:
- **Input**: Task overview + Gemini-generated video description (applications used, steps taken)
- **Model**: DeepSeek via Chutes API (cost-optimized reasoning model)
- **Output**: Score 0.0-1.0 + rationale explaining the score
- **Evaluation**: How well the video demonstrates task completion

---

## 7. Final Score Calculation

**File**: `validator_api/validator_api/scoring/scoring_service.py:639-659`

```python
completion_gemini_score = completion_score_breakdown.completion_score
final_score = completion_gemini_score * boosted_multiplier

return VideoScore(
    task_uniqueness_score=task_uniqueness_score,              # Not used in final_score
    video_completion_score=completion_gemini_score,            # Primary score component
    description_uniqueness_score=video_description_uniqueness_score,  # Not used
    video_uniqueness_score=video_uniqueness_score,             # Threshold check only
    boosted_multiplier=boosted_multiplier,                     # Optional 2x-5x multiplier
    final_score=final_score,                                   # completion * boost
    task_overview=task_overview,
    completion_score_breakdown=completion_score_breakdown,
    detailed_video_description=video_description,
)
```

**Final Score Formula**:
```
final_score = completion_score × boosted_multiplier

Where:
- completion_score: 0.0-1.0 from DeepSeek LLM evaluation
- boosted_multiplier: 1.0 (default) or 2x-5x for marketplace videos
```

**Important**: Uniqueness scores are **not factored into final_score**. They are only used for:
1. Rejection if `video_uniqueness_score < 0.02`
2. Analytics/tracking (stored but not weighted)

---

## 8. Complete Data Flow Diagram

```
USER SUBMITS FOCUS VIDEO
─────────────────────────────────────────────────────────────

1. Upload video to GCS bucket
2. Trigger /api/focus/get_focus_score endpoint
3. run_focus_scoring() background task starts


SCORING PHASE 1: VALIDATION
─────────────────────────────────────────────────────────────

4. Check video duration
   ├─> If < 2 minutes: REJECT (VideoTooShortError)
   ├─> If > 90 minutes: REJECT (VideoTooLongError)
   └─> If 2-90 minutes: Continue


SCORING PHASE 2: EMBEDDINGS (Parallel Execution)
─────────────────────────────────────────────────────────────

5a. Task Embedding & Uniqueness
    └─> task_overview = "# {title}\n\n{description}"
    └─> OpenAI text-embedding-3-large → task_embedding
    └─> Pinecone query focus-task-overview-index
    └─> task_uniqueness_score = 1 - similarity

5b. Video Description & Uniqueness
    └─> Gemini 2.5 Flash watches video → DetailedVideoDescription
        ├─> applications_used: ["VS Code", "Chrome", ...]
        └─> completion_sequence_steps: [step1, step2, ...]
    └─> OpenAI embeds description JSON → description_embedding
    └─> Pinecone query focus-video-description-index
    └─> description_uniqueness_score = 1 - similarity

5c. Video Visual Embedding & Uniqueness
    └─> Select random 120-second segment
    └─> Google MultiModalEmbedding → video_embedding
    └─> Pinecone query focus-completion-video-index
    └─> video_uniqueness_score = 1 - similarity


SCORING PHASE 3: UNIQUENESS THRESHOLD
─────────────────────────────────────────────────────────────

6. Check video_uniqueness_score
   └─> If < 0.02: REJECT (VideoUniquenessError)
   └─> If >= 0.02: Continue


SCORING PHASE 4: LEGITIMACY CHECKS
─────────────────────────────────────────────────────────────

7. Run ChatOnlyCheck
   └─> If video shows only chat: REJECT (final_score = 0.0)
   └─> If passes: Continue


SCORING PHASE 5: COMPLETION EVALUATION
─────────────────────────────────────────────────────────────

8. LLM Evaluation (DeepSeek)
   └─> Input: task_overview + video_description
   └─> DeepSeek evaluates task completion quality
   └─> Returns: completion_score (0.0-1.0) + rationale


SCORING PHASE 6: FINAL SCORE CALCULATION
─────────────────────────────────────────────────────────────

9. Calculate final score
   └─> Check if boosted task (marketplace videos)
       ├─> If boosted: boosted_multiplier = 2x-5x
       └─> If not: boosted_multiplier = 1.0
   └─> final_score = completion_score × boosted_multiplier


RESULT
─────────────────────────────────────────────────────────────

10. Return VideoScore
    ├─> task_uniqueness_score: (analytics only)
    ├─> video_completion_score: completion_score
    ├─> description_uniqueness_score: (analytics only)
    ├─> video_uniqueness_score: (threshold check only)
    ├─> boosted_multiplier: 1.0 or 2x-5x
    └─> final_score: completion × boost

11. Update database
    └─> State: PENDING_HUMAN_REVIEW (marketplace) or APPROVED
```

---

## 9. YouTube Video Embeddings (Comparison)

For context, the OMEGA Labs Bittensor Subnet also uses embeddings for YouTube videos submitted by miners. Here's how it differs from Focus videos:

### YouTube Video Embedding Architecture

**File**: `omega/imagebind_wrapper.py:105-264`

- **Model**: Meta's ImageBind (with optional VideoBind v0.2)
- **Modalities**: 3 separate embeddings generated
  - Video frames embedding
  - Audio track embedding
  - Text description embedding
- **Dimension**: 1024-dimensional vectors

### YouTube Novelty Detection

**File**: `validator_api/validator_api/score.py:32-97`

**Two-Phase Process**:
1. **Local Novelty**: Compare videos within current batch using cosine similarity
2. **Global Novelty**: Query Pinecone for similarity with historical videos

```python
# Phase 1: Local novelty (within batch)
local_novelty_scores = compute_novelty_score_among_batch(embeddings)

# Phase 2: Global novelty (Pinecone query)
global_novelty_scores = await asyncio.gather(
    *[
        query_pinecone(video=embedding.tolist())
        for embedding, local_score in zip(embeddings.video, local_novelty_scores)
        if local_score >= DIFFERENCE_THRESHOLD
    ]
)

# Final novelty = min(local, global)
true_novelty_scores = [
    min(local_score, global_score)
    for local_score, global_score in zip(local_novelty_scores, global_novelty_scores)
]
```

### YouTube vs Focus Video Comparison

| Aspect | YouTube Videos (Miners) | Focus Videos (Users) |
|--------|------------------------|----------------------|
| **Embedding Model** | ImageBind (1024D) | Google MultiModalEmbedding + OpenAI text-embedding-3-large |
| **Pinecone Indexes** | 2 (video + audio) | 3 (task + description + video) |
| **Uniqueness Scope** | Batch + global | Global only |
| **Threshold** | DIFFERENCE_THRESHOLD (~0.1) | MIN_VIDEO_UNIQUENESS_SCORE (0.02) |
| **Scoring Model** | Relevance + Novelty + Detail | DeepSeek completion evaluation |
| **Upload Target** | Hugging Face dataset | GCS bucket + marketplace |

---

## 10. Key Files and Line Numbers

| Component | File | Lines |
|-----------|------|-------|
| **FocusScoringService Init** | `validator_api/validator_api/scoring/scoring_service.py` | 428-441 |
| **Main score_video Function** | `validator_api/validator_api/scoring/scoring_service.py` | 523-659 |
| **Task Embedding + Uniqueness** | `validator_api/validator_api/scoring/scoring_service.py` | 488-494 |
| **Description Embedding + Uniqueness** | `validator_api/validator_api/scoring/scoring_service.py` | 506-521 |
| **Video Embedding + Uniqueness** | `validator_api/validator_api/scoring/scoring_service.py` | 496-504 |
| **Video Embedding Generation** | `validator_api/validator_api/scoring/scoring_service.py` | 356-387 |
| **Pinecone Query Function** | `validator_api/validator_api/scoring/scoring_service.py` | 390-425 |
| **Task Uniqueness Score** | `validator_api/validator_api/scoring/scoring_service.py` | 444-447 |
| **Description Uniqueness Score** | `validator_api/validator_api/scoring/scoring_service.py` | 449-454 |
| **Video Uniqueness Score** | `validator_api/validator_api/scoring/scoring_service.py` | 456-457 |
| **OpenAI Text Embedding** | `validator_api/validator_api/scoring/scoring_service.py` | 460-486 |
| **Completion Score Breakdown** | `validator_api/validator_api/scoring/scoring_service.py` | 305-353 |
| **Detailed Video Description** | `validator_api/validator_api/scoring/video_description.py` | 35-80 |
| **Uniqueness Threshold Check** | `validator_api/validator_api/scoring/scoring_service.py` | 599-600 |
| **Legitimacy Checks** | `validator_api/validator_api/scoring/scoring_service.py` | 602-632 |
| **VideoScore Model** | `validator_api/validator_api/database/models/scoring.py` | 66-81 |
| **YouTube ImageBind Wrapper** | `omega/imagebind_wrapper.py` | 105-264 |
| **YouTube Pinecone Upload** | `validator_api/validator_api/score.py` | 100-144 |
| **YouTube Novelty Detection** | `validator_api/validator_api/score.py` | 55-97 |

---

## 11. Summary: How `embed_and_get_task_uniqueness_score` is Used

### Function Definition

```python
# Line 488-494
async def embed_and_get_task_uniqueness_score(
    self, task_overview: str
) -> Tuple[Optional[List[float]], Optional[float]]:
    embedding = await self.get_text_embedding(task_overview)
    if embedding is None:
        return None, None
    return embedding, await self._get_task_uniqueness_score(embedding)
```

### What it does:
1. **Embeds** the task text using OpenAI `text-embedding-3-large`
2. **Queries** Pinecone `focus-task-overview-index` for similar tasks
3. **Returns** uniqueness score = `1 - similarity_with_closest_match`

### How it's used in final scoring:
- **Stored**: `VideoScore.task_uniqueness_score` for analytics
- **NOT used**: Not factored into `final_score` calculation
- **NOT enforced**: No rejection threshold (unlike video uniqueness)

### Purpose
Track task diversity and prevent spam, but currently not a gating factor for acceptance.

The **only uniqueness score that affects rejection** is `video_uniqueness_score` (must be ≥ 0.02).

---

## 12. Configuration Constants

**File**: `validator_api/validator_api/scoring/scoring_service.py:71-75`

```python
TWO_MINUTES = 120  # Minimum video duration in seconds
NINETY_MINUTES = 5400  # Maximum video duration in seconds
FOCUS_VIDEO_MIN_SCORE = 0.05  # Minimum acceptable completion score
FOCUS_VIDEO_MAX_SCORE = 1.0  # Maximum completion score
MIN_VIDEO_UNIQUENESS_SCORE = 0.02  # Minimum video uniqueness (2%)
```

---

## 13. API Models

### VideoScore Model

**File**: `validator_api/validator_api/database/models/scoring.py:66-81`

```python
class VideoScore(BaseModel):
    # Scores
    task_uniqueness_score: Optional[float]           # Analytics only
    video_completion_score: float                    # Used in final_score
    description_uniqueness_score: Optional[float]    # Analytics only
    video_uniqueness_score: float                    # Threshold check only
    boosted_multiplier: Optional[float]              # Marketplace boost
    final_score: float                               # completion × boost

    # Metadata
    task_overview: str
    completion_score_breakdown: CompletionScore
    detailed_video_description: DetailedVideoDescription
```

### FocusVideoEmbeddings Model

```python
class FocusVideoEmbeddings(BaseModel):
    task_overview_embedding: Optional[List[float]]              # ~3072D
    detailed_video_description_embedding: Optional[List[float]] # ~3072D
    video_embedding: Optional[List[float]]                      # Varies
```

---

## 14. Future Enhancements

Based on comments in the code, potential future improvements include:

1. **Additional Legitimacy Checks**:
   - YouTube/movie watching detection
   - Screen recording exploit detection
   - Automation detection

2. **Uniqueness Score Weighting**:
   - Currently only `video_uniqueness_score` affects rejection
   - Could incorporate `task_uniqueness_score` and `description_uniqueness_score` into final scoring

3. **Dynamic Threshold Adjustment**:
   - Adjust `MIN_VIDEO_UNIQUENESS_SCORE` based on dataset size
   - Implement adaptive thresholds for different task types

4. **Multi-Segment Video Embedding**:
   - Currently uses random 120-second segment
   - Could embed multiple segments and average for better coverage

---

## Appendix: Related Documentation

- **Focus Video Scoring System**: `docs/scoring.md` (comprehensive system documentation)
- **CLAUDE.md Project Guide**: Root-level project overview
- **Validator API Config**: `validator_api/validator_api/config.py` (environment variables)
