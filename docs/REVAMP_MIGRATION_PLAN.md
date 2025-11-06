# SN24 REVAMP: MIGRATION TO COMPUTER USE AGENT TRAINING DATA

## Executive Summary

This document outlines the migration plan for transitioning OMEGA Labs Bittensor Subnet (SN24) from multimodal YouTube/Focus video collection to **computer use agent training data** collection, similar to OpenCUA's AgentNet dataset.

**Goal**: Create the world's largest computer use agent training dataset by incentivizing miners and Focus users to submit high-quality screen recordings with action trajectories, reasoning, and task completion data.

**Timeline**: 4-phase migration over 8-12 weeks

---

## 1. UNDERSTANDING AGENTNET DATASET STRUCTURE

Based on analysis of `xlangai/AgentNet` dataset, here's what we need to replicate:

### 1.1 AgentNet Schema (Per Task)

```python
{
    # Task Definition
    "task_id": str,                          # UUID identifier
    "instruction": str,                      # Natural language task description
    "natural_language_task": str,            # Original task phrasing
    "actual_task": str,                      # Executed task variant
    
    # Trajectory Data
    "traj": [                                # List of action steps
        {
            "index": int,                    # Step sequence number
            "image": str,                    # Screenshot filename (e.g., "uuid.png")
            "value": {
                # Action Information
                "code": str,                 # PyAutoGUI command (e.g., "pyautogui.click(x=0.018, y=0.508)")
                "action": str,               # Human-readable action description
                "thought": str,              # Agent's reasoning before action
                "reflection": str,           # Analysis after action
                
                # Observation
                "observation": str,          # Detailed screen state description
                
                # Validation
                "last_step_correct": bool,   # Was this step correct?
                "last_step_redundant": bool, # Was this step necessary?
            }
        },
        # ... more steps
    ],
    
    # Task Metadata
    "domain": str,                           # Application category (e.g., "infeasible", "productivity")
    "task_completed": bool,                  # Success indicator
    "task_difficulty": int,                  # Complexity rating (1-10)
    
    # Quality Scores
    "alignment_score": int,                  # How well actions align with task (0-10)
    "efficiency_score": int,                 # How efficiently task was completed (0-10)
    "reason": str,                           # Completion notes or failure explanation
}
```

### 1.2 Images Storage

AgentNet stores screenshots separately in compressed ZIP archives:
- `ubuntu_images/images.zip` (split into .z01, .z02, ... files)
- `win_mac_images/images.zip` (split similarly)

Each screenshot referenced by trajectory step via `"image": "uuid.png"`.

---

## 2. CURRENT SN24 ARCHITECTURE SUMMARY

### 2.1 Current Data Flow

```
YOUTUBE PATH (65% rewards):
Miner → YouTube search → Download video → ImageBind embeddings → Submit VideoMetadata
Validator → Spot-check embeddings → Score (relevance + novelty + detail) → Upload to HF

FOCUS VIDEO PATH (30% rewards):
User (Focus App) → Record screen → Upload to GCS → Validator API scoring
→ Multi-phase evaluation (embeddings + LLM) → Marketplace listing
→ Miner purchase → Score boost
```

### 2.2 Key Components to Preserve

**Keep (Core Infrastructure)**:
- Bittensor protocol (weights, emissions, metagraph)
- Validator/Miner base classes
- Database models (with modifications)
- Validator API (FastAPI)
- HuggingFace upload pipeline
- Pinecone for novelty/uniqueness checking

**Modify (Adapt for Agent Data)**:
- Protocol definitions (`omega/protocol.py`)
- Scoring logic (from video quality → trajectory quality)
- Database schema (add trajectory tables)
- Miner submission format

**Remove (YouTube-specific)**:
- ImageBind wrapper (replace with action extraction)
- YouTube scraping (`video_utils.py`)
- YouTube-specific scoring logic
- Audio diarization (not needed for agent data)

---

## 3. NEW ARCHITECTURE: COMPUTER USE AGENT DATA

### 3.1 Target Data Schema

```python
class AgentTrajectory(BaseModel):
    """
    Represents a complete computer use agent trajectory.
    Matches AgentNet format with SN24-specific additions.
    """
    
    # Identifiers
    trajectory_id: str                       # UUID
    video_id: Optional[str]                  # Reference to original video (if available)
    user_id: str                             # Submitter (miner or focus user)
    
    # Task Definition
    task_id: str                             # Reference to Task table
    instruction: str                         # Natural language task
    natural_language_task: str               # Original phrasing
    actual_task: str                         # What was actually done
    
    # Trajectory Steps
    traj: List[TrajectoryStep]               # Action sequence
    
    # Metadata
    domain: str                              # Application category
    os: str                                  # "ubuntu", "macos", "windows"
    task_difficulty: int                     # 1-10
    
    # Completion Status
    task_completed: bool                     # Did agent finish?
    alignment_score: int                     # 0-10
    efficiency_score: int                    # 0-10
    reason: str                              # Explanation
    
    # SN24-Specific
    submitted_at: int                        # Timestamp
    miner_hotkey: str                        # Submitter identity
    validator_uid: int                       # Validator that scored

class TrajectoryStep(BaseModel):
    """Single step in an agent trajectory."""
    
    index: int                               # Step number
    image: str                               # Screenshot filename/ID
    
    # Action
    code: str                                # PyAutoGUI-style command
    action: str                              # Human-readable description
    
    # Reasoning
    thought: str                             # Why agent chose this action
    reflection: str                          # Post-action analysis
    
    # Observation
    observation: str                         # Screen state description
    
    # Validation
    last_step_correct: bool                  # Was this step correct?
    last_step_redundant: bool                # Was this step necessary?
```

### 3.2 New Data Flow

```
MINER/USER SUBMISSION:
User/Miner → Record screen during task → Extract actions from video
→ Generate trajectory with reasoning → Submit to Validator API

VALIDATOR PROCESSING:
Validator receives AgentTrajectory → Multi-phase validation:
  1. Format validation (schema compliance)
  2. Action validity (plausible actions for given UI state)
  3. Trajectory coherence (logical flow)
  4. Novelty check (Pinecone similarity)
  5. Task completion evaluation (LLM scoring)
  6. Final score calculation
→ Store in database → Upload to HuggingFace

REWARDS:
Validator sets weights based on trajectory scores
→ Miners/users receive TAO emissions
→ High-quality trajectories earn more
```

---

## 4. MIGRATION PHASES

### PHASE 1: Infrastructure Setup (Weeks 1-2)

#### 1.1 New Protocol Definitions

**File**: `omega/protocol.py` (add new classes)

```python
class TrajectoryStep(BaseModel):
    index: int
    image: str
    code: str
    action: str
    thought: str
    reflection: str
    observation: str
    last_step_correct: bool
    last_step_redundant: bool

class AgentTrajectoryMetadata(BaseModel):
    trajectory_id: str
    task_id: str
    instruction: str
    natural_language_task: str
    actual_task: str
    traj: List[TrajectoryStep]
    domain: str
    os: str
    task_difficulty: int
    task_completed: bool
    alignment_score: int
    efficiency_score: int
    reason: str

class AgentTrajectories(bt.Synapse):
    """Synapse for submitting agent trajectories."""
    
    query: str  # Task category/type
    num_trajectories: int = 4
    trajectory_metadata: Optional[List[AgentTrajectoryMetadata]] = None
```

#### 1.2 Database Schema Updates

**File**: `validator_api/validator_api/database/models/agent_trajectory.py` (new)

```python
class AgentTrajectory(Base):
    __tablename__ = "agent_trajectories"
    
    id = Column(String(36), primary_key=True)
    trajectory_id = Column(String(36), unique=True, nullable=False)
    video_id = Column(String(255), ForeignKey("focus_videos.video_id"), nullable=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=False)
    
    # Task info
    task_id = Column(String(36), ForeignKey("tasks.id"))
    instruction = Column(Text, nullable=False)
    natural_language_task = Column(Text)
    actual_task = Column(Text)
    
    # Trajectory data (JSON serialized)
    traj = Column(JSON, nullable=False)
    
    # Metadata
    domain = Column(String(100))
    os = Column(String(20))
    task_difficulty = Column(Integer)
    
    # Completion
    task_completed = Column(Boolean, default=False)
    alignment_score = Column(Integer)
    efficiency_score = Column(Integer)
    reason = Column(Text)
    
    # SN24 tracking
    submitted_at = Column(DateTime, default=datetime.now)
    miner_hotkey = Column(String(255))
    validator_uid = Column(Integer)
    
    # Scoring
    final_score = Column(Float)
    action_validity_score = Column(Float)
    coherence_score = Column(Float)
    completion_score = Column(Float)
    novelty_score = Column(Float)
    
    # State
    approved = Column(Boolean, default=False)
    state = Column(Enum(TrajectoryState))  # SUBMITTED, VALIDATING, APPROVED, REJECTED

class TrajectoryImage(Base):
    """Stores screenshots associated with trajectories."""
    
    __tablename__ = "trajectory_images"
    
    id = Column(String(36), primary_key=True)
    trajectory_id = Column(String(36), ForeignKey("agent_trajectories.trajectory_id"))
    step_index = Column(Integer)
    image_id = Column(String(255), unique=True)  # UUID.png
    
    # Storage
    gcs_uri = Column(String(500))  # gs://bucket/images/uuid.png
    
    # Metadata
    width = Column(Integer)
    height = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.now)
```

**Migration SQL**:
```sql
-- Run this to update database
CREATE TABLE agent_trajectories (
    id VARCHAR(36) PRIMARY KEY,
    trajectory_id VARCHAR(36) UNIQUE NOT NULL,
    video_id VARCHAR(255),
    user_id VARCHAR(255) NOT NULL,
    task_id VARCHAR(36),
    instruction TEXT NOT NULL,
    natural_language_task TEXT,
    actual_task TEXT,
    traj JSON NOT NULL,
    domain VARCHAR(100),
    os VARCHAR(20),
    task_difficulty INTEGER,
    task_completed BOOLEAN DEFAULT FALSE,
    alignment_score INTEGER,
    efficiency_score INTEGER,
    reason TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    miner_hotkey VARCHAR(255),
    validator_uid INTEGER,
    final_score FLOAT,
    action_validity_score FLOAT,
    coherence_score FLOAT,
    completion_score FLOAT,
    novelty_score FLOAT,
    approved BOOLEAN DEFAULT FALSE,
    state VARCHAR(20),
    FOREIGN KEY (video_id) REFERENCES focus_videos(video_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE trajectory_images (
    id VARCHAR(36) PRIMARY KEY,
    trajectory_id VARCHAR(36) NOT NULL,
    step_index INTEGER NOT NULL,
    image_id VARCHAR(255) UNIQUE NOT NULL,
    gcs_uri VARCHAR(500),
    width INTEGER,
    height INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trajectory_id) REFERENCES agent_trajectories(trajectory_id)
);

CREATE INDEX idx_trajectory_images_trajectory ON trajectory_images(trajectory_id);
CREATE INDEX idx_agent_trajectories_user ON agent_trajectories(user_id);
CREATE INDEX idx_agent_trajectories_score ON agent_trajectories(final_score DESC);
```

#### 1.3 API Endpoints

**File**: `validator_api/app.py` (add new endpoints)

```python
@app.post("/api/agent/submit_trajectory")
async def submit_trajectory(request: AgentTrajectorySubmission):
    """
    Accept trajectory submission from miners or Focus users.
    
    Request:
    {
        "trajectory": AgentTrajectoryMetadata,
        "images": [{"step_index": 0, "image_base64": "..."}],
        "submitter_hotkey": "5F...",
    }
    
    Response:
    {
        "trajectory_id": "uuid",
        "status": "received",
        "estimated_score": 0.85
    }
    """
    # 1. Validate schema
    # 2. Upload images to GCS
    # 3. Store in database (state=SUBMITTED)
    # 4. Trigger async scoring
    # 5. Return trajectory_id

@app.post("/api/agent/get_trajectory_score")
async def get_trajectory_score(trajectory_id: str):
    """Get current scoring status and result."""
    # Return VideoScore-equivalent for trajectories

@app.get("/api/agent/get_list")
async def get_agent_trajectory_list(
    domain: Optional[str] = None,
    os: Optional[str] = None,
    min_score: float = 0.0,
    limit: int = 100
):
    """
    Get list of approved trajectories for miners to review.
    """
    # Query database for approved trajectories
    # Return list with scores and metadata

@app.get("/api/agent/download_trajectory/{trajectory_id}")
async def download_trajectory(trajectory_id: str):
    """
    Download complete trajectory with images.
    Returns ZIP file with trajectory.json + images/
    """
    # Package trajectory + images into ZIP
    # Return as download
```

---

### PHASE 2: Action Extraction & Trajectory Generation (Weeks 3-5)

This is the most complex phase - extracting structured action sequences from screen recordings.

#### 2.1 Action Extraction Module

**File**: `omega/action_extraction.py` (new)

```python
class ActionExtractor:
    """
    Extract action sequences from screen recordings.
    Uses computer vision + LLM to generate PyAutoGUI-style commands.
    """
    
    def __init__(self):
        # Load models
        self.vision_model = load_vision_model()  # e.g., GPT-4V, Gemini
        self.ocr_engine = pytesseract
        self.ui_detector = load_ui_detection_model()
    
    async def extract_from_video(
        self,
        video_path: str,
        task_description: str,
        fps: int = 1  # Sample 1 frame per second
    ) -> List[TrajectoryStep]:
        """
        Main extraction pipeline.
        
        Steps:
        1. Extract frames from video at specified FPS
        2. For each frame:
           a. Detect UI elements (buttons, text fields, etc.)
           b. Compare with previous frame to detect changes
           c. Infer action that caused change
           d. Generate PyAutoGUI command
           e. Use LLM to generate thought/reflection/observation
        3. Return complete trajectory
        """
        
        frames = extract_frames_from_video(video_path, fps)
        trajectory = []
        
        for i, (frame, next_frame) in enumerate(zip(frames[:-1], frames[1:])):
            # Detect what changed
            change = self.detect_change(frame, next_frame)
            
            if change is None:
                continue  # No significant change
            
            # Infer action
            action_info = await self.infer_action(
                frame, next_frame, change, task_description
            )
            
            # Generate step
            step = TrajectoryStep(
                index=i,
                image=f"{uuid.uuid4()}.png",
                code=action_info.code,
                action=action_info.action,
                thought=action_info.thought,
                reflection=action_info.reflection,
                observation=action_info.observation,
                last_step_correct=True,  # Default to True, validator will check
                last_step_redundant=False,
            )
            
            trajectory.append(step)
        
        return trajectory
    
    async def infer_action(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        change: Dict,
        task_description: str
    ) -> ActionInfo:
        """
        Use vision LLM to infer what action was taken.
        """
        
        prompt = f"""
        You are analyzing a screen recording of a computer use task.
        
        Task: {task_description}
        
        I'm showing you two consecutive frames. Something changed between them.
        Change detected: {change}
        
        Please infer:
        1. What action the user/agent took (e.g., "clicked button", "typed text")
        2. The PyAutoGUI command that would reproduce this action
        3. The agent's likely reasoning (thought) before taking this action
        4. The agent's reflection after seeing the result
        5. A detailed observation of the screen state after the action
        
        Format your response as JSON matching this schema:
        {{
            "code": "pyautogui.click(x=0.5, y=0.3)",
            "action": "Clicked the 'Submit' button",
            "thought": "I need to submit the form to complete the task",
            "reflection": "The form was successfully submitted, and a confirmation message appeared",
            "observation": "The screen now shows a green confirmation message saying 'Form submitted successfully'"
        }}
        """
        
        # Call vision model with both frames
        response = await self.vision_model.generate(
            prompt=prompt,
            images=[frame_before, frame_after]
        )
        
        return ActionInfo.parse_raw(response)
```

**Alternative Approach (Cheaper/Faster)**:

If using LLMs for every step is too expensive, consider:

1. **Template-based extraction**:
   - Use CV to detect clicks (mouse cursor movement + sudden stop)
   - Use OCR diff to detect typing
   - Use window detection to detect app switches
   - Generate actions from templates

2. **Hybrid approach**:
   - Use CV for action detection
   - Use LLM only for thought/reflection/observation generation
   - Batch process for cost efficiency

#### 2.2 Integration with Focus Videos

**File**: `validator_api/validator_api/focus_to_trajectory.py` (new)

```python
async def convert_focus_video_to_trajectory(
    video_id: str,
    task_description: str,
    user_id: str
) -> AgentTrajectory:
    """
    Convert existing Focus video to agent trajectory format.
    
    This allows gradual migration - existing Focus videos can be
    converted to trajectory format without re-recording.
    """
    
    # 1. Get video from GCS
    video_path = download_from_gcs(video_id)
    
    # 2. Extract actions
    extractor = ActionExtractor()
    steps = await extractor.extract_from_video(video_path, task_description)
    
    # 3. Upload step images to GCS
    image_ids = []
    for step in steps:
        image_path = extract_frame_at_step(video_path, step.index)
        gcs_uri = upload_image_to_gcs(image_path, step.image)
        image_ids.append(step.image)
    
    # 4. Create trajectory
    trajectory = AgentTrajectory(
        trajectory_id=str(uuid.uuid4()),
        video_id=video_id,
        user_id=user_id,
        instruction=task_description,
        traj=steps,
        # ... other fields
    )
    
    return trajectory
```

---

### PHASE 3: Validator Scoring Logic (Weeks 5-7)

#### 3.1 Trajectory Scoring Service

**File**: `validator_api/validator_api/scoring/trajectory_scoring_service.py` (new)

```python
class TrajectoryScoringService:
    """
    Score agent trajectories on multiple dimensions.
    Similar to Focus video scoring, but different criteria.
    """
    
    def __init__(self):
        self.pinecone_index = get_pinecone_index("trajectory-embeddings")
        self.llm_client = get_llm_client()  # DeepSeek or similar
    
    async def score_trajectory(
        self,
        trajectory: AgentTrajectory
    ) -> TrajectoryScore:
        """
        Complete scoring pipeline.
        
        Phases:
        1. Format validation
        2. Action validity check
        3. Trajectory coherence evaluation
        4. Novelty check (Pinecone)
        5. Task completion scoring (LLM)
        6. Final score calculation
        """
        
        # PHASE 1: Format Validation
        self.validate_format(trajectory)
        
        # PHASE 2: Action Validity
        action_validity_score = await self.check_action_validity(trajectory)
        
        # PHASE 3: Coherence
        coherence_score = await self.check_coherence(trajectory)
        
        # PHASE 4: Novelty
        novelty_score = await self.check_novelty(trajectory)
        
        # PHASE 5: Completion
        completion_score = await self.evaluate_completion(trajectory)
        
        # PHASE 6: Final Score
        final_score = self.calculate_final_score(
            action_validity_score,
            coherence_score,
            novelty_score,
            completion_score
        )
        
        return TrajectoryScore(
            action_validity_score=action_validity_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            completion_score=completion_score,
            final_score=final_score,
            details={...}
        )
    
    async def check_action_validity(self, trajectory: AgentTrajectory) -> float:
        """
        Verify each action is plausible given the UI state.
        
        Checks:
        - PyAutoGUI syntax is valid
        - Click coordinates are within screen bounds
        - Actions are appropriate for the application
        - No impossible sequences (e.g., typing without focus)
        """
        
        valid_count = 0
        total_count = len(trajectory.traj)
        
        for step in trajectory.traj:
            # Parse PyAutoGUI command
            try:
                action = parse_pyautogui_command(step.code)
            except:
                continue  # Invalid syntax
            
            # Check coordinates
            if action.type == "click":
                if not (0 <= action.x <= 1 and 0 <= action.y <= 1):
                    continue  # Out of bounds
            
            # More sophisticated checks here...
            
            valid_count += 1
        
        return valid_count / total_count
    
    async def check_coherence(self, trajectory: AgentTrajectory) -> float:
        """
        Evaluate if trajectory steps logically follow each other.
        
        Uses LLM to check if:
        - Each action makes sense given the previous state
        - The sequence progresses toward the goal
        - No contradictory actions
        """
        
        prompt = f"""
        Evaluate the logical coherence of this action sequence for the task: "{trajectory.instruction}"
        
        Actions:
        {format_trajectory_for_llm(trajectory)}
        
        Rate coherence from 0.0 (incoherent) to 1.0 (perfectly logical).
        Consider:
        - Do actions follow logically?
        - Is there unnecessary backtracking?
        - Are there contradictory actions?
        
        Return just a number between 0.0 and 1.0.
        """
        
        response = await self.llm_client.generate(prompt)
        return float(response.strip())
    
    async def check_novelty(self, trajectory: AgentTrajectory) -> float:
        """
        Check if this trajectory is sufficiently different from existing ones.
        
        Uses Pinecone to find similar trajectories and compute uniqueness.
        """
        
        # Generate embedding for this trajectory
        embedding = await self.embed_trajectory(trajectory)
        
        # Query Pinecone
        results = await self.pinecone_index.query(
            vector=embedding,
            top_k=1
        )
        
        if len(results.matches) == 0:
            return 1.0  # Completely unique
        
        similarity = results.matches[0].score
        return 1.0 - similarity  # Return novelty (inverse of similarity)
    
    async def embed_trajectory(self, trajectory: AgentTrajectory) -> List[float]:
        """
        Generate embedding for trajectory.
        
        Combines:
        - Task description embedding
        - Action sequence embedding
        - Application/domain embedding
        """
        
        # Create text representation
        text = f"""
        Task: {trajectory.instruction}
        Domain: {trajectory.domain}
        OS: {trajectory.os}
        Actions: {", ".join([step.action for step in trajectory.traj])}
        """
        
        # Use OpenAI or similar
        embedding = await get_text_embedding(text)
        return embedding
    
    async def evaluate_completion(self, trajectory: AgentTrajectory) -> float:
        """
        Use LLM to evaluate if task was completed successfully.
        
        Similar to Focus video completion scoring.
        """
        
        prompt = f"""
        Evaluate if this agent successfully completed the task.
        
        Task: {trajectory.instruction}
        
        Actions taken:
        {format_trajectory_steps(trajectory.traj)}
        
        Final observation: {trajectory.traj[-1].observation}
        
        Did the agent complete the task? Rate from 0.0 (failed) to 1.0 (perfect completion).
        
        Consider:
        - Was the goal achieved?
        - Were actions appropriate?
        - Was it done efficiently?
        
        Return JSON:
        {{
            "completion_score": 0.85,
            "rationale": "The agent successfully..."
        }}
        """
        
        response = await self.llm_client.generate(prompt, response_format="json")
        result = json.loads(response)
        
        return result["completion_score"]
    
    def calculate_final_score(
        self,
        action_validity: float,
        coherence: float,
        novelty: float,
        completion: float
    ) -> float:
        """
        Combine individual scores into final score.
        
        Weights:
        - Action validity: 20% (must be technically correct)
        - Coherence: 20% (must be logical)
        - Novelty: 20% (must be unique)
        - Completion: 40% (most important - did it work?)
        """
        
        final = (
            action_validity * 0.20 +
            coherence * 0.20 +
            novelty * 0.20 +
            completion * 0.40
        )
        
        return max(0.0, min(1.0, final))
```

#### 3.2 Validator Integration

**File**: `neurons/validator.py` (modify)

```python
# Add new forward function for trajectories

async def forward_agent_trajectories(self):
    """
    Query miners for agent trajectory submissions.
    """
    
    # Select miners
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    
    # Send request
    query = random.choice(self.all_task_categories)  # e.g., "productivity", "coding"
    input_synapse = AgentTrajectories(
        query=query,
        num_trajectories=4
    )
    
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    responses = await self.dendrite(
        axons=axons,
        synapse=input_synapse,
        deserialize=False,
        timeout=CLIENT_TIMEOUT_SECONDS_TRAJECTORY,  # e.g., 180s
    )
    
    # Score responses
    rewards = await self.handle_checks_and_rewards_trajectories(
        input_synapse=input_synapse,
        responses=responses
    )
    
    # Update scores
    self.update_trajectory_scores(rewards, miner_uids)
```

---

### PHASE 4: Rewards Transition & HuggingFace Upload (Weeks 7-8)

#### 4.1 Gradual Rewards Shift

Update reward distribution over time:

**Week 1-2** (Testing):
```python
YOUTUBE_REWARDS_PERCENT = 0.60      # 60%
FOCUS_REWARDS_PERCENT = 0.30        # 30%
AGENT_TRAJECTORY_REWARDS_PERCENT = 0.10  # 10% (testing)
```

**Week 3-4** (Ramp up):
```python
YOUTUBE_REWARDS_PERCENT = 0.40      # 40%
FOCUS_REWARDS_PERCENT = 0.20        # 20%
AGENT_TRAJECTORY_REWARDS_PERCENT = 0.40  # 40%
```

**Week 5-6** (Majority):
```python
YOUTUBE_REWARDS_PERCENT = 0.20      # 20%
FOCUS_REWARDS_PERCENT = 0.10        # 10%
AGENT_TRAJECTORY_REWARDS_PERCENT = 0.70  # 70%
```

**Week 7+** (Full migration):
```python
YOUTUBE_REWARDS_PERCENT = 0.00      # 0% (deprecated)
FOCUS_REWARDS_PERCENT = 0.00        # 0% (deprecated)
AGENT_TRAJECTORY_REWARDS_PERCENT = 1.00  # 100%
```

#### 4.2 HuggingFace Upload

**File**: `validator_api/validator_api/dataset_upload.py` (modify)

```python
class AgentTrajectoryDatasetUploader:
    """
    Upload agent trajectories to HuggingFace in AgentNet-compatible format.
    """
    
    def __init__(self):
        self.current_batch = []
        self.min_batch_size = 16
        self.desired_batch_size = 128
        self.max_batch_size = 512
        
        self.hf_repo = "omegalabsinc/omega-agent-trajectories"
    
    def add_trajectories(
        self,
        trajectories: List[AgentTrajectory],
        scores: List[TrajectoryScore]
    ):
        """Add validated trajectories to batch."""
        
        for trajectory, score in zip(trajectories, scores):
            # Convert to HF format (matches AgentNet schema)
            self.current_batch.append({
                "task_id": trajectory.trajectory_id,
                "instruction": trajectory.instruction,
                "natural_language_task": trajectory.natural_language_task,
                "actual_task": trajectory.actual_task,
                
                # Trajectory steps (JSON)
                "traj": [
                    {
                        "index": step.index,
                        "image": step.image,
                        "value": {
                            "code": step.code,
                            "action": step.action,
                            "thought": step.thought,
                            "reflection": step.reflection,
                            "observation": step.observation,
                            "last_step_correct": step.last_step_correct,
                            "last_step_redundant": step.last_step_redundant,
                        }
                    }
                    for step in trajectory.traj
                ],
                
                # Metadata
                "domain": trajectory.domain,
                "task_completed": trajectory.task_completed,
                "task_difficulty": trajectory.task_difficulty,
                "alignment_score": score.action_validity_score * 10,  # Scale to 0-10
                "efficiency_score": score.coherence_score * 10,
                "reason": trajectory.reason,
                
                # SN24-specific (optional, for analytics)
                "omega_final_score": score.final_score,
                "omega_novelty_score": score.novelty_score,
                "omega_completion_score": score.completion_score,
                "omega_submitted_at": trajectory.submitted_at,
                "omega_miner_hotkey": trajectory.miner_hotkey,
            })
        
        if len(self.current_batch) >= self.desired_batch_size:
            self.submit()
    
    def submit(self):
        """Upload batch to HuggingFace."""
        
        if len(self.current_batch) < self.min_batch_size:
            return
        
        # Take batch
        data_to_upload = self.current_batch[:self.desired_batch_size]
        self.current_batch = self.current_batch[self.desired_batch_size:]
        
        # Create JSONL file (matches AgentNet format)
        jsonl_content = "\n".join(
            json.dumps(item) for item in data_to_upload
        )
        
        # Upload to HF
        batch_id = str(ulid.new())
        path = f"data/{batch_id}.jsonl"
        
        HF_API.upload_file(
            path_or_fileobj=BytesIO(jsonl_content.encode("utf-8")),
            path_in_repo=path,
            repo_id=self.hf_repo,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
        
        bt.logging.info(f"Uploaded {len(data_to_upload)} trajectories to HF: {path}")
    
    async def upload_images(self, trajectory_id: str):
        """
        Upload trajectory screenshots to HuggingFace.
        
        Images stored separately (like AgentNet):
        - data/images/{trajectory_id}/0.png
        - data/images/{trajectory_id}/1.png
        - ...
        """
        
        # Get images from database
        images = await get_trajectory_images(trajectory_id)
        
        for image in images:
            # Download from GCS
            image_data = download_from_gcs(image.gcs_uri)
            
            # Upload to HF
            path = f"data/images/{trajectory_id}/{image.step_index}.png"
            
            HF_API.upload_file(
                path_or_fileobj=BytesIO(image_data),
                path_in_repo=path,
                repo_id=self.hf_repo,
                repo_type="dataset",
                token=config.HF_TOKEN,
            )
```

---

## 5. MINER IMPLEMENTATION

### 5.1 New Miner Forward Function

**File**: `neurons/miner.py` (add)

```python
async def forward_agent_trajectories(self, synapse: omega.protocol.AgentTrajectories):
    """
    Handle trajectory submission requests from validators.
    
    Miners can:
    1. Submit their own recorded trajectories (if they have them)
    2. Proxy requests to Focus users
    3. Generate synthetic trajectories (advanced)
    """
    
    # Option 1: Return pre-recorded trajectories
    trajectories = await self.get_prerecorded_trajectories(
        query=synapse.query,
        num_trajectories=synapse.num_trajectories
    )
    
    synapse.trajectory_metadata = trajectories
    return synapse

async def get_prerecorded_trajectories(
    self,
    query: str,
    num_trajectories: int
) -> List[AgentTrajectoryMetadata]:
    """
    Retrieve pre-recorded trajectories from local storage or API.
    
    Miners can:
    - Record their own trajectories
    - Purchase from Focus users
    - Generate synthetically
    """
    
    # Load from local database or API
    trajectories = load_trajectories_from_storage(
        category=query,
        limit=num_trajectories
    )
    
    return trajectories
```

### 5.2 Miner Recording Tool

Create a simple tool for miners to record their own trajectories:

**File**: `scripts/record_trajectory.py` (new)

```python
"""
Tool for miners to record their own computer use trajectories.

Usage:
    python scripts/record_trajectory.py \\
        --task "Debug Python script" \\
        --output trajectory.json
"""

import argparse
from omega.action_extraction import ActionExtractor
import mss  # For screen recording
import keyboard
import mouse

class TrajectoryRecorder:
    """Record screen + actions in real-time."""
    
    def __init__(self):
        self.actions = []
        self.screenshots = []
        self.start_time = None
    
    def start_recording(self, task_description: str):
        """Start recording user actions."""
        
        print(f"Recording task: {task_description}")
        print("Press ESC to stop recording")
        
        self.start_time = time.time()
        
        # Set up listeners
        keyboard.on_press(self.on_key_press)
        mouse.on_click(self.on_mouse_click)
        
        # Capture screenshots at 1 FPS
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            
            while True:
                # Check if ESC pressed
                if keyboard.is_pressed('esc'):
                    break
                
                # Capture screenshot
                img = sct.grab(monitor)
                self.screenshots.append(np.array(img))
                
                time.sleep(1.0)  # 1 FPS
        
        print(f"Recording stopped. Captured {len(self.screenshots)} frames")
    
    def on_key_press(self, event):
        """Record keyboard event."""
        self.actions.append({
            "type": "keypress",
            "key": event.name,
            "timestamp": time.time() - self.start_time
        })
    
    def on_mouse_click(self, x, y, button, pressed):
        """Record mouse click."""
        if pressed:
            self.actions.append({
                "type": "click",
                "x": x,
                "y": y,
                "button": button.name,
                "timestamp": time.time() - self.start_time
            })
    
    async def generate_trajectory(
        self,
        task_description: str,
        output_path: str
    ):
        """
        Process recorded actions into trajectory format.
        """
        
        # Use ActionExtractor to generate thoughts/reflections/observations
        extractor = ActionExtractor()
        
        trajectory = await extractor.process_recording(
            screenshots=self.screenshots,
            actions=self.actions,
            task_description=task_description
        )
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(trajectory.dict(), f, indent=2)
        
        print(f"Trajectory saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    
    recorder = TrajectoryRecorder()
    recorder.start_recording(args.task)
    
    asyncio.run(recorder.generate_trajectory(args.task, args.output))
```

---

## 6. CLEANUP: REMOVING DEPRECATED CODE

### 6.1 Files to Remove (After Full Migration)

```bash
# YouTube-specific code (no longer needed)
rm omega/video_utils.py
rm omega/imagebind_wrapper.py
rm omega/miner_utils.py  # Replace with trajectory_miner_utils.py
rm omega/augment.py

# Audio-specific (if not needed)
rm omega/diarization_pipeline.py
rm omega/audio_scoring.py
rm omega/test_audio.py

# Old tests
rm TEST_get_emission.py
rm test_audio_dataset.py
```

### 6.2 Files to Keep (Core Infrastructure)

```bash
# Keep these - they're core Bittensor infrastructure
omega/base/
omega/protocol.py  # Modified for trajectories
neurons/validator.py  # Modified
neurons/miner.py  # Modified
validator_api/  # Modified for trajectories
scripts/
docs/
```

### 6.3 Code Sections to Remove

**In `neurons/validator.py`**:
```python
# Remove YouTube scoring functions
# - handle_checks_and_rewards_youtube()
# - update_scores() [YouTube-specific]
# - YouTube validation logic

# Remove audio scoring
# - forward_audios()
# - handle_checks_and_reward_audio()
```

**In `omega/protocol.py`**:
```python
# Keep Videos/Audios for backward compatibility during migration
# But deprecate after Phase 4
```

---

## 7. TESTING STRATEGY

### 7.1 Phase 1 Testing

**Goal**: Verify infrastructure works

```bash
# Test database migrations
python scripts/test_db_migrations.py

# Test API endpoints
python scripts/test_trajectory_api.py

# Test protocol serialization
python scripts/test_trajectory_protocol.py
```

### 7.2 Phase 2 Testing

**Goal**: Verify action extraction works

```bash
# Test on sample videos
python scripts/test_action_extraction.py \\
    --video sample_videos/coding_task.mp4 \\
    --task "Debug Python script"

# Verify output matches AgentNet format
python scripts/validate_trajectory_format.py \\
    --trajectory output/trajectory.json
```

### 7.3 Phase 3 Testing

**Goal**: Verify scoring works

```bash
# Test scoring service
python scripts/test_trajectory_scoring.py \\
    --trajectory test_data/sample_trajectory.json

# Compare scores with manual evaluation
python scripts/compare_scoring_results.py
```

### 7.4 Phase 4 Testing

**Goal**: Verify end-to-end flow

```bash
# Test full validator flow
python neurons/validator.py --netuid 24 --subtensor.network test

# Test miner submissions
python neurons/miner.py --netuid 24 --subtensor.network test

# Verify HuggingFace uploads
python scripts/verify_hf_uploads.py
```

---

## 8. DEPLOYMENT PLAN

### 8.1 Testnet Deployment (Week 6)

1. Deploy updated code to testnet
2. Run alongside existing system (parallel testing)
3. Monitor metrics:
   - Trajectory submission rate
   - Scoring accuracy
   - HuggingFace upload success rate
4. Fix bugs

### 8.2 Mainnet Soft Launch (Week 7)

1. Deploy to mainnet with low reward percentage (10%)
2. Monitor miner adoption
3. Validate data quality
4. Collect feedback

### 8.3 Mainnet Full Launch (Week 8)

1. Increase rewards gradually
2. Deprecate YouTube/Focus video paths
3. Full migration complete

---

## 9. COST ANALYSIS

### 9.1 LLM Costs

**Action Extraction** (per trajectory):
- Frames: ~60 frames @ 1 FPS (1 minute video)
- LLM calls: ~60 calls to vision model
- Cost per call: ~$0.01 (GPT-4V) or ~$0.002 (Gemini)
- **Total per trajectory: $0.12 - $0.60**

**Optimization**:
- Use cheaper models (Gemini, Claude Haiku)
- Batch processing
- Cache common patterns
- **Target: <$0.10 per trajectory**

**Scoring** (per trajectory):
- Coherence check: 1 LLM call (~$0.001)
- Completion evaluation: 1 LLM call (~$0.001)
- **Total: ~$0.002**

### 9.2 Storage Costs

**Images**:
- Screenshots: ~60 per trajectory @ 100KB each
- Total: ~6MB per trajectory
- GCS cost: $0.00012 per trajectory
- **Negligible**

**Database**:
- Trajectory JSON: ~50KB per trajectory
- PostgreSQL storage: ~$0.000005 per trajectory
- **Negligible**

### 9.3 Total Cost Estimate

**Per 1000 trajectories**:
- Action extraction: $100-600
- Scoring: $2
- Storage: $0.12
- **Total: ~$100-600**

**Monthly (targeting 10K trajectories)**:
- **$1000-6000/month in LLM costs**

**Optimization strategies**:
1. Use Gemini 2.5 Flash ($0.002/call) instead of GPT-4V
2. Implement caching for common UI patterns
3. Use template-based extraction where possible
4. Batch process during off-peak hours

**Target**: <$2000/month for 10K trajectories

---

## 10. SUCCESS METRICS

### 10.1 Adoption Metrics

- **Trajectory submission rate**: Target 100+ per day by Week 8
- **Miner participation**: 70%+ of active miners submitting
- **Focus user adoption**: 50%+ of Focus users recording trajectories

### 10.2 Quality Metrics

- **Average trajectory score**: Target >0.70
- **Rejection rate**: <20%
- **Uniqueness (novelty)**: >0.80 average

### 10.3 Dataset Metrics

- **Total trajectories**: 10,000+ by end of Phase 4
- **Unique tasks**: 1,000+ distinct tasks
- **OS diversity**: 40% Ubuntu, 30% macOS, 30% Windows
- **Domain coverage**: All major categories (coding, productivity, design, etc.)

---

## 11. RISKS & MITIGATION

### 11.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Action extraction accuracy too low | High | Medium | Use hybrid CV+LLM approach, iterate on prompts |
| LLM costs too high | High | Medium | Use cheaper models, optimize prompts, cache |
| Miners don't adopt | High | Low | Gradual rewards increase, documentation, support |
| Trajectories too similar (low novelty) | Medium | Medium | Strong novelty scoring, diverse task prompts |
| Scoring too slow | Medium | Low | Async processing, batch operations |

### 11.2 Economic Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Miners game the system | High | Medium | Robust validation, spot-checking, blacklist |
| Data quality degrades | High | Medium | Continuous monitoring, manual review samples |
| Rewards not attractive enough | Medium | Low | Dynamic reward adjustment based on participation |

---

## 12. NEXT STEPS

### Immediate Actions (Week 1):

1. **Review this document** with team and get approval
2. **Set up development branch**: `git checkout -b revamp-agent-trajectories`
3. **Create Phase 1 tasks** in project management tool
4. **Allocate resources**:
   - 1-2 backend engineers for API/database
   - 1 ML engineer for action extraction
   - 1 devops for deployment
5. **Set up monitoring**: Create dashboard for tracking metrics

### Week 1 Deliverables:

- [ ] Database schema created and tested
- [ ] API endpoints implemented (basic versions)
- [ ] Protocol definitions added to omega/protocol.py
- [ ] Test dataset created (10 sample trajectories)

### Week 2 Deliverables:

- [ ] Action extraction module prototype
- [ ] Scoring service basic implementation
- [ ] Integration tests passing
- [ ] Documentation updated

---

## 13. REFERENCES

- **AgentNet Dataset**: https://huggingface.co/datasets/xlangai/AgentNet
- **Current SN24 Docs**: `/docs/focus_video_scoring_and_embeddings.md`
- **Bittensor Docs**: https://docs.bittensor.com/
- **PyAutoGUI Docs**: https://pyautogui.readthedocs.io/

---

## APPENDIX A: AGENTNET DATASET STATISTICS

From our analysis:

- **Total entries**: 22.6K computer-use tasks
- **Split**: 5K Ubuntu + 18K Win/Mac
- **Average trajectory length**: 7-8 steps
- **Screenshot format**: PNG images, stored separately
- **Data format**: JSONL (JSON Lines)
- **Image storage**: Compressed ZIP archives

---

## APPENDIX B: SAMPLE TRAJECTORY

```json
{
    "task_id": "0030dc52-2a4a-4c0e-895b-48284c200efe",
    "instruction": "Open the Pikachu picture on the desktop using GIMP, and then select the 'Venetian Blinds' filter in the animation.",
    "natural_language_task": "Open the image editing software GIMP, load the pikachu.jpeg file from the desktop...",
    "actual_task": "Launch GIMP application, open the pikachu.jpeg file...",
    "domain": "image-editing",
    "os": "ubuntu",
    "task_difficulty": 4,
    "task_completed": false,
    "alignment_score": 6,
    "efficiency_score": 9,
    "reason": "The agent successfully completed the first part...",
    "traj": [
        {
            "index": 0,
            "image": "30a6f01b-9daf-4107-a755-a4a602d0de8e.png",
            "value": {
                "code": "pyautogui.click(x=0.018, y=0.508)",
                "action": "Click on the GIMP application icon in the left taskbar to launch the image editing software.",
                "thought": "I need to start by opening GIMP since the task requires working with an image in GIMP...",
                "observation": "The current screen shows a desktop environment with a blue background...",
                "reflection": "The last action successfully launched GIMP...",
                "last_step_correct": true,
                "last_step_redundant": false
            }
        }
    ]
}
```

---

**END OF MIGRATION PLAN**
