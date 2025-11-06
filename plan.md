# SN24 Computer Use Agent Dataset - REVISED Strategic Plan

## Critical Realization

**My previous plan was fundamentally flawed.** You cannot reliably extract precise click coordinates and keyboard inputs from video frames alone. We need **real-time event capture** during recording.

**The Truth**: To create high-quality computer use training data, we need:
1. ✅ Screen recording (we have this via Focus Videos)
2. ❌ **Precise mouse coordinates and clicks** (we DON'T have this)
3. ❌ **Keyboard inputs and text typed** (we DON'T have this)
4. ❌ **Window/application context** (we DON'T have this)

## Revised Executive Summary

Transform SN24 by building a **new recording client** that captures:
- Screen video (like Focus app does now)
- **+ Real-time mouse events** (position, clicks, drags)
- **+ Keyboard events** (keys pressed, text typed)
- **+ System context** (active window, running apps)

Then use AI to add reasoning/observations to create complete AgentNet-style trajectories.

**Timeline**: 8-12 weeks (cannot be rushed - need to rebuild recording layer)
**Cost**: ~$200-500/month (AI for reasoning/observations only, not action extraction)

---

## 1. The Hard Truth: What We Actually Need

### 1.1 AgentNet Data Requirements

Looking at the actual AgentNet data:
```python
{
    "code": "pyautogui.click(x=0.018, y=0.508)",  # PRECISE coordinates needed
    "action": "Clicked the GIMP icon",
    "thought": "I need to open GIMP",
    "observation": "Desktop with taskbar visible",
    "reflection": "GIMP successfully opened"
}
```

**Cannot be extracted from video alone:**
- `x=0.018, y=0.508` - Exact normalized coordinates (0-1 range)
- Mouse movements between frames
- Keyboard key sequences (can't OCR what was typed)
- Right-clicks vs left-clicks vs double-clicks

**Can be inferred from video with AI:**
- `action`: "Clicked the GIMP icon" 
- `thought`: Why the action was taken
- `observation`: What was visible on screen
- `reflection`: What happened after

### 1.2 What Focus Videos Currently Provide

```
Current Focus Video:
- Screen recording (MP4)
- Task description
- User's description of what they did

Missing:
- Mouse position data
- Click events (with coordinates)
- Keyboard events
- Scroll events
- Window focus changes
```

---

## 2. Two-Path Strategy

### Path A: New Recording Client (Recommended)

Build a proper event-capture recording tool that replaces/enhances the Focus app.

### Path B: Hybrid Approach (Faster, Lower Quality)

Use existing Focus Videos + AI inference for approximate coordinates (good enough for many use cases).

---

## 3. PATH A: New Recording Client (HIGH QUALITY)

### 3.1 Architecture

```
┌─────────────────────────────────────────────────┐
│         Enhanced Focus Recording Client          │
├─────────────────────────────────────────────────┤
│                                                  │
│  Screen Capture Thread                          │
│  ├─ Record video (ffmpeg)                       │
│  └─ Capture screenshots at events               │
│                                                  │
│  Event Capture Thread                           │
│  ├─ Mouse: position, clicks, scrolls            │
│  ├─ Keyboard: keys pressed, text typed          │
│  ├─ Window: active app, focused element         │
│  └─ Timestamp synchronization                   │
│                                                  │
│  Output: recording.mp4 + events.json            │
└─────────────────────────────────────────────────┘
```

### 3.2 Technical Implementation

#### Client-Side Recording (Cross-Platform)

**Option 1: Python-Based (Easiest)**
```python
# omega_recorder/recorder.py

import pyautogui
import pynput
import mss
import cv2
import json
from datetime import datetime

class OmegaRecorder:
    """
    Record screen + mouse/keyboard events simultaneously.
    """
    
    def __init__(self):
        self.events = []
        self.start_time = None
        self.screen_size = pyautogui.size()
        
    def start_recording(self, task_description: str):
        """Start recording video + events."""
        
        self.start_time = datetime.now()
        self.task_description = task_description
        
        # Start video recording thread
        self.video_thread = threading.Thread(target=self._record_video)
        self.video_thread.start()
        
        # Set up event listeners
        self.mouse_listener = pynput.mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_scroll
        )
        self.keyboard_listener = pynput.keyboard.Listener(
            on_press=self._on_key_press
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        print("🔴 Recording started... Press ESC to stop")
        
    def _on_mouse_click(self, x, y, button, pressed):
        """Capture mouse click with precise coordinates."""
        if pressed:
            # Normalize coordinates to 0-1 range (like AgentNet)
            norm_x = x / self.screen_size.width
            norm_y = y / self.screen_size.height
            
            event = {
                "type": "click",
                "timestamp": (datetime.now() - self.start_time).total_seconds(),
                "x": round(norm_x, 3),
                "y": round(norm_y, 3),
                "button": button.name,  # 'left', 'right', 'middle'
                "code": f"pyautogui.click(x={norm_x:.3f}, y={norm_y:.3f})"
            }
            
            self.events.append(event)
            print(f"  Click at ({norm_x:.3f}, {norm_y:.3f})")
    
    def _on_key_press(self, key):
        """Capture keyboard input."""
        try:
            char = key.char
            event = {
                "type": "keypress",
                "timestamp": (datetime.now() - self.start_time).total_seconds(),
                "key": char,
                "code": f"pyautogui.press('{char}')"
            }
        except AttributeError:
            # Special key (enter, ctrl, etc.)
            event = {
                "type": "keypress",
                "timestamp": (datetime.now() - self.start_time).total_seconds(),
                "key": str(key),
                "code": f"pyautogui.press('{key.name}')"
            }
        
        self.events.append(event)
    
    def _record_video(self):
        """Record screen video."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            fps = 30
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('recording.mp4', fourcc, fps, 
                                  (monitor["width"], monitor["height"]))
            
            while self.recording:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                out.write(img)
                
            out.release()
    
    def stop_recording(self):
        """Stop recording and save events."""
        self.recording = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.video_thread.join()
        
        # Save events to JSON
        output = {
            "task": self.task_description,
            "events": self.events,
            "duration": (datetime.now() - self.start_time).total_seconds()
        }
        
        with open('events.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Recording saved: {len(self.events)} events captured")
```

**Option 2: Electron App (Better UX)**
- Build cross-platform desktop app
- Native OS event hooks (more reliable)
- Better UI for users
- Can integrate with existing Focus app

#### Server-Side Processing

```python
# validator_api/validator_api/services/trajectory_builder.py

class TrajectoryBuilder:
    """
    Convert raw events.json + video into AgentNet format.
    Uses AI to add reasoning/observations.
    """
    
    async def build_trajectory(
        self, 
        video_id: str,
        events: List[Dict],
        task_description: str
    ) -> Dict:
        """
        Combine events with AI-generated context.
        """
        
        # 1. Group events into logical steps
        steps = self._group_events_into_steps(events)
        
        # 2. For each step, get screenshot
        screenshots = await self._extract_screenshots(video_id, steps)
        
        # 3. Use AI to generate thought/observation/reflection
        trajectory = []
        
        for i, (step, screenshot) in enumerate(zip(steps, screenshots)):
            # We have precise action from events.json
            action_code = step['code']  # e.g., "pyautogui.click(x=0.5, y=0.3)"
            
            # Use Gemini to generate context
            context = await self._generate_context(
                screenshot=screenshot,
                action_code=action_code,
                task_description=task_description,
                previous_steps=trajectory[:i]
            )
            
            trajectory.append({
                "index": i,
                "image": f"{video_id}_{i}.png",
                "value": {
                    "code": action_code,  # From recorded events (precise!)
                    "action": context['action'],  # From AI
                    "thought": context['thought'],  # From AI
                    "observation": context['observation'],  # From AI
                    "reflection": context['reflection'],  # From AI
                    "last_step_correct": True,
                    "last_step_redundant": False
                }
            })
        
        return {
            "task_id": str(uuid.uuid4()),
            "instruction": task_description,
            "traj": trajectory,
            # ... other fields
        }
    
    def _group_events_into_steps(self, events: List[Dict]) -> List[Dict]:
        """
        Group rapid events into single steps.
        
        Example:
        - Multiple keypresses → "typed 'hello world'"
        - Click + small movement + click → "double clicked"
        """
        
        steps = []
        current_step = []
        last_timestamp = 0
        
        for event in events:
            # If >1 second gap, start new step
            if event['timestamp'] - last_timestamp > 1.0:
                if current_step:
                    steps.append(self._merge_events(current_step))
                current_step = [event]
            else:
                current_step.append(event)
            
            last_timestamp = event['timestamp']
        
        if current_step:
            steps.append(self._merge_events(current_step))
        
        return steps
    
    async def _generate_context(
        self,
        screenshot: bytes,
        action_code: str,
        task_description: str,
        previous_steps: List[Dict]
    ) -> Dict:
        """
        Use Gemini to generate thought/observation/reflection.
        Cost: ~$0.002 per step.
        """
        
        prompt = f"""
        You are analyzing a computer use task.
        
        Task: {task_description}
        
        Previous steps: {json.dumps(previous_steps[-3:], indent=2) if previous_steps else "None"}
        
        The user just performed this action: {action_code}
        
        Looking at this screenshot, provide:
        1. action: Human-readable description of what was done
        2. thought: Why the user likely took this action
        3. observation: What is visible on screen now
        4. reflection: What happened as a result
        
        Return as JSON:
        {{
            "action": "Clicked the File menu",
            "thought": "I need to open a file, so clicking the File menu to see options",
            "observation": "The screen shows a code editor with a File menu dropdown now visible",
            "reflection": "The File menu opened successfully, showing options like Open, Save, etc."
        }}
        """
        
        response = await self.gemini_client.generate_content(
            contents=[
                {"text": prompt},
                {"image": screenshot}
            ]
        )
        
        return json.loads(response.text)
```

### 3.3 Integration with Focus App

**Two Options:**

**Option A: Replace Focus App Recording**
- Build new Omega Recorder app
- Users download and use instead of Focus app
- Uploads video + events.json to validator API

**Option B: Enhance Focus App**
- Add event capture to existing Focus app
- Minimal disruption to users
- Backward compatible (events.json optional initially)

### 3.4 Deployment

```bash
# Desktop App Distribution
# Windows: omega-recorder-setup.exe
# macOS: OmegaRecorder.dmg
# Linux: omega-recorder.AppImage

# Users install and run
omega-recorder start --task "Debug Python script"
# ... user performs task ...
omega-recorder stop

# Uploads:
# - recording.mp4 (screen video)
# - events.json (mouse/keyboard data)
# - metadata.json (task, duration, OS, etc.)
```

---

## 4. PATH B: Hybrid Approach (LOWER QUALITY, FASTER)

If rebuilding the recorder is not feasible, we can use AI to **estimate** coordinates.

### 4.1 AI-Based Coordinate Inference

```python
class CoordinateInferenceService:
    """
    Infer approximate click coordinates from video analysis.
    Accuracy: 70-80% (good enough for some use cases).
    """
    
    async def infer_coordinates(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        task_context: str
    ) -> Dict:
        """
        Use vision model to guess where user clicked.
        """
        
        prompt = """
        These are two consecutive frames from a screen recording.
        Something changed between them.
        
        Analyze:
        1. What UI element likely changed (button pressed, menu opened, etc.)
        2. Estimate the coordinates of where the user clicked (0-1 normalized)
        3. What type of action (click, double-click, right-click)
        
        Return as JSON:
        {
            "x": 0.523,  # Normalized x coordinate
            "y": 0.312,  # Normalized y coordinate
            "action_type": "left_click",
            "confidence": 0.85
        }
        """
        
        response = await self.gpt4v_client.analyze(
            prompt=prompt,
            images=[frame_before, frame_after]
        )
        
        return json.loads(response)
```

### 4.2 Limitations

- ❌ Coordinates are estimates (±20px error typical)
- ❌ Cannot detect keyboard inputs reliably
- ❌ Misses subtle actions (hover, right-click)
- ❌ Expensive (GPT-4V required: $0.60/video)
- ✅ Works with existing Focus Videos (no app changes)

---

## 5. RECOMMENDED APPROACH

### Phase 1: Build New Recorder (Weeks 1-4)

**Week 1-2: Core Recorder**
- [ ] Build Python-based recorder (mouse + keyboard + screen)
- [ ] Test on all platforms (Windows, macOS, Linux)
- [ ] Implement event grouping logic
- [ ] Test data quality with 10 sample recordings

**Week 3-4: Integration**
- [ ] Build trajectory builder service
- [ ] Integrate Gemini for context generation
- [ ] Test end-to-end: recording → processing → AgentNet format
- [ ] Validate against manual labels

### Phase 2: Deploy & Incentivize (Weeks 5-6)

**Week 5: Soft Launch**
- [ ] Release recorder to 10 beta testers
- [ ] Collect feedback
- [ ] Fix bugs
- [ ] Optimize performance

**Week 6: Public Release**
- [ ] Publish recorder (GitHub + download links)
- [ ] Create documentation/tutorials
- [ ] Announce 2x rewards for recordings with events.json
- [ ] Monitor adoption

### Phase 3: Scale & Optimize (Weeks 7-12)

**Week 7-8: Grow Dataset**
- [ ] Target 100 recordings/week
- [ ] Manual quality checks on samples
- [ ] Optimize AI prompts based on results
- [ ] Calculate actual costs

**Week 9-10: HuggingFace Migration**
- [ ] Upload trajectories in AgentNet format
- [ ] Maintain backward compatibility
- [ ] Announce new dataset

**Week 11-12: Full Transition**
- [ ] Make event capture mandatory
- [ ] Deprecate video-only submissions
- [ ] 100% trajectory-based rewards

---

## 6. Revised Cost Analysis

### With Proper Event Capture

**Recording Client**: Free (open source, users run locally)

**Server Processing Per Recording**:
- Video storage: $0.001 (GCS)
- Events.json storage: <$0.0001 (tiny file)
- AI context generation: 10 steps × $0.002 = $0.02 (Gemini)
- Screenshot extraction: $0.001
- **Total per recording: $0.022**

**Monthly (1000 recordings)**: $22
**Monthly (10,000 recordings)**: $220

**Much cheaper because**:
- No expensive coordinate inference needed
- AI only generates text (cheap)
- Precise actions come from event logs (free)

---

## 7. Alternative: Hybrid Start Strategy

### Months 1-2: Video-Only (Path B)
- Use AI inference for existing Focus Videos
- Lower quality but immediate results
- Build initial dataset (1000 trajectories)
- Cost: ~$600/month

### Month 3+: Event-Capture (Path A)
- Release new recorder
- Higher quality, lower cost
- Gradually replace video-only submissions
- Cost: ~$220/month

**Advantage**: Start collecting data immediately while building proper solution.

---

## 8. Critical Success Factors

### Must Have for Quality Data

1. **Precise coordinates** - Cannot train agents without exact click positions
2. **Keyboard inputs** - Need to know what was typed
3. **Timing** - Action sequences must have accurate timestamps
4. **Screenshots** - Visual context for each action

### Quality Metrics

- Coordinate precision: <5px error (Path A) vs ~20px (Path B)
- Action completeness: 100% (Path A) vs 60-70% (Path B)
- Cost per trajectory: $0.02 (Path A) vs $0.60 (Path B)

---

## 9. Updated Next Steps

### Immediate (This Week)

1. **Decision Point**: Path A (new recorder) or Path B (AI inference)?
   - Path A: Better quality, takes 4 weeks to build
   - Path B: Lower quality, can start immediately

2. **If Path A** (Recommended):
   ```bash
   # Start building recorder
   mkdir omega-recorder
   cd omega-recorder
   python -m venv venv
   pip install pynput mss opencv-python pyautogui
   
   # Create prototype
   python recorder.py --task "test recording"
   ```

3. **If Path B** (Faster):
   ```bash
   # Start with AI inference
   python scripts/infer_coordinates_from_video.py \
       --video existing_focus_video.mp4 \
       --task "Debug Python script"
   ```

### Week 1 Deliverables

**Path A**:
- [ ] Working recorder prototype (Python)
- [ ] Sample recording with events.json
- [ ] Validation: events match video

**Path B**:
- [ ] AI coordinate inference working
- [ ] Process 10 existing Focus Videos
- [ ] Quality assessment report

---

## 10. Final Recommendation

**GO WITH PATH A** - Build the proper recorder.

**Why:**
1. **Quality Matters**: Agents trained on bad data perform badly
2. **Long-term Cost**: $22/month vs $600/month (27x cheaper!)
3. **Sustainability**: Can scale to 100K+ trajectories affordably
4. **Competitive Advantage**: Precise coordinates = better training data

**Timeline**: 
- 4 weeks to build recorder
- 2 weeks to deploy and test
- 2 weeks to collect initial 1000 recordings
- **8 weeks total to first quality dataset**

This is longer than my initial plan, but it's the RIGHT approach. Building on shaky foundations (inferred coordinates) will lead to low-quality data that won't be useful for training agents.

---

## 11. Conclusion

**My first plan was wrong.** I underestimated the importance of precise event capture.

**The truth:**
- Cannot extract precise coordinates from video alone
- Need real-time mouse/keyboard logging
- Must build a new recording client
- Takes longer but produces real value

**The good news:**
- Event capture is a solved problem (pynput, mss)
- Can build cross-platform recorder in 2-4 weeks
- Processing costs are low (~$0.02 per recording)
- Results in dataset 10x better than AI-inferred coordinates

**Start building the recorder this week.** It's the foundation for everything else.

