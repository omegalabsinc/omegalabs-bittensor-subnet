"""
Test script to extract trajectory by analyzing individual frames instead of full video.

This approach:
1. Extracts frames from video at regular intervals
2. Passes each frame to Gemini Flash individually
3. Asks only: "Where is the mouse cursor?" and "What action is happening?"
4. Should be more precise than full video analysis
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

import vertexai
from vertexai.generative_models import Part
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from pydantic import BaseModel, Field
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from validator_api.validator_api.config import (
    GOOGLE_LOCATION,
    GOOGLE_CLOUD_BUCKET_NAME,
)

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


class FrameAnalysis(BaseModel):
    """Analysis of a single frame."""
    cursor_visible: bool = Field(
        description="Whether the mouse cursor is clearly visible in this frame"
    )
    cursor_coordinates: Optional[List[float]] = Field(
        default=None,
        description="Normalized [x, y] coordinates of cursor tip if visible. Format: [0.0-1.0, 0.0-1.0]"
    )
    action_type: Optional[str] = Field(
        default=None,
        description="Type of action if detectable: 'click', 'double_click', 'right_click', 'typing', or 'moving'"
    )
    ui_element: Optional[str] = Field(
        default=None,
        description="Description of the UI element under or near the cursor"
    )
    screen_context: str = Field(
        description="Brief description of what's on screen (app, page, etc.)"
    )


FRAME_ANALYSIS_PROMPT = """
Analyze this screenshot and identify the mouse cursor position.

Your task:
1. Look for the mouse cursor (arrow pointer) on screen
2. If visible, determine its EXACT position as normalized coordinates
3. Identify what UI element it's pointing to or hovering over
4. Detect if an action (click, typing) appears to be happening

**COORDINATE EXTRACTION**:
- Normalized [x, y] where 0.0 = left/top edge, 1.0 = right/bottom edge
- Focus on the TIP of the cursor arrow
- Example: [0.25, 0.50] = 25% from left, 50% from top (middle-left of screen)
- Be PRECISE - don't default to center [0.5, 0.5] unless cursor is actually there

**IF CURSOR NOT VISIBLE**: Set cursor_visible=false and cursor_coordinates=null

Respond in JSON format following the FrameAnalysis schema.
"""


def extract_frames_from_video(
    video_path: str,
    sample_rate: float = 1.0,
    max_frames: int = 100
) -> List[tuple[float, np.ndarray]]:
    """
    Extract frames from video at specified sample rate.

    Args:
        video_path: Path to video file
        sample_rate: Extract one frame per N seconds
        max_frames: Maximum number of frames to extract

    Returns:
        List of (timestamp, frame) tuples
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video info: {duration:.1f}s, {fps:.1f} fps, {total_frames} frames")

    frames = []
    frame_interval = int(fps * sample_rate)

    frame_num = 0
    while frame_num < total_frames and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_num / fps
        frames.append((timestamp, frame))

        frame_num += frame_interval

    cap.release()
    print(f"Extracted {len(frames)} frames (1 per {sample_rate}s)")
    return frames


async def analyze_frame(
    frame: np.ndarray,
    model: GenerativeModel
) -> FrameAnalysis:
    """
    Analyze a single frame to detect cursor position and action.

    Args:
        frame: OpenCV frame (numpy array)
        model: Gemini model instance

    Returns:
        FrameAnalysis with cursor coordinates if detected
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encode as JPEG
    success, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("Failed to encode frame")

    # Create image part from bytes
    image_bytes = buffer.tobytes()
    image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")

    # Generate analysis
    response = await model.generate_content_async([
        image_part,
        FRAME_ANALYSIS_PROMPT
    ])

    result = FrameAnalysis(**json.loads(response.text))
    return result


async def extract_trajectory_frame_by_frame(
    video_path: str,
    sample_rate: float = 1.0,
    max_frames: int = 100
) -> List[Dict[str, Any]]:
    """
    Extract trajectory by analyzing individual frames.

    Args:
        video_path: Path to video file
        sample_rate: Sample one frame per N seconds
        max_frames: Maximum frames to analyze

    Returns:
        List of trajectory steps with timestamps and coordinates
    """
    if not GOOGLE_PROJECT_ID:
        raise ValueError("GOOGLE_PROJECT_ID not set in .env file")

    vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)

    # Configure Gemini Flash
    model_name = "gemini-2.0-flash-exp"
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # Convert schema
    schema = FrameAnalysis.model_json_schema()

    def clean_schema(obj):
        """Remove unsupported fields for Vertex AI."""
        if isinstance(obj, dict):
            if "$ref" in obj and "$defs" in schema:
                ref_path = obj["$ref"].split("/")[-1]
                if ref_path in schema["$defs"]:
                    return clean_schema(schema["$defs"][ref_path])

            cleaned = {}
            for k, v in obj.items():
                if k in ["$defs", "title", "anyOf", "allOf", "oneOf"]:
                    if k == "anyOf":
                        for option in v:
                            if option.get("type") != "null":
                                return clean_schema(option)
                    continue
                cleaned[k] = clean_schema(v)
            return cleaned
        elif isinstance(obj, list):
            return [clean_schema(item) for item in obj]
        return obj

    schema = clean_schema(schema)

    model = GenerativeModel(
        model_name,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(
            temperature=0.2,  # Lower temp for more consistent coordinate extraction
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    # Extract frames from video
    print(f"\n{'='*60}")
    print("FRAME-BY-FRAME EXTRACTION")
    print(f"{'='*60}\n")

    frames = extract_frames_from_video(video_path, sample_rate, max_frames)

    # Analyze each frame
    trajectory = []

    for i, (timestamp, frame) in enumerate(frames, 1):
        print(f"Analyzing frame {i}/{len(frames)} @ {timestamp:.1f}s...", end=" ")

        try:
            analysis = await analyze_frame(frame, model)

            if analysis.cursor_visible and analysis.cursor_coordinates:
                trajectory.append({
                    "timestamp_seconds": timestamp,
                    "frame_number": i,
                    "cursor_coordinates": analysis.cursor_coordinates,
                    "action_type": analysis.action_type or "cursor_visible",
                    "ui_element": analysis.ui_element,
                    "screen_context": analysis.screen_context
                })
                x, y = analysis.cursor_coordinates
                print(f"✓ Cursor at [{x:.3f}, {y:.3f}] - {analysis.ui_element or 'N/A'}")
            else:
                print("⊘ No cursor")

        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"Extracted {len(trajectory)} steps with cursor coordinates")
    print(f"{'='*60}\n")

    return trajectory


def print_trajectory_results(trajectory: List[Dict[str, Any]]):
    """Pretty print trajectory results."""
    print(f"🎯 TRAJECTORY RESULTS\n")
    print(f"Total steps with cursor: {len(trajectory)}\n")

    for i, step in enumerate(trajectory, 1):
        x, y = step['cursor_coordinates']
        print(f"Step {i} @ {step['timestamp_seconds']:.1f}s:")
        print(f"  └─ Cursor: [{x:.3f}, {y:.3f}] ({x*100:.1f}%, {y*100:.1f}%)")
        print(f"  └─ Element: {step['ui_element'] or 'N/A'}")
        print(f"  └─ Context: {step['screen_context']}")
        print()


async def main():
    """Test frame-by-frame extraction on Focus Video.mp4"""

    video_path = "videos/Focus Video.mp4"
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found at {video_path}")
        return

    print(f"🎬 Testing FRAME-BY-FRAME extraction on: {video_path}")
    print(f"📋 Method: Extract individual frames → Analyze each with Gemini Flash\n")

    try:
        # Extract trajectory (1 frame per second, max 10 frames for testing)
        trajectory = await extract_trajectory_frame_by_frame(
            video_path,
            sample_rate=5.0,  # One frame every 5 seconds
            max_frames=10
        )

        # Print results
        print_trajectory_results(trajectory)

        # Save to JSON
        output_file = "trajectory_frame_by_frame_result.json"
        with open(output_file, 'w') as f:
            json.dump(trajectory, f, indent=2)

        print(f"✅ Results saved to {output_file}")

        # Compare with full video approach
        if Path("trajectory_extraction_result.json").exists():
            with open("trajectory_extraction_result.json") as f:
                full_video_result = json.load(f)
                full_video_steps = len(full_video_result.get("completion_steps_breakdown", []))

            print(f"\n{'='*60}")
            print("COMPARISON")
            print(f"{'='*60}")
            print(f"Full Video approach: {full_video_steps} steps")
            print(f"Frame-by-Frame approach: {len(trajectory)} steps")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
