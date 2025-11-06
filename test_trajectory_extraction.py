"""
Test script to extract trajectory data with mouse clicks and coordinates from Focus videos.

This script demonstrates the new trajectory extraction capability using Gemini 2.5 Flash
to analyze screen recordings and extract precise mouse click coordinates, actions, and
reasoning for training computer use agents.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
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

# Add parent directory to path to import from validator_api
sys.path.insert(0, str(Path(__file__).parent))

from validator_api.validator_api.database.models.scoring import (
    DetailedVideoDescription,
)
from validator_api.validator_api.scoring import focus_scoring_prompts
from validator_api.validator_api.config import (
    GOOGLE_LOCATION,
    GOOGLE_CLOUD_BUCKET_NAME,
)

# Handle both GOOGLE_PROJECT_ID and GOOGLE_CLOUD_PROJECT
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


async def upload_video_to_gcs(local_video_path: str, gcs_bucket: str, gcs_path: str) -> str:
    """
    Upload a local video to Google Cloud Storage.

    Args:
        local_video_path: Path to local video file
        gcs_bucket: GCS bucket name
        gcs_path: Destination path in GCS (e.g., 'clips/test_video.mp4')

    Returns:
        GCS URI (gs://bucket/path)
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob = bucket.blob(gcs_path)

    print(f"Uploading {local_video_path} to gs://{gcs_bucket}/{gcs_path}...")
    blob.upload_from_filename(local_video_path)
    print(f"Upload complete!")

    return f"gs://{gcs_bucket}/{gcs_path}"


async def extract_trajectory_from_video(
    video_uri: str,
    task_overview: str = "Complete a task on the computer"
) -> DetailedVideoDescription:
    """
    Extract detailed trajectory with mouse clicks and coordinates from a video.

    Args:
        video_uri: GCS URI (gs://bucket/path) or local file path
        task_overview: Description of the task being performed

    Returns:
        DetailedVideoDescription with trajectory data including coordinates
    """
    # Initialize Vertex AI using config
    if not GOOGLE_PROJECT_ID:
        raise ValueError("GOOGLE_PROJECT_ID not set in .env file")

    vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)

    # Configure Gemini model
    model_name = "gemini-2.5-pro"  # Using Pro for better accuracy
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # Convert Pydantic schema to Vertex AI compatible format
    schema = DetailedVideoDescription.model_json_schema()

    def clean_schema(obj):
        """Remove unsupported fields and inline $refs for Vertex AI."""
        if isinstance(obj, dict):
            # Handle $ref by inlining definitions
            if "$ref" in obj and "$defs" in schema:
                ref_path = obj["$ref"].split("/")[-1]
                if ref_path in schema["$defs"]:
                    return clean_schema(schema["$defs"][ref_path])

            # Remove unsupported fields
            cleaned = {}
            for k, v in obj.items():
                if k in ["$defs", "title", "anyOf", "allOf", "oneOf"]:
                    # Handle anyOf (used for Optional fields) - take the non-null type
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
        system_instruction=focus_scoring_prompts.DETAILED_DESCRIPTION_SYSTEM_PROMPT.strip(),
        safety_settings=safety_settings,
        generation_config=GenerationConfig(
            temperature=1,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    # Prepare video input
    parts = []
    if video_uri.startswith("gs://"):
        # GCS URI
        print(f"Using GCS video: {video_uri}")
        parts.append(Part.from_uri(video_uri, mime_type="video/mp4"))
    else:
        # Local file - need to upload to GCS first
        print(f"Local video file detected. Need to upload to GCS first.")
        if not GOOGLE_CLOUD_BUCKET_NAME:
            raise ValueError("GOOGLE_CLOUD_BUCKET_NAME not set in .env file")

        # Generate unique path
        import uuid
        video_filename = Path(video_uri).name
        gcs_path = f"test_clips/{uuid.uuid4()}_{video_filename}"
        video_uri = await upload_video_to_gcs(video_uri, GOOGLE_CLOUD_BUCKET_NAME, gcs_path)
        parts.append(Part.from_uri(video_uri, mime_type="video/mp4"))

    # Add user prompt
    user_prompt = focus_scoring_prompts.DETAILED_DESCRIPTION_USER_PROMPT.format(
        task_overview=task_overview
    )
    parts.append(user_prompt.strip())

    # Make request
    print(f"\n{'='*60}")
    print("Extracting trajectory from video with Gemini 2.5 Flash...")
    print(f"{'='*60}\n")

    response = await model.generate_content_async(parts)
    result = DetailedVideoDescription(**json.loads(response.text))

    return result


def print_trajectory_results(description: DetailedVideoDescription):
    """Pretty print the extracted trajectory data."""
    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}\n")

    print(f"📝 Description: {description.description}\n")

    print(f"🛠️  Applications Used:")
    for app in description.applications_used:
        print(f"   - {app}")
    print()

    print(f"📋 Completion Steps (Text):")
    for i, step in enumerate(description.completion_sequence_steps, 1):
        print(f"   {i}. {step}")
    print()

    if description.completion_steps_breakdown:
        print(f"🎯 TRAJECTORY BREAKDOWN (with coordinates):")
        print(f"   Total Steps: {len(description.completion_steps_breakdown)}\n")

        for i, traj_step in enumerate(description.completion_steps_breakdown, 1):
            print(f"   Step {i} @ {traj_step.timestamp_seconds:.1f}s:")
            print(f"   ├─ Action: {traj_step.action.type.upper()}")

            if traj_step.action.coordinates:
                x, y = traj_step.action.coordinates
                print(f"   ├─ Coordinates: [{x:.3f}, {y:.3f}] (normalized)")
                print(f"   │  (≈ {x*100:.1f}% from left, {y*100:.1f}% from top)")
            else:
                print(f"   ├─ Coordinates: None")

            if traj_step.action.element:
                print(f"   ├─ Element: {traj_step.action.element}")

            if traj_step.action.text:
                print(f"   ├─ Text: \"{traj_step.action.text}\"")

            if traj_step.action.url:
                print(f"   ├─ URL: {traj_step.action.url}")

            print(f"   ├─ Thought: {traj_step.thought}")
            print(f"   └─ Observation: {traj_step.observation}")
            print()
    else:
        print("⚠️  No trajectory breakdown extracted (completion_steps_breakdown is empty)")

    print(f"💬 User Feedback: {description.user_feedback}\n")
    print(f"{'='*60}\n")


async def main():
    """Test trajectory extraction on Focus Video.mp4"""

    # Check if video file exists
    video_path = "videos/Focus Video.mp4"
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found at {video_path}")
        print("   Please ensure the video exists in the videos/ folder")
        return

    # Task overview (you can customize this based on what the video shows)
    task_overview = """
# Task: Complete a focused work task on the computer

Please record yourself completing a productive task on your computer.
This could be coding, research, writing, or any meaningful computer-based work.
"""

    print(f"🎬 Testing trajectory extraction on: {video_path}")
    print(f"📋 Task: {task_overview.strip()}\n")

    try:
        # Extract trajectory
        description = await extract_trajectory_from_video(video_path, task_overview)

        # Print results
        print_trajectory_results(description)

        # Save to JSON file
        output_file = "trajectory_extraction_result.json"
        with open(output_file, 'w') as f:
            json.dump(description.model_dump(), f, indent=2)

        print(f"✅ Results saved to {output_file}")

        # Show AgentNet-style format conversion
        print(f"\n{'='*60}")
        print("AGENTNET-STYLE FORMAT PREVIEW")
        print(f"{'='*60}\n")

        if description.completion_steps_breakdown:
            agentnet_trajectory = []
            for step in description.completion_steps_breakdown:
                agentnet_step = {
                    "timestamp": f"{step.timestamp_seconds:.1f}s",
                    "action": {
                        "type": step.action.type,
                        "coordinates": step.action.coordinates,
                        "element": step.action.element,
                        "text": step.action.text,
                    },
                    "thought": step.thought,
                    "observation": step.observation,
                }
                # Generate PyAutoGUI-style code
                if step.action.coordinates:
                    x, y = step.action.coordinates
                    if step.action.type == "click":
                        agentnet_step["code"] = f"pyautogui.click(x={x:.3f}, y={y:.3f})"
                    elif step.action.type == "double_click":
                        agentnet_step["code"] = f"pyautogui.doubleClick(x={x:.3f}, y={y:.3f})"
                    elif step.action.type == "right_click":
                        agentnet_step["code"] = f"pyautogui.rightClick(x={x:.3f}, y={y:.3f})"
                elif step.action.type == "type" and step.action.text:
                    agentnet_step["code"] = f"pyautogui.write('{step.action.text}')"

                agentnet_trajectory.append(agentnet_step)

            print(json.dumps(agentnet_trajectory[:3], indent=2))
            print(f"\n... ({len(agentnet_trajectory)} total steps)")

    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
