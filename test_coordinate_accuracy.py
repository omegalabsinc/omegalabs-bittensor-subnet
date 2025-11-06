"""
Test coordinate accuracy by visualizing extracted trajectories on video frames.

This script:
1. Extracts frames from the video at trajectory timestamps
2. Overlays click coordinates with crosshairs and markers
3. Creates annotated frames for visual inspection
4. Generates an accuracy assessment report
5. Optionally crops video sections for detailed analysis
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


def load_trajectory_data(json_path: str) -> List[dict]:
    """Load trajectory data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('completion_steps_breakdown', [])


def extract_frame_at_timestamp(video_path: str, timestamp_seconds: float) -> Optional[np.ndarray]:
    """Extract a frame from video at specific timestamp."""
    cap = cv2.VideoCapture(video_path)

    # Set position to timestamp (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def draw_crosshair(frame: np.ndarray, x: int, y: int, color=(0, 255, 0), size=20, thickness=2):
    """Draw a crosshair at specified coordinates."""
    height, width = frame.shape[:2]

    # Draw horizontal line
    cv2.line(frame, (max(0, x - size), y), (min(width, x + size), y), color, thickness)
    # Draw vertical line
    cv2.line(frame, (x, max(0, y - size)), (x, min(height, y + size)), color, thickness)
    # Draw circle at center
    cv2.circle(frame, (x, y), 5, color, -1)


def draw_click_annotation(frame: np.ndarray, x: int, y: int, step_num: int,
                          action_type: str, element: str = None):
    """Draw detailed annotation for a click."""
    height, width = frame.shape[:2]

    # Choose color based on action type
    colors = {
        'click': (0, 255, 0),       # Green
        'double_click': (0, 255, 255),  # Yellow
        'right_click': (255, 0, 255),   # Magenta
        'type': (255, 255, 0),      # Cyan
        'scroll': (128, 128, 255),  # Light blue
        'drag': (255, 128, 0),      # Orange
        'other': (128, 128, 128)    # Gray
    }
    color = colors.get(action_type, (0, 255, 0))

    # Draw crosshair
    draw_crosshair(frame, x, y, color, size=30, thickness=3)

    # Draw larger circle
    cv2.circle(frame, (x, y), 20, color, 2)

    # Add step number label
    label_y = max(30, y - 35)
    label_text = f"Step {step_num}: {action_type.upper()}"

    # Draw text background
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(frame,
                  (x - 5, label_y - text_height - 5),
                  (x + text_width + 5, label_y + 5),
                  (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, label_text, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add element description if available
    if element:
        elem_y = label_y + 25
        elem_text = element[:40] + "..." if len(element) > 40 else element
        (elem_width, elem_height), _ = cv2.getTextSize(
            elem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame,
                      (x - 5, elem_y - elem_height - 5),
                      (x + elem_width + 5, elem_y + 5),
                      (0, 0, 0), -1)
        cv2.putText(frame, elem_text, (x, elem_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def visualize_trajectory_step(video_path: str, step: dict, step_num: int,
                              output_dir: str) -> Tuple[bool, str]:
    """
    Visualize a single trajectory step by extracting frame and overlaying coordinates.

    Returns:
        (success, message) tuple
    """
    timestamp = step['timestamp_seconds']
    action = step['action']
    action_type = action['type']
    coordinates = action.get('coordinates')
    element = action.get('element', 'Unknown')

    if not coordinates:
        return False, f"Step {step_num}: No coordinates (action type: {action_type})"

    # Extract frame
    frame = extract_frame_at_timestamp(video_path, timestamp)
    if frame is None:
        return False, f"Step {step_num}: Failed to extract frame at {timestamp}s"

    height, width = frame.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    norm_x, norm_y = coordinates
    pixel_x = int(norm_x * width)
    pixel_y = int(norm_y * height)

    # Create annotated frame
    annotated = frame.copy()
    draw_click_annotation(annotated, pixel_x, pixel_y, step_num, action_type, element)

    # Add timestamp and thought info
    info_text = f"Time: {timestamp:.1f}s | Thought: {step['thought'][:60]}..."
    cv2.putText(annotated, info_text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save annotated frame
    output_path = os.path.join(output_dir, f"step_{step_num:03d}_{timestamp:.1f}s.jpg")
    cv2.imwrite(output_path, annotated)

    return True, f"Step {step_num}: ({pixel_x}, {pixel_y}) at {timestamp:.1f}s - {element}"


def crop_video_section(video_path: str, start_time: float, duration: float,
                       output_path: str):
    """Crop a section of the video for detailed analysis."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set position to start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    end_time = start_time + duration
    frame_count = 0

    print(f"Cropping video from {start_time}s to {end_time}s...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > end_time:
            break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"✅ Cropped {frame_count} frames to {output_path}")


def create_annotated_video(video_path: str, trajectory_data: List[dict],
                           output_path: str, steps_to_annotate: Optional[List[int]] = None):
    """
    Create a video with trajectory annotations overlaid.

    Args:
        video_path: Path to input video
        trajectory_data: List of trajectory steps
        output_path: Path for output video
        steps_to_annotate: Optional list of step indices to annotate. If None, annotate all.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create timestamp -> step mapping
    timestamp_map = {}
    for i, step in enumerate(trajectory_data):
        if steps_to_annotate is None or i in steps_to_annotate:
            if step['action'].get('coordinates'):
                timestamp_map[step['timestamp_seconds']] = (i + 1, step)

    print(f"Creating annotated video with {len(timestamp_map)} annotations...")
    print(f"Processing {total_frames} frames...")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Check if there's an annotation near this timestamp (within 0.5 seconds)
        for timestamp, (step_num, step) in timestamp_map.items():
            if abs(current_time - timestamp) < 0.5:
                action = step['action']
                coordinates = action.get('coordinates')
                if coordinates:
                    norm_x, norm_y = coordinates
                    pixel_x = int(norm_x * width)
                    pixel_y = int(norm_y * height)

                    draw_click_annotation(frame, pixel_x, pixel_y, step_num,
                                        action['type'], action.get('element'))

        out.write(frame)
        frame_num += 1

        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")

    cap.release()
    out.release()

    print(f"✅ Annotated video saved to {output_path}")


def generate_accuracy_report(video_path: str, trajectory_data: List[dict],
                             output_dir: str, sample_size: int = 20):
    """
    Generate accuracy report by visualizing sample trajectory steps.

    Args:
        video_path: Path to video file
        trajectory_data: List of trajectory steps
        output_dir: Directory to save frames
        sample_size: Number of steps to sample for visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter steps with coordinates
    steps_with_coords = [
        (i, step) for i, step in enumerate(trajectory_data)
        if step['action'].get('coordinates')
    ]

    print(f"\n{'='*60}")
    print(f"COORDINATE ACCURACY TEST")
    print(f"{'='*60}\n")
    print(f"Total trajectory steps: {len(trajectory_data)}")
    print(f"Steps with coordinates: {len(steps_with_coords)}")
    print(f"Steps to visualize: {min(sample_size, len(steps_with_coords))}\n")

    # Sample steps evenly distributed
    if len(steps_with_coords) > sample_size:
        step_indices = np.linspace(0, len(steps_with_coords) - 1, sample_size, dtype=int)
        sampled_steps = [steps_with_coords[i] for i in step_indices]
    else:
        sampled_steps = steps_with_coords

    # Visualize each sampled step
    results = []
    for original_idx, step in sampled_steps:
        step_num = original_idx + 1
        success, message = visualize_trajectory_step(video_path, step, step_num, output_dir)
        results.append((success, message))

        status = "✅" if success else "❌"
        print(f"{status} {message}")

    # Summary
    successful = sum(1 for s, _ in results if s)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully visualized: {successful}/{len(results)} steps")
    print(f"Output directory: {output_dir}")
    print(f"\n💡 Review the images in '{output_dir}' to assess coordinate accuracy")
    print(f"   Look for: Are the crosshairs on the correct UI elements?")


def main():
    """Main function to test coordinate accuracy."""

    # Configuration
    video_path = "videos/Focus Video.mp4"
    trajectory_json = "trajectory_extraction_result.json"
    output_dir = "coordinate_accuracy_test"

    print(f"🎬 Testing Coordinate Accuracy")
    print(f"Video: {video_path}")
    print(f"Trajectory data: {trajectory_json}\n")

    # Check files exist
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return

    if not Path(trajectory_json).exists():
        print(f"❌ Error: Trajectory JSON not found: {trajectory_json}")
        print("   Please run test_trajectory_extraction.py first")
        return

    # Load trajectory data
    trajectory_data = load_trajectory_data(trajectory_json)
    if not trajectory_data:
        print(f"❌ Error: No trajectory data found in {trajectory_json}")
        return

    # Generate accuracy report with sample frames
    generate_accuracy_report(video_path, trajectory_data, output_dir, sample_size=30)

    # Optional: Create a short annotated video clip for visual verification
    print(f"\n{'='*60}")
    print("CREATING ANNOTATED VIDEO CLIP")
    print(f"{'='*60}\n")

    # Crop first 60 seconds for testing
    cropped_video = "videos/Focus_Video_60s.mp4"
    crop_video_section(video_path, start_time=0, duration=60, output_path=cropped_video)

    # Create annotated version of the cropped video
    # Only annotate first 20 steps for speed
    annotated_output = "coordinate_accuracy_test/annotated_video_60s.mp4"
    steps_to_annotate = list(range(20))  # First 20 steps
    create_annotated_video(cropped_video, trajectory_data, annotated_output, steps_to_annotate)

    print(f"\n{'='*60}")
    print("✅ ACCURACY TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 Check these outputs:")
    print(f"   1. Frame samples: {output_dir}/")
    print(f"   2. Cropped video: {cropped_video}")
    print(f"   3. Annotated video: {annotated_output}")
    print(f"\n🔍 Visual Inspection:")
    print(f"   - Open the annotated frames to see if crosshairs align with UI elements")
    print(f"   - Watch the annotated video to see clicks in context")
    print(f"   - Green crosshairs = clicks, Yellow = double-clicks, etc.")


if __name__ == "__main__":
    main()
