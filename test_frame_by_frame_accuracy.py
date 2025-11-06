"""
Visual accuracy test for frame-by-frame cursor extraction.
Creates annotated images showing where Gemini thinks the cursor is.
"""

import json
import cv2
import os
from pathlib import Path


def draw_cursor_crosshair(frame, x, y, step_num, element=None):
    """Draw a crosshair at the detected cursor position."""
    color = (0, 255, 0)  # Green

    # Draw crosshair
    cv2.line(frame, (max(0, x - 30), y), (min(frame.shape[1], x + 30), y), color, 3)
    cv2.line(frame, (x, max(0, y - 30)), (x, min(frame.shape[0], y + 30)), color, 3)
    cv2.circle(frame, (x, y), 5, color, -1)
    cv2.circle(frame, (x, y), 20, color, 2)

    # Add label
    label = f"Step {step_num}: [{x}, {y}]"
    cv2.putText(frame, label, (x, max(30, y - 35)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if element:
        cv2.putText(frame, element, (x, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    """Test frame-by-frame accuracy."""

    # Load results
    with open("trajectory_frame_by_frame_result.json") as f:
        trajectory = json.load(f)

    # Setup output directory
    output_dir = "frame_by_frame_accuracy_test"
    os.makedirs(output_dir, exist_ok=True)

    # Video path
    video_path = "videos/Focus Video.mp4"
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Frame-by-Frame Accuracy Test")
    print(f"{'='*60}\n")
    print(f"Total steps: {len(trajectory)}")
    print(f"Video dimensions: {width}x{height}\n")

    # Process each step
    for i, step in enumerate(trajectory, 1):
        timestamp = step['timestamp_seconds']
        coords = step['cursor_coordinates']
        element = step.get('ui_element')

        # Extract frame
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()

        if not ret:
            print(f"❌ Step {i}: Failed to extract frame at {timestamp}s")
            continue

        # Convert normalized coordinates to pixels
        norm_x, norm_y = coords
        pixel_x = int(norm_x * width)
        pixel_y = int(norm_y * height)

        # Draw crosshair
        annotated = frame.copy()
        draw_cursor_crosshair(annotated, pixel_x, pixel_y, i, element)

        # Save
        output_path = os.path.join(output_dir, f"step_{i:03d}_{timestamp:.1f}s.jpg")
        cv2.imwrite(output_path, annotated)

        print(f"✓ Step {i} @ {timestamp:.1f}s: [{norm_x:.3f}, {norm_y:.3f}] → ({pixel_x}, {pixel_y})")
        print(f"  Element: {element or 'N/A'}")

    cap.release()

    print(f"\n{'='*60}")
    print(f"✅ Saved {len(trajectory)} annotated frames to {output_dir}/")
    print(f"{'='*60}\n")

    # Compare with full video results
    if Path("trajectory_extraction_result.json").exists():
        with open("trajectory_extraction_result.json") as f:
            full_video_data = json.load(f)
            full_video_steps = full_video_data.get("completion_steps_breakdown", [])

        print(f"COMPARISON:")
        print(f"  Full Video: {len(full_video_steps)} steps from entire video")
        print(f"  Frame-by-Frame: {len(trajectory)} steps from 10 sampled frames")
        print(f"  Efficiency: {len(trajectory)/10:.1f} steps per frame")


if __name__ == "__main__":
    main()
