"""
Test frame-by-frame cursor extraction accuracy against ground truth coordinates.

Screenshots have filenames with actual cursor positions:
  click_1_20251104_183318_x774.29296875_y606.91796875.png

We'll extract coordinates with Gemini and compare against these ground truth values.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
from dotenv import load_dotenv

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
import sys

sys.path.insert(0, str(Path(__file__).parent))
from validator_api.validator_api.config import GOOGLE_LOCATION

GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


class FrameAnalysis(BaseModel):
    """Analysis of a single frame."""
    cursor_visible: bool = Field(
        description="Whether the mouse cursor is clearly visible in this frame"
    )
    cursor_coordinates: List[float] | None = Field(
        default=None,
        description="Normalized [x, y] coordinates of cursor tip if visible. Format: [0.0-1.0, 0.0-1.0]"
    )
    ui_element: str | None = Field(
        default=None,
        description="Description of the UI element under or near the cursor"
    )


FRAME_ANALYSIS_PROMPT = """
You are a precision cursor detection system. Analyze this screenshot to locate the EXACT position of the mouse cursor.

**COORDINATE SYSTEM SPECIFICATION**:
- Origin: Top-left corner of image
- X-axis: Increases from left (0.0) to right (1.0)
- Y-axis: Increases from top (0.0) to bottom (1.0)
- Format: Normalized coordinates [x, y] in range 0.0 to 1.0
- Precision: Use 3+ decimal places (e.g., 0.523, not 0.5)

**STEP-BY-STEP ANALYSIS REQUIRED**:

Step 1 - VISUAL SCAN:
Describe what you see in the image. Where do you notice the cursor? What type of cursor is it (arrow pointer, I-beam, hand)?

Step 2 - REGION IDENTIFICATION:
Identify the general region where the cursor appears:
- Horizontal: Far left (0.0-0.2), Left (0.2-0.4), Center (0.4-0.6), Right (0.6-0.8), Far right (0.8-1.0)
- Vertical: Top (0.0-0.25), Upper-middle (0.25-0.5), Lower-middle (0.5-0.75), Bottom (0.75-1.0)

Step 3 - PERCENTAGE ESTIMATION:
Estimate the cursor position as percentages:
- "The cursor appears approximately X% from the left edge"
- "The cursor appears approximately Y% from the top edge"

Step 4 - PRECISE COORDINATES:
Convert percentages to normalized coordinates:
- X coordinate = percentage / 100
- Y coordinate = percentage / 100
- Measure from the TIP of cursor arrow or center of I-beam cursor

Step 5 - VERIFICATION:
Double-check your answer:
- Does [x, y] match the region you identified in Step 2?
- Are coordinates in valid range (0.0-1.0)?
- Did you measure from cursor tip, not the UI element it's pointing at?

**EXAMPLES OF CORRECT ANALYSIS**:

Example 1 - Center-screen cursor:
"Step 1: I see a white arrow cursor in a text input field
Step 2: Horizontally centered (0.4-0.6), vertically in upper-middle (0.25-0.5)
Step 3: Approximately 52% from left, 35% from top
Step 4: [0.52, 0.35]
Step 5: ✓ Matches center region, valid range, measured from cursor tip"

Example 2 - Left sidebar cursor:
"Step 1: I see a white arrow cursor over a menu item in the left sidebar
Step 2: Far left region (0.0-0.2), vertically in upper-middle (0.25-0.5)
Step 3: Approximately 10% from left, 40% from top
Step 4: [0.10, 0.40]
Step 5: ✓ Matches far-left region, valid range, cursor clearly visible in sidebar"

Example 3 - Top-right cursor:
"Step 1: I see a hand pointer cursor over a button in the top-right corner
Step 2: Right region (0.6-0.8), top region (0.0-0.25)
Step 3: Approximately 78% from left, 12% from top
Step 4: [0.78, 0.12]
Step 5: ✓ Matches top-right region, valid range"

**CRITICAL REMINDERS**:
✓ Scan ALL screen areas including sidebars and edges
✓ Measure from cursor TIP, not the element it's pointing at
✓ Use precise decimal values (0.523 not 0.5)
✓ Verify your answer matches the visual description

**IF CURSOR NOT VISIBLE**:
Set cursor_visible=false ONLY after thoroughly scanning the entire image.

Now analyze the provided screenshot following all 5 steps above, then respond in JSON format.
"""


def parse_coordinates_from_filename(filename: str) -> Tuple[float, float] | None:
    """
    Extract ground truth coordinates from filename.

    Format: click_1_20251104_183318_x774.29296875_y606.91796875.png
    Returns: (x, y) in pixels
    """
    match = re.search(r'x([\d.]+)_y([\d.]+)\.png', filename)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return (x, y)
    return None


async def analyze_screenshot(
    image_path: str,
    model: GenerativeModel
) -> FrameAnalysis:
    """Analyze a single screenshot to detect cursor position."""

    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Create image part
    image_part = Part.from_data(data=image_bytes, mime_type="image/png")

    # Generate analysis
    response = await model.generate_content_async([
        image_part,
        FRAME_ANALYSIS_PROMPT
    ])

    result = FrameAnalysis(**json.loads(response.text))
    return result


def calculate_error(
    ground_truth: Tuple[float, float],
    detected: Tuple[float, float],
    image_width: int,
    image_height: int
) -> Dict[str, float]:
    """
    Calculate error metrics between ground truth and detected coordinates.

    Returns dict with:
    - pixel_error: Euclidean distance in pixels
    - x_error: Horizontal error in pixels
    - y_error: Vertical error in pixels
    - x_error_pct: Horizontal error as % of width
    - y_error_pct: Vertical error as % of height
    """
    gt_x, gt_y = ground_truth
    det_x, det_y = detected

    x_error = abs(det_x - gt_x)
    y_error = abs(det_y - gt_y)
    pixel_error = np.sqrt(x_error**2 + y_error**2)

    return {
        'pixel_error': pixel_error,
        'x_error': x_error,
        'y_error': y_error,
        'x_error_pct': (x_error / image_width) * 100,
        'y_error_pct': (y_error / image_height) * 100
    }


def draw_comparison(
    image_path: str,
    ground_truth: Tuple[float, float],
    detected: Tuple[float, float],
    error: Dict[str, float],
    output_path: str
):
    """
    Draw comparison visualization showing:
    - Ground truth (RED crosshair)
    - Detected (GREEN crosshair)
    - Error line connecting them
    - Error metrics
    """
    img = cv2.imread(image_path)
    gt_x, gt_y = int(ground_truth[0]), int(ground_truth[1])
    det_x, det_y = int(detected[0]), int(detected[1])

    # Draw ground truth (RED)
    cv2.line(img, (gt_x - 30, gt_y), (gt_x + 30, gt_y), (0, 0, 255), 3)
    cv2.line(img, (gt_x, gt_y - 30), (gt_x, gt_y + 30), (0, 0, 255), 3)
    cv2.circle(img, (gt_x, gt_y), 10, (0, 0, 255), 2)
    cv2.putText(img, f"GT: ({gt_x}, {gt_y})", (gt_x - 60, gt_y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw detected (GREEN)
    cv2.line(img, (det_x - 30, det_y), (det_x + 30, det_y), (0, 255, 0), 3)
    cv2.line(img, (det_x, det_y - 30), (det_x, det_y + 30), (0, 255, 0), 3)
    cv2.circle(img, (det_x, det_y), 10, (0, 255, 0), 2)
    cv2.putText(img, f"AI: ({det_x}, {det_y})", (det_x - 60, det_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw error line
    cv2.line(img, (gt_x, gt_y), (det_x, det_y), (0, 255, 255), 2)

    # Draw error metrics
    error_text = f"Error: {error['pixel_error']:.1f}px"
    cv2.putText(img, error_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, error_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

    cv2.imwrite(output_path, img)


async def test_ground_truth_accuracy():
    """Test cursor extraction accuracy against ground truth screenshots."""

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
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    # Get all screenshots
    screenshots_dir = Path("screenshots")
    screenshots = sorted(screenshots_dir.glob("*.png"))

    print(f"\n{'='*70}")
    print("GROUND TRUTH ACCURACY TEST")
    print(f"{'='*70}\n")
    print(f"Testing {len(screenshots)} screenshots with known coordinates\n")

    # Setup output directory
    output_dir = Path("ground_truth_accuracy_test")
    output_dir.mkdir(exist_ok=True)

    results = []
    errors = []

    # Process each screenshot
    for i, screenshot_path in enumerate(screenshots, 1):
        filename = screenshot_path.name
        print(f"\n[{i}/{len(screenshots)}] {filename}")

        # Parse ground truth coordinates
        ground_truth = parse_coordinates_from_filename(filename)
        if not ground_truth:
            print(f"  ⚠️  Could not parse coordinates from filename")
            continue

        gt_x, gt_y = ground_truth

        # Get image dimensions
        img = cv2.imread(str(screenshot_path))
        height, width = img.shape[:2]

        # Analyze with Gemini
        try:
            analysis = await analyze_screenshot(str(screenshot_path), model)

            if analysis.cursor_visible and analysis.cursor_coordinates:
                # Convert normalized to pixels
                norm_x, norm_y = analysis.cursor_coordinates
                det_x = norm_x * width
                det_y = norm_y * height
                detected = (det_x, det_y)

                # Calculate error
                error = calculate_error(ground_truth, detected, width, height)
                errors.append(error['pixel_error'])

                # Print results
                print(f"  Ground Truth: ({gt_x:.1f}, {gt_y:.1f})")
                print(f"  AI Detected:  ({det_x:.1f}, {det_y:.1f})")
                print(f"  ✓ Error: {error['pixel_error']:.1f}px "
                      f"(X: {error['x_error']:.1f}px, Y: {error['y_error']:.1f}px)")

                # Draw comparison
                output_path = output_dir / f"comparison_{filename}"
                draw_comparison(str(screenshot_path), ground_truth, detected,
                              error, str(output_path))

                results.append({
                    'filename': filename,
                    'ground_truth': ground_truth,
                    'detected': detected,
                    'error': error,
                    'ui_element': analysis.ui_element
                })
            else:
                print(f"  ✗ Cursor not detected by AI")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Calculate statistics
    if errors:
        print(f"\n{'='*70}")
        print("ACCURACY STATISTICS")
        print(f"{'='*70}\n")
        print(f"Total Screenshots: {len(screenshots)}")
        print(f"Successfully Analyzed: {len(errors)}")
        print(f"Detection Rate: {len(errors)/len(screenshots)*100:.1f}%\n")

        print(f"Pixel Error Statistics:")
        print(f"  Mean Error:   {np.mean(errors):.1f}px")
        print(f"  Median Error: {np.median(errors):.1f}px")
        print(f"  Std Dev:      {np.std(errors):.1f}px")
        print(f"  Min Error:    {np.min(errors):.1f}px")
        print(f"  Max Error:    {np.max(errors):.1f}px")
        print(f"  95th %ile:    {np.percentile(errors, 95):.1f}px\n")

        # Accuracy buckets
        under_5px = sum(1 for e in errors if e < 5)
        under_10px = sum(1 for e in errors if e < 10)
        under_20px = sum(1 for e in errors if e < 20)
        under_50px = sum(1 for e in errors if e < 50)

        print(f"Accuracy Distribution:")
        print(f"  < 5px:   {under_5px}/{len(errors)} ({under_5px/len(errors)*100:.1f}%)")
        print(f"  < 10px:  {under_10px}/{len(errors)} ({under_10px/len(errors)*100:.1f}%)")
        print(f"  < 20px:  {under_20px}/{len(errors)} ({under_20px/len(errors)*100:.1f}%)")
        print(f"  < 50px:  {under_50px}/{len(errors)} ({under_50px/len(errors)*100:.1f}%)")

        print(f"\n{'='*70}")
        print(f"✅ Results saved to {output_dir}/")
        print(f"{'='*70}\n")

        # Save detailed results
        with open(output_dir / "accuracy_results.json", 'w') as f:
            json.dump({
                'statistics': {
                    'mean_error': float(np.mean(errors)),
                    'median_error': float(np.median(errors)),
                    'std_dev': float(np.std(errors)),
                    'min_error': float(np.min(errors)),
                    'max_error': float(np.max(errors)),
                    'p95_error': float(np.percentile(errors, 95))
                },
                'results': results
            }, f, indent=2)


async def main():
    await test_ground_truth_accuracy()


if __name__ == "__main__":
    asyncio.run(main())
