# Coordinate Accuracy Test Report

## Overview
This report shows the accuracy of AI-extracted mouse click coordinates from video analysis using Gemini 2.5 Flash.

## Test Configuration
- **Video**: Focus Video.mp4
- **Total Trajectory Steps**: 113
- **Steps with Coordinates**: 113 (100%)
- **Frames Extracted**: 19 (first 60 seconds)
- **Method**: Gemini 2.5 Flash visual analysis with structured output

## Sample Coordinate Extractions

### Step 1 @ 1.1s - Click on Task Item
- **Coordinates**: (1789, 311) - [0.932, 0.288] normalized
- **Element**: "Record demo video of upload feature task item"
- **Action**: click
- **File**: `step_001_1.1s.jpg`

### Step 4 @ 13.5s - Select Task Session
- **Coordinates**: (741, 279) - [0.386, 0.259] normalized
- **Element**: "Record demo video of upload feature task session entry"
- **Action**: click
- **File**: `step_004_13.5s.jpg`

### Step 8 @ 50.7s - Click Cancel Button
- **Coordinates**: (850, 762) - [0.443, 0.706] normalized
- **Element**: "Cancel button"
- **Action**: click
- **File**: `step_008_50.7s.jpg`

### Step 55 @ 416.4s - Password Input
- **Coordinates**: (960, 545) - [0.500, 0.505] normalized
- **Element**: "Password input field"
- **Action**: type
- **File**: `step_055_416.4s.jpg`

### Step 62 @ 436.4s - CAPTCHA Retry
- **Coordinates**: (958, 832) - [0.499, 0.771] normalized
- **Element**: "Try again button"
- **Action**: click
- **File**: `step_062_436.4s.jpg`

## Visualization Features

Each annotated frame includes:
- ✅ **Green crosshair** marking the exact click location
- ✅ **Circle** highlighting the clickable area
- ✅ **Step number** and action type label
- ✅ **Element description** below the crosshair
- ✅ **Timestamp and thought** at the bottom

## How to Assess Accuracy

1. **Open the annotated frame images** in `coordinate_accuracy_test/`
2. **Check if the green crosshair** is centered on the UI element mentioned
3. **Watch the annotated video** (`annotated_video_60s.mp4`) to see clicks in context
4. **Look for**:
   - ✅ Crosshair on correct button/link/input
   - ✅ Reasonable proximity to clickable element
   - ⚠️ Coordinates off by more than 50 pixels
   - ❌ Completely wrong element

## Color Coding
- 🟢 **Green**: Regular click
- 🟡 **Yellow**: Double-click
- 🟣 **Magenta**: Right-click
- 🔵 **Cyan**: Type action
- 🟠 **Orange**: Drag action

## Accuracy Expectations

Based on AI visual analysis:
- **High accuracy** (±10-20px): Simple, centered UI elements
- **Medium accuracy** (±20-50px): Small buttons, edge elements
- **Lower accuracy** (±50-100px): Complex layouts, overlapping elements
- **Not extracted**: Mouse movements without clicks

## Files Generated

1. **Frame samples**: 19 annotated images showing specific click moments
2. **Cropped video**: First 60 seconds of original video
3. **Annotated video**: Cropped video with overlaid click markers
4. **This report**: Accuracy assessment documentation

## Next Steps

1. **Manual Review**: Open a few sample images to visually verify accuracy
2. **Watch Video**: Play `annotated_video_60s.mp4` to see clicks in action
3. **Iterate if Needed**: If accuracy is too low, consider:
   - Using event-capture recorder (pynput) for ±1px accuracy
   - Increasing Gemini prompt specificity
   - Post-processing coordinate refinement

## Conclusion

This test provides a visual way to assess whether AI-extracted coordinates are accurate enough for your use case. For training computer use agents, coordinates should generally be within ±20-50 pixels of the actual click location.

---
Generated: 2025-11-04
