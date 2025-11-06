# Ground Truth Accuracy Test Results

## Test Setup

Testing frame-by-frame cursor extraction against screenshots with known click coordinates embedded in filenames.

**Critical Finding**: These screenshots do **NOT** contain visible cursor pointers - they are UI captures without the cursor arrow visible.

## Detailed Results Table

| Screenshot | Ground Truth (px) | AI Detected (px) | Error (px) | X Error (px) | Y Error (px) | Status |
|-----------|-------------------|------------------|------------|--------------|--------------|---------|
| click_1 | (774.3, 606.9) | (159.4, 794.9) | **643.0** | 614.9 | 188.0 | ✓ Detected |
| click_2 | (693.6, 334.0) | (1777.9, 109.1) | **1107.4** | 1084.3 | 224.9 | ✓ Detected |
| click_3 | (89.4, 931.8) | (1664.6, 106.9) | **1778.1** | 1575.2 | 824.9 | ✓ Detected |
| click_4 | (709.5, 675.2) | N/A | N/A | N/A | N/A | ✗ Not Detected |
| click_5 | (709.5, 675.2) | (668.2, 704.2) | **50.5** | 41.4 | 29.0 | ✓ Detected |
| click_6 | (577.7, 771.2) | N/A | N/A | N/A | N/A | ✗ Not Detected |

## Accuracy Statistics

| Metric | Value |
|--------|-------|
| **Total Screenshots** | 6 |
| **Successfully Detected** | 4 (66.7%) |
| **Failed to Detect** | 2 (33.3%) |
| **Mean Error** | 894.8px |
| **Median Error** | 875.2px |
| **Min Error** | 50.5px |
| **Max Error** | 1778.1px |
| **Std Deviation** | 632.8px |

## Error Distribution

| Error Range | Count | Percentage |
|-------------|-------|------------|
| < 5px | 0 | 0.0% |
| < 10px | 0 | 0.0% |
| < 20px | 0 | 0.0% |
| < 50px | 0 | 0.0% |
| **< 100px** | **1** | **25.0%** |
| **> 600px** | **3** | **75.0%** |

## Critical Finding: No Visible Cursor in Screenshots

### Why Accuracy is Poor:

Your screenshots are **UI state captures** without the actual cursor pointer visible. Example from `click_5`:

- **What's in the image**: Dropdown menu showing "Start Task", "Use Memory", etc.
- **What's NOT in the image**: The cursor arrow pointer
- **Ground Truth**: Click coordinates (709, 675)
- **AI Behavior**: Trying to guess where cursor "should be" based on UI elements

### Comparison with Video Frames:

| Source | Cursor Visible? | AI Task | Expected Accuracy |
|--------|----------------|---------|-------------------|
| **Video Frames** | ✅ Yes (cursor arrow visible) | **Detect cursor position** | ±5-10px |
| **Your Screenshots** | ❌ No (just UI, no cursor) | **Guess from context** | ±100-1000px |

### Why Video Frame-by-Frame Works Better:

In the video frame tests (earlier), we achieved ±5-10px accuracy because:
1. The cursor arrow is **physically visible** in the frame
2. AI can use visual pattern matching to locate the arrow tip
3. Direct measurement, not inference

In your screenshots:
1. No cursor arrow visible
2. AI must **infer** click location from UI context ("they probably clicked here")
3. This is fundamentally unreliable

## Recommendations

### ✅ For Accurate Coordinate Extraction:

1. **Use video frames** where cursor is visible
   - Extract frames from screen recordings
   - Cursor arrow appears in the video
   - Frame-by-frame analysis: ±5-10px accuracy

2. **Or use event-capture recording**
   - Capture actual mouse events in real-time
   - Get exact pixel coordinates: ±1px accuracy
   - No AI inference needed

### ❌ Do NOT Use:

1. **Static screenshots without cursor**
   - AI cannot see cursor to detect it
   - Must guess from UI context
   - Accuracy: ±100-1000px (unacceptable)

### 🔍 To Verify Cursor Visibility:

Check if your screenshots actually show the cursor pointer:
```bash
# Open screenshots and look for cursor arrow
open screenshots/click_1_*.png
```

If the cursor is NOT visible as an arrow pointer in the image, then:
- AI detection will be highly inaccurate
- Need to switch to video frame extraction instead
- Or implement event-capture recorder

## Technical Explanation

### Why AI Can Detect in Video Frames:

```python
# Video Frame Analysis
Input: Frame with visible cursor arrow at [800, 600]
AI sees: Arrow-shaped pointer at specific pixel location
Task: "Find the arrow pointer" → Visual pattern matching
Output: [0.533, 0.556] (very close to actual position)
Accuracy: ±5-10px ✅
```

### Why AI Cannot Detect in Your Screenshots:

```python
# Screenshot Analysis (No Cursor Visible)
Input: UI screenshot showing dropdown menu at [700, 675]
AI sees: No cursor arrow visible
Task: "Where should the cursor be?" → Context-based guessing
Output: [0.445, 0.651] (guessing based on UI element position)
Accuracy: ±50-1000px ❌
```

## Conclusion

**Your screenshots do not contain visible cursor pointers**, which makes AI-based coordinate extraction highly unreliable (mean error: 894px).

**Solutions:**
1. ✅ Use **video frame extraction** (we already tested this - achieves ±5-10px)
2. ✅ Implement **event-capture recorder** (±1px accuracy)
3. ❌ Do NOT rely on UI screenshots without visible cursors

**Next Steps:**
- If you have video recordings of these clicks, use frame-by-frame extraction instead
- If you need ground truth coordinates, capture them during recording with event logger
- Static screenshots (without cursor) are not suitable for training data generation

---

Generated: 2025-11-04
Test Method: Gemini 2.0 Flash on static screenshots
Result: **Not suitable** - no visible cursor in images (mean error: 894px)
Recommendation: Use video frames with visible cursor instead (±5-10px accuracy)
