# Ground Truth Accuracy Test - Detailed Results

## Test Configuration
- **Model**: Gemini 2.0 Flash Experimental
- **Method**: Frame-by-frame single image analysis
- **Total Screenshots**: 7 (all with visible cursors)
- **Detection Rate**: 100% (7/7 detected)

## Detailed Comparison Table

| # | Screenshot | Ground Truth (px) | AI Detected (px) | Total Error | X Error | Y Error | Accuracy |
|---|-----------|-------------------|------------------|-------------|---------|---------|----------|
| 1 | click_11 | (756, 605) | (778, 592) | **25.4px** | 21.9px | 12.9px | ✅ Good |
| 2 | click_12 | (1162, 98) | (1200, 157) | **70.0px** | 38.4px | 58.5px | ⚠️ Fair |
| 3 | click_13 | (1274, 176) | (1254, 204) | **34.4px** | 20.1px | 27.9px | ✅ Good |
| 4 | click_14 | (41, 117) | (1139, 118) | **1097.6px** | 1097.6px | 0.7px | ❌ Poor |
| 5 | click_15 | (41, 110) | (96, 76) | **65.0px** | 55.4px | 34.0px | ⚠️ Fair |
| 6 | click_16 | (305, 110) | (1014, 109) | **708.7px** | 708.7px | 1.2px | ❌ Poor |
| 7 | click_17 | (161, 228) | (326, 389) | **230.6px** | 165.6px | 160.5px | ❌ Poor |

## Summary Statistics

### Overall Accuracy
| Metric | Value |
|--------|-------|
| **Mean Error** | 318.8px |
| **Median Error** | 70.0px |
| **Std Deviation** | 389.0px |
| **Min Error** | 25.4px |
| **Max Error** | 1097.6px |
| **95th Percentile** | 980.9px |

### Error Distribution
| Error Range | Count | Percentage |
|-------------|-------|------------|
| **< 50px** (Excellent) | 2 | 28.6% |
| **50-100px** (Good) | 2 | 28.6% |
| **100-300px** (Fair) | 1 | 14.3% |
| **> 300px** (Poor) | 2 | 28.6% |

### By Accuracy Quality
| Quality | Count | Percentage | Definition |
|---------|-------|------------|------------|
| ✅ **Good** (< 50px) | 2 | 28.6% | Acceptable for training |
| ⚠️ **Fair** (50-100px) | 2 | 28.6% | Marginal accuracy |
| ❌ **Poor** (> 100px) | 3 | 42.8% | Unacceptable |

## Analysis by Screenshot

### Good Accuracy (< 50px)

**Click 11: 25.4px error**
- Ground Truth: (756, 605)
- AI Detected: (778, 592)
- Context: Text input field with visible cursor
- Why good: Cursor clearly visible, AI located it accurately

**Click 13: 34.4px error**
- Ground Truth: (1274, 176)
- AI Detected: (1254, 204)
- Context: UI element with cursor
- Why good: Clear cursor position, reasonable detection

### Fair Accuracy (50-100px)

**Click 12: 70.0px error**
- Ground Truth: (1162, 98)
- AI Detected: (1200, 157)
- Context: Top area of screen
- Analysis: Off by ~40px horizontally, ~60px vertically

**Click 15: 65.0px error**
- Ground Truth: (41, 110)
- AI Detected: (96, 76)
- Context: Left edge of screen
- Analysis: Off by ~55px horizontally, ~34px vertically

### Poor Accuracy (> 100px)

**Click 14: 1097.6px error** ⚠️ **OUTLIER**
- Ground Truth: (41, 117)
- AI Detected: (1139, 118)
- Context: Far left edge
- Analysis: Y-coordinate perfect (0.7px error), but X completely wrong
- Likely issue: Cursor at screen edge, AI may have confused it

**Click 16: 708.7px error** ⚠️ **OUTLIER**
- Ground Truth: (305, 110)
- AI Detected: (1014, 109)
- Context: Left-center area
- Analysis: Y-coordinate perfect (1.2px error), but X very wrong
- Pattern: Similar to click_14 - X coordinate detection issue

**Click 17: 230.6px error**
- Ground Truth: (161, 228)
- AI Detected: (326, 389)
- Context: Left side, mid-height
- Analysis: Both X and Y significantly off

## Key Findings

### ✅ Strengths:
1. **100% detection rate** - AI found cursor in all 7 images
2. **2 excellent results** (< 50px) - 25.4px and 34.4px errors
3. **Median error: 70px** - When working well, accuracy is reasonable
4. **Y-coordinate accuracy** - Often very precise on vertical positioning

### ❌ Weaknesses:
1. **High variance** (std dev: 389px) - Inconsistent performance
2. **X-coordinate issues** - Multiple cases with massive horizontal errors
3. **Edge detection problems** - Cursor near screen edges causes failures
4. **Mean error: 318.8px** - Dragged down by outliers

### 🔍 Pattern Analysis:

**Bimodal Distribution:**
- **Mode 1**: Good accuracy (25-70px) - 57% of cases
- **Mode 2**: Poor accuracy (230-1097px) - 43% of cases

**X-Coordinate Issue:**
- Clicks 14, 16: Perfect Y, terrible X
- Suggests systematic issue with horizontal cursor detection
- Possibly related to cursor position near edges or specific UI elements

## Comparison with Previous Tests

| Test | Cursor Visible? | Mean Error | Median Error | Best Case | Detection Rate |
|------|----------------|------------|--------------|-----------|----------------|
| **Video Frames** (earlier test) | ✅ Yes | N/A* | ~5-10px | ~5-10px | ~70% |
| **Screenshots v1** (no cursor) | ❌ No | 894.8px | 875.2px | 50.5px | 66.7% |
| **Screenshots v2** (with cursor) | ✅ Yes | **318.8px** | **70.0px** | **25.4px** | **100%** |

*Earlier video frame test was qualitative visual inspection, not quantitative ground truth

## Conclusions

### Performance Assessment:

**When it works well (57% of cases):**
- Error: 25-70px
- Usable for approximate trajectory visualization
- May be acceptable for coarse-grained training data

**When it fails (43% of cases):**
- Error: 230-1097px
- Systematic X-coordinate detection issues
- Not usable for training

### Root Causes of Failures:

1. **Edge Effects**: Cursor near screen edges (X: 41px, 305px)
2. **Horizontal Bias**: Y-coordinates often accurate even when X fails
3. **Context Confusion**: AI may be using UI element positions instead of actual cursor

### Recommendations:

**✅ DO:**
- Use for **approximate trajectory understanding** (70px median is reasonable)
- Filter out outliers (> 100px error) in post-processing
- Focus on center-screen interactions (better accuracy)
- Use as **supplementary data**, not primary training source

**❌ DO NOT:**
- Rely on for **precise PyAutoGUI commands** (too much variance)
- Use for **edge interactions** (consistent failures)
- Expect **production-quality training data** (43% failure rate)

**🎯 BETTER ALTERNATIVES:**

1. **Event-Capture Recorder** (±1px accuracy)
   - Still the gold standard for production data
   - No inference errors
   - Captures exact click coordinates

2. **Hybrid Approach**
   - Frame-by-frame for center-screen actions (filter < 100px)
   - Event capture for critical/edge interactions
   - Use AI extraction for understanding, not precision

## Cost-Benefit Analysis

**For 10,000 screenshots:**
- **Cost**: ~$10 (Gemini Flash, $0.001 per image)
- **Quality**: 57% good (< 100px), 43% poor (> 100px)
- **Value**: Reasonable for exploratory analysis, not production

**Verdict**:
- ✅ Good for **retrofitting existing screenshots** at low cost
- ⚠️ Quality inconsistent, requires filtering
- ❌ Not a replacement for proper event capture

---

**Generated**: 2025-11-04
**Test Method**: Gemini 2.0 Flash Experimental on single screenshots
**Ground Truth Source**: Embedded in filenames (actual click coordinates)
**Result**: 28.6% excellent, 28.6% fair, 42.8% poor accuracy
