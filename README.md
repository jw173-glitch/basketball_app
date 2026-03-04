# Basketball Shot Consistency Analyzer

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
streamlit run app.py
```

### 3. How to Use
1. Upload a **reference shooting video** in the sidebar (pro player / standard form; side angle at 45° recommended)
2. Upload **your shooting video**
3. Click "🚀 Start Analysis"
4. View scores, joint angle curves, and improvement suggestions

---

## Project Structure

```
basketball_app/
├── app.py                      # Streamlit main app
├── requirements.txt
├── core/
│   ├── pose_extractor.py       # MediaPipe pose extraction
│   └── consistency_scorer.py  # DTW consistency scoring
└── utils/
    └── visualizer.py           # All chart generation
```

## Core Modules

| Module | Function |
|--------|----------|
| `PoseExtractor` | Extract 33 keypoint coordinates + 8 joint angles per frame |
| `ConsistencyScorer` | DTW comparison → joint scores → weighted total → text feedback |
| `visualizer` | Angle curves, radar chart, bar chart, skeleton video |

## Analyzed Joints

| Joint | Weight | Description |
|-------|--------|-------------|
| Right Elbow | 35% | Most critical for the shooting release |
| Right Shoulder | 20% | Upper body power source |
| Right Knee | 20% | Jump / push-off motion |
| Right Hip | 15% | Core body balance |
| Left Elbow | 10% | Guide hand control |

## Consistency Score Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 85-100 | Very consistent, professional level |
| B | 70-84  | Good, minor adjustments needed |
| C | 55-69  | Some variation, targeted practice needed |
| D | 0-54   | Inconsistent, focus on fundamentals |

## Filming Tips
- **Angle**: Side view at 45° or direct side view works best
- **Distance**: Ensure the full body is visible in the frame
- **Duration**: A complete shooting motion (jump to landing), 3-5 seconds is enough
- **Stability**: Use a tripod or rest the phone/camera on a stable surface
