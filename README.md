# Basketball Shot Consistency Analyzer

A Django web app that analyzes basketball shooting form using MediaPipe pose estimation and DTW-based consistency scoring. Supports per-user accounts, shot comparison, and personalized ML model training.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
python manage.py migrate
```

### 3. Run
```bash
python manage.py runserver
```

Then open `http://127.0.0.1:8000` and register an account.

---

## Features

### Compare Shots
Upload a reference video (pro player or standard form) and your own shot. The app extracts joint angles frame-by-frame and uses DTW to measure how closely your mechanics match the reference. Results include an overall score (0–100), per-joint radar chart, angle curves, and written feedback.

### Train My Model
Upload your own labeled shots (made / missed). Once you have at least 3 made and 3 missed, the app trains a personalized binary classifier in the background. The model learns what your good shots look like from your own data — no reference video needed.

### Score My Shot
Upload a new shot video and your trained model predicts the probability it's a made shot, returning a 0–100 score.

---

## Project Structure

```
basketball_app/
├── manage.py
├── requirements.txt
├── db.sqlite3                          # SQLite database (auto-created)
├── config/
│   ├── settings.py
│   └── urls.py
├── shot_analyzer/                      # Django app
│   ├── models.py                       # AnalysisSession, TrainingSample, PersonalModel
│   ├── views.py                        # All page logic
│   ├── urls.py
│   ├── forms.py
│   ├── services.py                     # Background training thread
│   └── templates/shot_analyzer/
│       ├── base.html
│       ├── home.html
│       ├── compare.html
│       ├── results.html
│       ├── training.html
│       ├── score.html
│       └── history.html
├── core/
│   ├── pose_extractor.py               # MediaPipe pose extraction (+ world landmarks)
│   ├── consistency_scorer.py           # DTW consistency scoring
│   ├── shot_featurizer.py              # ShotSequence → 93-dim feature vector (phase-aware)
│   ├── shot_trainer.py                 # AdaptiveModelFitter (OneClassSVM) + legacy ShotModelFitter
│   ├── shot_augmenter.py               # Jitter + time-warp augmentation for small datasets
│   ├── shot_detector.py                # Auto-detect shot segments in a long video
│   └── shot_phase_detector.py          # Key-frame detection within one shot (P1/P2/P4/P7)
└── utils/
    └── visualizer.py                   # Radar chart, angle curves, bar chart, score card
```

---

## Page Routes

| URL | Description |
|-----|-------------|
| `/` | Home — recent sessions and model status |
| `/compare/` | Upload two videos and run DTW analysis |
| `/results/<id>/` | View analysis results with charts |
| `/history/` | All past analysis sessions |
| `/training/` | Upload labeled shots and train personal model |
| `/score/` | Score a new shot using trained model |

---

## Core Modules

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `pose_extractor.py` | `PoseExtractor` | Extracts 33 keypoint coordinates (image + world) + 8 joint angles per frame via MediaPipe |
| `consistency_scorer.py` | `ConsistencyScorer` | DTW comparison → per-joint scores → weighted total → feedback |
| `shot_featurizer.py` | `featurize()` | Converts a `ShotSequence` into a 93-dimensional feature vector (global stats + phase-aware features) |
| `shot_trainer.py` | `AdaptiveModelFitter` | One-class SVM on made shots only; stage-aware scoring formula |
| `shot_augmenter.py` | `augment_sequence()` | Jitter + time-warp augmentation to grow small made-shot datasets |
| `shot_detector.py` | `ShotDetector` | Detects and segments individual shooting motions from a long video using knee-angle valleys |
| `shot_phase_detector.py` | `ShotPhaseDetector` | Locates key frames (P1/P2/P4/P7) within one shot using MediaPipe world landmarks |
| `visualizer.py` | — | Angle curves, radar chart, bar chart, score card (matplotlib → base64 PNG) |

---

## Shot Phase Detection

`ShotPhaseDetector` finds four key frames within a single shooting motion, mirroring the wrist-height curve:

```
height
 |        P4 top (release)
 |       /
 |      /
 |  P2 /
 |   |/
 |  P1        P7 follow-through end
 |             \____
 +-------------------> time
```

| Phase | Name | Detection method |
|-------|------|-----------------|
| P1 | `P1_address` | Fixed offset before P2 (ready stance) |
| P2 | `P2_load` | Wrist lowest point before release (loading dip) |
| P4 | `P4_top` | Wrist at highest point = release |
| P7 | `P7_followthrough` | First valley after release (follow-through end) |

**Height signal (priority order)**

1. **MediaPipe world landmarks** (`pose_world_landmarks`) — coordinates estimated in metres with origin at the hip centre. Height = `−world_y`. More stable than image coordinates because MediaPipe compensates for body-proportion scale, but the estimate is still derived from a single monocular camera so it is not perfectly distance-independent — it degrades when the shooting angle or distance changes significantly between recordings.
2. **Inverted image landmarks** — `1.0 − landmark_y` (normalised 0–1). Used as fallback when world landmarks are unavailable (e.g. sequences loaded from stored JSON).

### Usage

```python
from core.shot_phase_detector import ShotPhaseDetector

detector = ShotPhaseDetector()

# Option A — from a pre-extracted ShotSequence
kf = detector.detect(seq)

# Option B — directly from a video file (runs MediaPipe internally)
kf = detector.detect_from_video("shot.mp4", start_frame=120, end_frame=420)

# Frame index → timestamp is automatic: timestamp = frame_idx / fps
print(kf.as_table())
# Phase                  Frame  Time (s)
# --------------------------------------
# P1_address                 6     0.200
# P2_load                   18     0.600
# P4_top                    41     1.367
# P7_followthrough          59     1.967

# Save a JPEG screenshot for each phase
saved = detector.save_screenshots(
    "shot.mp4", kf,
    output_dir="./phases/",
    segment_start_frame=120,   # absolute frame offset in the source video
)
# → ./phases/P1_address.jpg
# → ./phases/P2_load.jpg
# → ./phases/P4_top.jpg
# → ./phases/P7_followthrough.jpg
```

---

## ML Model Details

### Adaptive scoring (made shots only)

The system requires **only made shots** — no missed-shot labeling needed. It automatically switches strategy as your dataset grows, and re-trains after every new upload.

| Stage | Made shots | Method | Auto-trigger |
|-------|-----------|--------|-------------|
| 1 | 1 – 9 | DTW against averaged personal template | ✓ after each upload |
| 2 | 10 – 19 | OneClassSVM on augmented data (~60 samples) | ✓ after each upload |
| 3 | 20 + | OneClassSVM on real data only | ✓ after each upload |

### Scoring formulas

**Stage 1 — DTW consistency**

```
score = 100 × e^(−d / 0.10)
```

`d` is the path-normalised DTW distance between the new shot and the personal reference template (average of all made shots). Both series are Z-score normalised before comparison, so absolute angle offsets from camera position are ignored — only motion shape matters.

| DTW distance | Score |
|-------------|-------|
| 0.00 | 100 |
| 0.03 | ~74 |
| 0.07 | ~50 |

**Stage 2 / 3 — OneClassSVM**

```
z     = (decision_score − real_train_mean) / real_train_std
score = 100 / (1 + e^(−(z + 1.1)))
```

`real_train_mean` and `real_train_std` are computed on the user's **real** made shots only (not on synthetic augmented samples), so `z = 0` always corresponds to the user's true average made shot.

| z | Score | Meaning |
|---|-------|---------|
| +2 | ~95 | Well above your average |
| 0 | **75** | Your typical made shot |
| −1 | ~52 | Slightly below average |
| −2 | ~29 | Clear outlier |

The sigmoid shift (+1.1) aligns stage 2/3 with stage 1: both give ~70–80 for a shot that matches the user's normal made-shot form, so the score does not jump when crossing the 10-shot threshold.

### Data augmentation (stage 2)

When made shots are between 10 and 19, each real sequence is augmented to reach ~60 training samples:

- **Time warp** — resample to ±15% of original length (models pacing variation)
- **Angle jitter** — add Gaussian noise ±3° to every joint angle (models frame-to-frame variation)

The model trains on real + augmented combined, but score calibration uses only the real shots.

### Feature Vector (93 dimensions)

| Group | Dims | Content |
|-------|------|---------|
| A — Global statistics | 58 | 8 joints × 7 stats (mean, std, min, max, range, p25, p75) + frame count + duration |
| B — Phase angles | 16 | Joint angle at each of P1/P2/P4/P7 for 4 key joints (right elbow, shoulder, knee, hip) |
| C — Phase deltas | 8 | (P4 − P2) and (P7 − P4) angle change per key joint — captures load-to-release drive |
| D — Release quality | 3 | Wrist height at release (P4), load timing fraction, release timing fraction |
| E — Angular velocity | 8 | Mean and peak angular speed (deg/frame) for 4 key joints |

Groups B–E restore temporal structure that group A loses: two sequences with the same mean/std but opposite motion order produce identical group-A features, but different group-B/C features.

### Model versioning

The model bundle stores the feature dimension it was trained on. If `featurize()` is upgraded and produces a different vector size, `predict()` raises `ModelVersionError` instead of silently producing wrong results. The score page will prompt you to clear samples and retrain.

---

## Analyzed Joints

| Joint | Default Weight |
|-------|---------------|
| Right Elbow | 25% |
| Right Shoulder | 20% |
| Right Knee | 20% |
| Right Hip | 15% |
| Left Elbow | 10% |
| Left Shoulder | 5% |
| Left Knee | 5% |

Weights are fully adjustable per analysis session via sliders.

---

## Consistency Score Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 85–100 | Very consistent |
| B | 70–84 | Good, minor variation |
| C | 55–69 | Noticeable inconsistency |
| D | 0–54 | Focus on fundamentals |

---

## Filming Tips

- **Angle**: Side view or 45° works best
- **Distance**: Full body visible in frame
- **Duration**: Complete motion from dip to follow-through (3–5 seconds)
- **Stability**: Use a tripod or stable surface
- **Consistency**: For personal model training, film all videos from the same angle
