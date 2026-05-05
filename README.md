# Basketball Shot Analyzer

A Django web app that analyzes basketball shooting form using MediaPipe pose estimation, DTW consistency scoring, and an adaptive one-class ML model. Built for CMU 18-738.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialize database
python manage.py migrate

# 3. Run
python manage.py runserver
```

Open `http://127.0.0.1:8000` and register an account.

**Recommended environment:** `conda activate basketball` (Python 3.10, MediaPipe 0.10, OpenCV 4.11)

---

## Features

### Compare Shots
Upload your shot against a professional player template or any reference video. MediaPipe extracts joint angles frame-by-frame; DTW measures how closely your mechanics match the reference. Results include an overall score (0–100), per-joint radar chart, angle curves, and written feedback. Shot boundaries are detected automatically — no manual trimming required.

### Train My Model (Made Shots Only)
Upload your made shots. The system builds a personal reference template and trains a scoring model automatically after every upload. No missed-shot labeling needed.

| Stage | Shots | Method |
|-------|-------|--------|
| 1 | 1 – 9 | DTW against your averaged template |
| 2 | 10 – 19 | OneClass SVM on augmented data (~60 samples) |
| 3 | 20 + | OneClass SVM on real data |

### Score My Shot
Upload a new shot and see how closely your mechanics match your own made-shot history. Results include a 0–100 score, radar chart, joint angle curves, per-joint breakdown, and feedback.

---

## Project Structure

```
basketball_app/
├── core/
│   ├── pose_extractor.py        # MediaPipe pose extraction (image + world landmarks)
│   ├── shot_detector.py         # Auto-detect shot segments from long video
│   ├── shot_phase_detector.py   # Key-frame detection: P1/P2/P4/P7
│   ├── shot_featurizer.py       # ShotSequence → 93-dim feature vector
│   ├── shot_augmenter.py        # Time-warp + angle jitter for small datasets
│   ├── shot_trainer.py          # AdaptiveModelFitter (OneClassSVM)
│   ├── consistency_scorer.py    # DTW scoring with per-joint breakdown
│   └── visualizer.py            # Radar chart, angle curves, score card
├── shot_analyzer/
│   ├── models.py                # AnalysisSession, TrainingSample, PersonalModel
│   ├── views.py
│   ├── services.py              # Background training thread
│   └── templates/
└── config/
```

---

## Shot Phase Detection

Key frames are detected from the wrist-height curve derived from MediaPipe world landmarks:

```
height
 |        P4 — release (wrist peak)
 |       /
 |  P2  /
 |   | /
 |  P1        P7 — follow-through end
 |              \____
 +--------------------> time
```

| Phase | Detection |
|-------|-----------|
| P1 address | Fixed offset before P2 |
| P2 load | Wrist lowest point before release |
| P4 top | Wrist at highest point |
| P7 follow-through | First valley after release |

World landmarks (metres, hip-origin) are used when available; falls back to inverted image-Y for stored sequences.

---

## ML Model

### Feature Vector (93 dimensions)

| Group | Dims | Content |
|-------|------|---------|
| A — Global statistics | 58 | 8 joints × 7 stats + frame count + duration |
| B — Phase angles | 16 | Joint angle at P1/P2/P4/P7 for 4 key joints |
| C — Phase deltas | 8 | (P4−P2) and (P7−P4) per key joint |
| D — Release quality | 3 | Wrist height at P4, load timing, release timing |
| E — Angular velocity | 8 | Mean and peak speed for 4 key joints |

Groups B–E restore temporal structure lost by summary statistics alone.

### Scoring Formulas

**Stage 1 — DTW**
```
score = 100 × e^(−d / 0.30)
```

**Stage 2/3 — OneClass SVM**
```
z     = (decision_score − real_train_mean) / real_train_std
score = 100 / (1 + e^(−(z + 1.1)))
```

`z = 0` (your average made shot) → **75 points**. Score calibration uses real shots only, not augmented data.

### Data Augmentation (Stage 2)

- **Time warp** — resample to ±15% of original length
- **Angle jitter** — Gaussian noise ±3° per joint

---

## Scoring Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 85–100 | Very consistent |
| B | 70–84 | Good, minor variation |
| C | 55–69 | Noticeable inconsistency |
| D | 0–54 | Focus on fundamentals |

---

## Joint Weights (Compare Shots)

| Joint | Default |
|-------|---------|
| Right Elbow | 25% |
| Right Shoulder | 20% |
| Right Knee | 20% |
| Right Hip | 15% |
| Left Elbow | 10% |
| Left Shoulder | 5% |
| Left Knee | 5% |

Adjustable per session via sliders.

---

## Future Direction

Planned upgrade to use **SMPL** (Skinned Multi-Person Linear Model) for the professional athlete reference database:
- Extract 3D body shape parameters (β) from YouTube clips of professional players
- Normalize comparisons by height and body proportion
- Enable SMPL-to-MediaPipe alignment for user-side real-time scoring

User-side processing will continue to use MediaPipe world landmarks, which provide camera-independent 3D joint coordinates comparable in accuracy to platform-native 3D pose frameworks.

---

## Filming Tips

- **Angle**: Side view or 45° diagonal works best
- **Distance**: Full body visible in frame
- **Duration**: Complete motion from dip to follow-through (3–5 seconds)
- **Stability**: Use a tripod or stable surface
- **Consistency**: Film all training videos from the same angle and distance
