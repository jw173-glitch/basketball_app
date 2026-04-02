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
│   ├── pose_extractor.py               # MediaPipe pose extraction
│   ├── consistency_scorer.py           # DTW consistency scoring
│   ├── shot_featurizer.py              # ShotSequence → 58-dim feature vector
│   └── shot_trainer.py                 # SVC / LogisticRegression fitting & prediction
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
| `pose_extractor.py` | `PoseExtractor` | Extracts 33 keypoint coordinates + 8 joint angles per frame via MediaPipe |
| `consistency_scorer.py` | `ConsistencyScorer` | DTW comparison → per-joint scores → weighted total → feedback |
| `shot_featurizer.py` | `featurize()` | Converts a `ShotSequence` into a 58-dimensional feature vector |
| `shot_trainer.py` | `ShotModelFitter` | Fits SVC or LogisticRegression; selects model based on sample count |
| `visualizer.py` | — | Angle curves, radar chart, bar chart, score card (matplotlib → base64 PNG) |

---

## ML Model Details

| Samples | Model Used | Reason |
|---------|-----------|--------|
| ≥ 15 | SVC (RBF kernel) | Better generalization for higher-dimensional data |
| < 15 | Logistic Regression (L2, C=0.1) | Fewer parameters, less overfitting risk |

Evaluation uses leave-one-out cross-validation. Score = `P(made) × 100`.

Feature vector (58 dimensions): for each of 8 joints — mean, std, min, max, range, 25th-percentile angle, 75th-percentile angle — plus normalized frame count and shot duration.

Minimum training requirement: **3 made + 3 missed shots**.

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
