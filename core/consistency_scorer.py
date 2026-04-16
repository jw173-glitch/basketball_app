"""
consistency_scorer.py
----------------------
DTW-based consistency scoring between two shot sequences.
Supports single comparison and multi-shot averaging with per-joint breakdown.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple
from dataclasses import dataclass

from core.pose_extractor import ShotSequence


# ── Joint weights ─────────────────────────────────────────────────────────────

JOINT_WEIGHTS: Dict[str, float] = {
    "right_elbow":    0.25,
    "right_shoulder": 0.20,
    "right_wrist":    0.00,
    "right_knee":     0.20,
    "right_hip":      0.15,
    "left_elbow":     0.10,
    "left_shoulder":  0.05,
    "left_knee":      0.05,
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class JointScore:
    joint: str
    dtw_distance: float
    score: float
    weight: float
    is_most_inconsistent: bool = False


@dataclass
class ConsistencyReport:
    overall_score: float
    joint_scores: List[JointScore]
    most_inconsistent_joint: str
    feedback: List[str]

    @property
    def grade(self) -> str:
        """Convert overall score to letter grade: A (≥85), B (≥70), C (≥55), D (<55)."""
        if self.overall_score >= 85:
            return "A"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 55:
            return "C"
        else:
            return "D"


# ── DTW helpers ───────────────────────────────────────────────────────────────

def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values via linear interpolation; return zeros if all NaN."""
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    indices = np.arange(len(arr))
    arr[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], arr[~nan_mask])
    return arr


def _smooth(arr: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing to reduce per-frame pose detection noise."""
    n = len(arr)
    if n < 5:
        return arr
    window = min(9, n if n % 2 == 1 else n - 1)
    return savgol_filter(arr, window_length=window, polyorder=2)


def _z_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize: removes absolute angle differences from camera/height.
    Only the shape of the motion curve is compared, not raw angle values."""
    std = np.std(arr)
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


def _cross_align(s1: np.ndarray, s2: np.ndarray) -> tuple:
    """Find the integer lag that maximises cross-correlation and trim to overlap.
    Corrects for shots that start at slightly different time offsets.
    Lag is capped at 30% of the shorter sequence length."""
    from scipy.signal import correlate
    corr = correlate(s1, s2, mode='full')
    lag = int(np.argmax(corr)) - (len(s2) - 1)
    max_lag = int(min(len(s1), len(s2)) * 0.30)
    lag = int(np.clip(lag, -max_lag, max_lag))
    if lag > 0:
        s1 = s1[lag:]
    elif lag < 0:
        s2 = s2[-lag:]
    min_len = min(len(s1), len(s2))
    return s1[:min_len], s2[:min_len]


def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute DTW distance between two joint angle time series.

    Pipeline:
      1. Fill NaN via linear interpolation.
      2. Savitzky-Golay smoothing to remove per-frame noise.
      3. Z-score normalization — compares motion shape, not raw angles.
      4. Cross-correlation alignment — corrects for different shot start offsets.
      5. Path-normalized DTW (÷ n+m) for length-invariant distance.
    """
    s1 = _z_normalize(_smooth(_fill_nan(s1)))
    s2 = _z_normalize(_smooth(_fill_nan(s2)))

    if len(s1) >= 5 and len(s2) >= 5:
        s1, s2 = _cross_align(s1, s2)

    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m] / (n + m))


def _normalize_score(dtw_dist: float, scale: float = 0.10) -> float:
    """Convert DTW distance to a 0-100 score via exponential decay.
    scale=0.10: distance 0.00→100, 0.07→50, 0.15→22."""
    return float(100.0 * np.exp(-dtw_dist / scale))


# ── Shooting-side detection & joint remapping ─────────────────────────────────

_MIRROR_JOINTS: Dict[str, str] = {
    "right_elbow":    "left_elbow",
    "left_elbow":     "right_elbow",
    "right_shoulder": "left_shoulder",
    "left_shoulder":  "right_shoulder",
    "right_knee":     "left_knee",
    "left_knee":      "right_knee",
    "right_hip":      "left_hip",
    "left_hip":       "right_hip",
}


def _detect_shooting_side(seq: ShotSequence) -> str:
    """Return 'right' or 'left': the elbow with greater range of motion is the shooting arm."""
    right_std = np.std(_fill_nan(seq.angle_series("right_elbow")))
    left_std  = np.std(_fill_nan(seq.angle_series("left_elbow")))
    return "right" if right_std >= left_std else "left"


# ── Main class ────────────────────────────────────────────────────────────────

class ConsistencyScorer:
    """
    Compare two shot sequences using DTW and produce a joint-level report.

    Usage:
        scorer = ConsistencyScorer()
        report = scorer.compare(reference_sequence, user_sequence)
    """

    def __init__(
        self,
        joint_weights: Dict[str, float] = None,
        dtw_scale: float = 0.10,
    ):
        self.weights   = joint_weights or JOINT_WEIGHTS
        self.dtw_scale = dtw_scale

    def compare(self, ref: ShotSequence, user: ShotSequence) -> ConsistencyReport:
        """Compare reference and user shot sequences joint-by-joint using DTW.

        Auto-detects shooting sides and mirrors joint labels if filmed from
        opposite camera angles. Returns a ConsistencyReport with per-joint
        scores and an overall weighted score."""
        ref_side  = _detect_shooting_side(ref)
        user_side = _detect_shooting_side(user)
        joint_map = _MIRROR_JOINTS if ref_side != user_side else {}

        joint_scores: List[JointScore] = []
        for joint, weight in self.weights.items():
            if weight <= 0:
                continue
            ref_series  = ref.angle_series(joint)
            user_series = user.angle_series(joint_map.get(joint, joint))
            dist  = _dtw_distance(ref_series, user_series)
            score = _normalize_score(dist, self.dtw_scale)
            joint_scores.append(JointScore(
                joint=joint,
                dtw_distance=round(dist, 4),
                score=round(score, 1),
                weight=weight,
            ))

        total_w = sum(js.weight for js in joint_scores)
        overall = sum(js.score * js.weight for js in joint_scores) / (total_w + 1e-8)

        worst = min(joint_scores, key=lambda js: js.score)
        worst.is_most_inconsistent = True

        return ConsistencyReport(
            overall_score=round(overall, 1),
            joint_scores=sorted(joint_scores, key=lambda js: js.score),
            most_inconsistent_joint=worst.joint,
            feedback=self._generate_feedback(joint_scores, worst),
        )

    def compare_multiple(
        self,
        shots: List[ShotSequence],
    ) -> Tuple["ConsistencyReport", List["ConsistencyReport"]]:
        """Measure self-consistency across a series of shots by comparing consecutive pairs."""
        if len(shots) < 2:
            raise ValueError("Need at least 2 shots to compare.")
        pair_reports = [self.compare(shots[i], shots[i + 1]) for i in range(len(shots) - 1)]
        avg_score = float(np.mean([r.overall_score for r in pair_reports]))
        worst_counts: Dict[str, int] = {}
        for r in pair_reports:
            worst_counts[r.most_inconsistent_joint] = worst_counts.get(r.most_inconsistent_joint, 0) + 1
        summary = ConsistencyReport(
            overall_score=round(avg_score, 1),
            joint_scores=pair_reports[0].joint_scores,
            most_inconsistent_joint=max(worst_counts, key=worst_counts.get),
            feedback=[f"Analyzed {len(shots)} shots — average consistency: {avg_score:.1f}"],
        )
        return summary, pair_reports

    def _generate_feedback(self, joint_scores: List[JointScore], worst: JointScore) -> List[str]:
        JOINT_LABELS = {
            "right_elbow":    "Right Elbow",    "left_elbow":     "Left Elbow",
            "right_shoulder": "Right Shoulder", "left_shoulder":  "Left Shoulder",
            "right_knee":     "Right Knee",     "left_knee":      "Left Knee",
            "right_hip":      "Right Hip",      "left_hip":       "Left Hip",
        }
        feedback = []
        avg = float(np.mean([js.score for js in joint_scores]))
        if avg >= 85:
            feedback.append("✅ Very consistent form — keep it up!")
        elif avg >= 70:
            feedback.append("👍 Good overall form with minor variation.")
        elif avg >= 55:
            feedback.append("⚠️ Noticeable inconsistency — targeted practice recommended.")
        else:
            feedback.append("❌ Low consistency — focus on fundamentals first.")

        label = JOINT_LABELS.get(worst.joint, worst.joint)
        feedback.append(f"🎯 Most inconsistent joint: **{label}** (score {worst.score:.0f})")

        for js in joint_scores:
            if js.score < 60:
                lbl = JOINT_LABELS.get(js.joint, js.joint)
                feedback.append(f"📌 {lbl} scored {js.score:.0f} — needs attention.")

        return feedback
