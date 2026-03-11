"""
consistency_scorer.py
----------------------
DTW-based consistency scoring between two shot sequences.
Supports: single comparison, multi-shot averaging, joint-level analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from core.pose_extractor import ShotSequence


# ── Joint weights ─────────────────────────────────────────────────────────────

JOINT_WEIGHTS: Dict[str, float] = {
    "right_elbow":    0.25,   # most critical for release
    "right_shoulder": 0.20,
    "right_wrist":    0.00,   # no direct wrist angle from MediaPipe
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
    dtw_distance: float       # raw DTW distance (lower = better)
    score: float              # normalized 0-100 (higher = better)
    weight: float
    is_most_inconsistent: bool = False


@dataclass
class ConsistencyReport:
    overall_score: float
    joint_scores: List[JointScore]
    most_inconsistent_joint: str
    most_inconsistent_phase: str   # "Loading" / "Rising" / "Release" / "Follow-through"
    feedback: List[str]

    @property
    def grade(self) -> str:
        """Convert the overall score into a letter grade: A (≥85), B (≥70), C (≥55), D (<55)."""
        if self.overall_score >= 85:
            return "A"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 55:
            return "C"
        else:
            return "D"


# ── DTW core ──────────────────────────────────────────────────────────────────

def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping distance between two 1-D angle sequences.
    DTW finds the optimal alignment between sequences of different lengths or speeds,
    making it more robust than frame-by-frame subtraction for comparing motion data.
    NaN values are interpolated before comparison. Returns a non-negative float —
    lower means the two sequences are more similar."""
    s1 = _fill_nan(s1)
    s2 = _fill_nan(s2)

    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values in a 1-D array using linear interpolation over valid neighbors.
    If the entire array is NaN (joint never detected), returns an all-zero array instead
    so downstream DTW computation does not crash."""
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    indices = np.arange(len(arr))
    arr[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], arr[~nan_mask])
    return arr


def _normalize_score(dtw_dist: float, scale: float = 200.0) -> float:
    """Convert a raw DTW distance into a human-readable 0–100 consistency score
    using exponential decay: score = 100 * exp(-distance / scale).
    A distance of 0 yields a perfect 100; the scale parameter controls how quickly
    the score drops — larger scale = more tolerant of deviation."""
    return float(100.0 * np.exp(-dtw_dist / scale))


# ── Shot phase detection ──────────────────────────────────────────────────────

def detect_shot_phases(seq: ShotSequence) -> Dict[str, Tuple[int, int]]:
    """Divide a shot sequence into four named phases by splitting frames into equal quarters:
    Loading (0–25%), Rising (25–50%), Release (50–75%), Follow-through (75–100%).
    Returns a dict mapping phase name → (start_frame, end_frame) index range.
    Note: this is a simple uniform split; a more advanced version could detect the
    actual jump point from the knee angle curve for more precise phase boundaries."""
    n = len(seq.frames)
    q = n // 4
    return {
        "Loading":        (0,      q),
        "Rising":         (q,      2 * q),
        "Release":        (2 * q,  3 * q),
        "Follow-through": (3 * q,  n),
    }


def find_most_inconsistent_phase(
    ref: ShotSequence,
    user: ShotSequence,
    joint: str,
) -> str:
    """Identify which shot phase has the largest DTW distance for a specific joint.
    The user sequence is split into four phases; the reference is sliced proportionally
    to match each phase's relative position. Returns the phase name (e.g. 'Release')
    where the given joint deviates most from the reference."""
    phases = detect_shot_phases(user)
    worst_phase, worst_dist = "Release", -1.0

    ref_series  = ref.angle_series(joint)
    user_series = user.angle_series(joint)

    for phase_name, (start, end) in phases.items():
        r_start = int(start / len(user.frames) * len(ref.frames))
        r_end   = int(end   / len(user.frames) * len(ref.frames))
        dist = _dtw_distance(ref_series[r_start:r_end], user_series[start:end])
        if dist > worst_dist:
            worst_dist  = dist
            worst_phase = phase_name

    return worst_phase


# ── Main class ────────────────────────────────────────────────────────────────

class ConsistencyScorer:
    """
    Usage:
        scorer = ConsistencyScorer()
        report = scorer.compare(reference_sequence, user_sequence)
    """

    def __init__(self, joint_weights: Dict[str, float] = None, dtw_scale: float = 200.0):
        """Set up the scorer with optional custom joint weights and a DTW scale factor.
        joint_weights: maps joint name → importance (must sum to 1.0); defaults to JOINT_WEIGHTS.
        dtw_scale: controls score sensitivity — increase to be more lenient with deviations."""
        self.weights   = joint_weights or JOINT_WEIGHTS
        self.dtw_scale = dtw_scale

    def compare(self, ref: ShotSequence, user: ShotSequence) -> ConsistencyReport:
        """Compare a reference shot sequence against a user shot sequence and produce a full report.
        For each active joint (weight > 0), computes DTW distance and converts it to a 0–100 score.
        The overall score is the weighted average across all joints. Also identifies the worst joint,
        its most inconsistent phase, and generates human-readable feedback."""
        joint_scores  = []
        active_joints = [j for j, w in self.weights.items() if w > 0]

        for joint in active_joints:
            ref_series  = ref.angle_series(joint)
            user_series = user.angle_series(joint)
            dist  = _dtw_distance(ref_series, user_series)
            score = _normalize_score(dist, self.dtw_scale)
            joint_scores.append(JointScore(
                joint=joint,
                dtw_distance=dist,
                score=score,
                weight=self.weights[joint],
            ))

        total_w = sum(js.weight for js in joint_scores)
        overall = sum(js.score * js.weight for js in joint_scores) / (total_w + 1e-8)

        worst = min(joint_scores, key=lambda js: js.score)
        worst.is_most_inconsistent = True

        worst_phase = find_most_inconsistent_phase(ref, user, worst.joint)
        feedback    = self._generate_feedback(joint_scores, worst, worst_phase)

        return ConsistencyReport(
            overall_score=round(overall, 1),
            joint_scores=sorted(joint_scores, key=lambda js: js.score),
            most_inconsistent_joint=worst.joint,
            most_inconsistent_phase=worst_phase,
            feedback=feedback,
        )

    def compare_multiple(
        self,
        shots: List[ShotSequence],
    ) -> Tuple[ConsistencyReport, List[ConsistencyReport]]:
        """Measure self-consistency across a series of shots by comparing consecutive pairs.
        Compares shots[0] vs shots[1], shots[1] vs shots[2], etc., producing one report per pair.
        Returns a summary report (average score, most frequently worst joint) alongside the
        full list of per-pair reports. Requires at least 2 shots."""
        if len(shots) < 2:
            raise ValueError("Need at least 2 shots to compare.")

        pair_reports = []
        for i in range(len(shots) - 1):
            pair_reports.append(self.compare(shots[i], shots[i + 1]))

        avg_score = np.mean([r.overall_score for r in pair_reports])
        worst_joint_counts: Dict[str, int] = {}
        for r in pair_reports:
            worst_joint_counts[r.most_inconsistent_joint] = \
                worst_joint_counts.get(r.most_inconsistent_joint, 0) + 1
        most_common_worst = max(worst_joint_counts, key=worst_joint_counts.get)

        summary = ConsistencyReport(
            overall_score=round(float(avg_score), 1),
            joint_scores=pair_reports[0].joint_scores,
            most_inconsistent_joint=most_common_worst,
            most_inconsistent_phase=pair_reports[0].most_inconsistent_phase,
            feedback=[f"Analyzed {len(shots)} shots — average consistency: {avg_score:.1f}"],
        )
        return summary, pair_reports

    # ── Feedback generation ───────────────────────────────────────────────────

    def _generate_feedback(
        self,
        joint_scores: List[JointScore],
        worst: JointScore,
        worst_phase: str,
    ) -> List[str]:
        """Build a list of human-readable feedback strings from the scoring results.
        Generates an overall assessment based on average score, calls out the most
        inconsistent joint and its worst phase, and flags any individual joint that
        scored below 60 as needing focused attention."""
        feedback = []

        JOINT_LABELS = {
            "right_elbow":    "Right Elbow",
            "left_elbow":     "Left Elbow",
            "right_shoulder": "Right Shoulder",
            "left_shoulder":  "Left Shoulder",
            "right_knee":     "Right Knee",
            "left_knee":      "Left Knee",
            "right_hip":      "Right Hip",
            "left_hip":       "Left Hip",
        }

        avg = np.mean([js.score for js in joint_scores])
        if avg >= 85:
            feedback.append("✅ Very consistent form — keep it up!")
        elif avg >= 70:
            feedback.append("👍 Good overall form with minor variation.")
        elif avg >= 55:
            feedback.append("⚠️ Noticeable inconsistency — targeted practice recommended.")
        else:
            feedback.append("❌ Low consistency — focus on fundamentals first.")

        label = JOINT_LABELS.get(worst.joint, worst.joint)
        feedback.append(f"🎯 Most inconsistent joint: **{label}** — biggest deviation during **{worst_phase}**.")

        for js in joint_scores:
            if js.score < 60:
                l = JOINT_LABELS.get(js.joint, js.joint)
                feedback.append(f"📌 {l} scored only {js.score:.0f} — needs focused attention.")

        return feedback
