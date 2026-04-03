"""
consistency_scorer.py
----------------------
DTW-based consistency scoring between two shot sequences.
Supports: single comparison, multi-shot averaging, joint-level analysis,
and phase-level scoring with adjustable per-phase weights.
"""

import numpy as np
from scipy.signal import savgol_filter
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

# Default phase weights — Release is most important for shooting form
PHASE_WEIGHTS_DEFAULT: Dict[str, float] = {
    "Loading":        0.15,
    "Rising":         0.20,
    "Release":        0.45,
    "Follow-through": 0.20,
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
class PhaseScore:
    """Score for one of the four shooting phases."""
    phase: str
    score: float              # 0-100
    weight: float             # relative importance (user-adjustable)


@dataclass
class ConsistencyReport:
    overall_score: float
    joint_scores: List[JointScore]
    phase_scores: List[PhaseScore]           # one entry per phase
    most_inconsistent_joint: str
    most_inconsistent_phase: str
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

def _z_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize a 1-D array: subtract mean and divide by std.
    This removes absolute-level differences caused by camera angle or body
    proportions, so DTW compares the *shape* of the motion curve rather than
    the raw angle values. If the joint barely moves (std ≈ 0), return zeros."""
    std = np.std(arr)
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std


def _smooth(arr: np.ndarray) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to reduce per-frame pose-detection noise.
    Uses a window of 9 frames and a 2nd-order polynomial fit.
    Falls back to a 5-frame window when the sequence is too short for 9."""
    n = len(arr)
    if n < 5:
        return arr
    window = min(9, n if n % 2 == 1 else n - 1)
    return savgol_filter(arr, window_length=window, polyorder=2)


def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Compute the Dynamic Time Warping distance between two 1-D angle sequences.
    Both sequences are smoothed (Savitzky-Golay), NaN-interpolated, then Z-score
    normalized before comparison so that differences in camera angle or body
    proportions do not inflate the distance — only the temporal shape of the
    motion matters. Returns a non-negative float — lower means more similar."""
    s1 = _z_normalize(_smooth(_fill_nan(s1)))
    s2 = _z_normalize(_smooth(_fill_nan(s2)))

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


def _normalize_score(dtw_dist: float, scale: float = 0.10) -> float:
    """Convert a Z-normalized, length-normalized DTW distance into a 0–100 score.
    After dividing by (n+m), typical distances range 0.0–0.3.
    scale=0.10 means:
      distance 0.00 → 100 pts  (identical pattern)
      distance 0.07 → 50 pts   (moderate deviation)
      distance 0.15 → 22 pts   (large deviation)
    Larger scale = more lenient."""
    return float(100.0 * np.exp(-dtw_dist / scale))


# ── Shot phase detection ──────────────────────────────────────────────────────

PHASE_NAMES = ["Loading", "Rising", "Release", "Follow-through"]


def detect_shot_phases(seq: ShotSequence) -> Dict[str, Tuple[int, int]]:
    """Divide a shot sequence into four named phases by splitting frames into equal quarters:
    Loading (0–25%), Rising (25–50%), Release (50–75%), Follow-through (75–100%).
    Returns a dict mapping phase name → (start_frame, end_frame) index range."""
    n = len(seq.frames)
    q = n // 4
    return {
        "Loading":        (0,      q),
        "Rising":         (q,      2 * q),
        "Release":        (2 * q,  3 * q),
        "Follow-through": (3 * q,  n),
    }


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
    """Return 'right' or 'left': whichever elbow has the greater range of motion
    is assumed to be the shooting arm.  Works regardless of camera angle."""
    right_std = np.std(_fill_nan(seq.angle_series("right_elbow")))
    left_std  = np.std(_fill_nan(seq.angle_series("left_elbow")))
    return "right" if right_std >= left_std else "left"


# ── Main class ────────────────────────────────────────────────────────────────

class ConsistencyScorer:
    """
    Usage:
        scorer = ConsistencyScorer()
        report = scorer.compare(reference_sequence, user_sequence)

    Phase-based scoring:
        Each shot is split into 4 phases (Loading / Rising / Release / Follow-through).
        Each phase is scored independently by computing DTW across all active joints
        within that phase slice. The overall score is the weighted average of the
        four phase scores. Phase weights are user-adjustable.
    """

    def __init__(
        self,
        joint_weights: Dict[str, float] = None,
        phase_weights: Dict[str, float] = None,
        dtw_scale: float = 0.10,
    ):
        """Set up the scorer with optional custom weights and a DTW scale factor.
        joint_weights: maps joint name → importance within each phase (must sum to 1.0).
        phase_weights: maps phase name → importance for overall score (must sum to 1.0).
        dtw_scale: controls score sensitivity — increase to be more lenient."""
        self.weights       = joint_weights or JOINT_WEIGHTS
        self.phase_weights = phase_weights or PHASE_WEIGHTS_DEFAULT
        self.dtw_scale     = dtw_scale

    # ── Public API ─────────────────────────────────────────────────────────────

    def compare(self, ref: ShotSequence, user: ShotSequence) -> ConsistencyReport:
        """Compare a reference shot against a user shot and produce a full report.

        Scoring process:
          1. Auto-detect shooting sides; mirror user joint labels if opposite sides.
          2. Split both sequences into 4 phases (proportionally aligned).
          3. For each phase, compute DTW for each active joint → weighted joint score.
          4. Phase score = weighted average of its joint scores.
          5. Overall score = weighted average of the 4 phase scores.
        """
        ref_side  = _detect_shooting_side(ref)
        user_side = _detect_shooting_side(user)
        joint_map = _MIRROR_JOINTS if ref_side != user_side else {}

        active_joints = [(j, w) for j, w in self.weights.items() if w > 0]
        n_ref  = len(ref.frames)
        n_user = len(user.frames)

        user_phases = detect_shot_phases(user)

        phase_scores: List[PhaseScore] = []
        # Accumulate per-joint scores across phases for joint breakdown
        joint_score_accum: Dict[str, List[float]] = {j: [] for j, _ in active_joints}

        for phase_name in PHASE_NAMES:
            u_start, u_end = user_phases[phase_name]
            # Slice ref proportionally
            r_start = int(u_start / n_user * n_ref)
            r_end   = int(u_end   / n_user * n_ref)

            phase_joint_scores: List[Tuple[float, float]] = []  # (score, weight)
            for joint, weight in active_joints:
                ref_slice  = ref.angle_series(joint)[r_start:r_end]
                user_slice = user.angle_series(joint_map.get(joint, joint))[u_start:u_end]
                if len(ref_slice) < 2 or len(user_slice) < 2:
                    continue
                dist  = _dtw_distance(ref_slice, user_slice)
                score = _normalize_score(dist, self.dtw_scale)
                phase_joint_scores.append((score, weight))
                joint_score_accum[joint].append(score)

            if phase_joint_scores:
                total_w     = sum(w for _, w in phase_joint_scores)
                phase_score = sum(s * w for s, w in phase_joint_scores) / (total_w + 1e-8)
            else:
                phase_score = 50.0

            ph_weight = self.phase_weights.get(phase_name, 0.25)
            phase_scores.append(PhaseScore(
                phase=phase_name,
                score=round(phase_score, 1),
                weight=ph_weight,
            ))

        # Overall score = weighted average of phase scores
        total_ph_w = sum(ps.weight for ps in phase_scores)
        overall    = sum(ps.score * ps.weight for ps in phase_scores) / (total_ph_w + 1e-8)

        # Per-joint summary: average of that joint's score across all 4 phases
        joint_scores: List[JointScore] = []
        for joint, weight in active_joints:
            scores_list = joint_score_accum[joint]
            if not scores_list:
                continue
            joint_scores.append(JointScore(
                joint=joint,
                dtw_distance=0.0,
                score=round(float(np.mean(scores_list)), 1),
                weight=weight,
            ))

        worst_joint = min(joint_scores, key=lambda js: js.score)
        worst_joint.is_most_inconsistent = True
        worst_phase = min(phase_scores, key=lambda ps: ps.score).phase

        feedback = self._generate_feedback(joint_scores, worst_joint, worst_phase, phase_scores)

        return ConsistencyReport(
            overall_score=round(overall, 1),
            joint_scores=sorted(joint_scores, key=lambda js: js.score),
            phase_scores=phase_scores,
            most_inconsistent_joint=worst_joint.joint,
            most_inconsistent_phase=worst_phase,
            feedback=feedback,
        )

    def compare_multiple(
        self,
        shots: List[ShotSequence],
    ) -> Tuple["ConsistencyReport", List["ConsistencyReport"]]:
        """Measure self-consistency across a series of shots by comparing consecutive pairs.
        Returns a summary report (average scores) alongside per-pair reports."""
        if len(shots) < 2:
            raise ValueError("Need at least 2 shots to compare.")

        pair_reports = [self.compare(shots[i], shots[i + 1]) for i in range(len(shots) - 1)]

        avg_score = np.mean([r.overall_score for r in pair_reports])

        # Most frequently worst joint across pairs
        worst_joint_counts: Dict[str, int] = {}
        for r in pair_reports:
            worst_joint_counts[r.most_inconsistent_joint] = \
                worst_joint_counts.get(r.most_inconsistent_joint, 0) + 1
        most_common_worst_joint = max(worst_joint_counts, key=worst_joint_counts.get)

        # Average phase scores
        avg_phase_scores: List[PhaseScore] = []
        for ps in pair_reports[0].phase_scores:
            avg_s = float(np.mean([r.phase_scores[i].score
                                   for i, _ in enumerate(r.phase_scores)
                                   for r in pair_reports
                                   if r.phase_scores[i].phase == ps.phase]))
            avg_phase_scores.append(PhaseScore(phase=ps.phase, score=round(avg_s, 1), weight=ps.weight))

        summary = ConsistencyReport(
            overall_score=round(float(avg_score), 1),
            joint_scores=pair_reports[0].joint_scores,
            phase_scores=avg_phase_scores,
            most_inconsistent_joint=most_common_worst_joint,
            most_inconsistent_phase=pair_reports[0].most_inconsistent_phase,
            feedback=[f"Analyzed {len(shots)} shots — average consistency: {avg_score:.1f}"],
        )
        return summary, pair_reports

    # ── Feedback generation ───────────────────────────────────────────────────

    def _generate_feedback(
        self,
        joint_scores: List[JointScore],
        worst_joint: JointScore,
        worst_phase: str,
        phase_scores: List[PhaseScore],
    ) -> List[str]:
        """Build a list of human-readable feedback strings from the scoring results."""
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

        label = JOINT_LABELS.get(worst_joint.joint, worst_joint.joint)
        feedback.append(
            f"🎯 Most inconsistent joint: **{label}** "
            f"(avg score {worst_joint.score:.0f}) — "
            f"biggest deviation during **{worst_phase}**."
        )

        # Call out any phase scoring below 50
        for ps in phase_scores:
            if ps.score < 50:
                feedback.append(
                    f"📌 **{ps.phase}** phase scored only {ps.score:.0f} — "
                    f"focus on this part of the motion."
                )

        # Call out any joint averaging below 60
        for js in joint_scores:
            if js.score < 60:
                lbl = JOINT_LABELS.get(js.joint, js.joint)
                feedback.append(f"📌 {lbl} averaged {js.score:.0f} across all phases — needs attention.")

        return feedback
