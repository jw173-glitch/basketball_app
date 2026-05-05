"""
shot_featurizer.py
------------------
Convert a ShotSequence into a fixed-length numpy feature vector for scikit-learn.

Feature groups (93 dimensions total)
--------------------------------------
A. Global statistics  [0:58]   — original 58-dim representation
   8 joints × 7 stats (mean, std, min, max, range, p25, p75) = 56
   + frame_count_norm, duration_sec                           =  2

B. Phase angles       [58:74]  — joint angle at each key frame
   4 key joints × 4 phases (P1/P2/P4/P7)                    = 16
   Captures *what* the body looks like at each moment.

C. Phase deltas       [74:82]  — how much each joint moved
   (P4 − P2) for 4 joints = 4   load→release drive
   (P7 − P4) for 4 joints = 4   release→follow-through
   Captures *how far* the joint travelled between phases.

D. Release quality    [82:85]  — wrist height and timing
   wrist height at P4 (world metres or normalised image-Y)   =  1
   P2 frame fraction (load timing)                           =  1
   P4 frame fraction (release timing)                        =  1

E. Angular velocity   [85:93]  — speed of key joints
   mean |velocity| and peak |velocity| for 4 key joints      =  8
   Captures how explosively the joint accelerated.
"""

import numpy as np
from typing import List

from core.pose_extractor import ShotSequence

# ── Joint lists ───────────────────────────────────────────────────────────────

JOINTS: List[str] = [
    "right_elbow",    "left_elbow",
    "right_shoulder", "left_shoulder",
    "right_knee",     "left_knee",
    "right_hip",      "left_hip",
]

# Joints used for phase/velocity features (shooting side, most informative)
PHASE_JOINTS: List[str] = [
    "right_elbow", "right_shoulder", "right_knee", "right_hip",
]

_PHASE_ORDER: List[str] = [
    "P1_address", "P2_load", "P4_top", "P7_followthrough",
]

FEATURE_DIM: int = 93

FEATURE_NAMES: List[str] = (
    # Group A
    [f"{j}_{s}" for j in JOINTS
     for s in ["mean", "std", "min", "max", "range", "p25", "p75"]]
    + ["frame_count_norm", "duration_sec"]
    # Group B
    + [f"{phase}_{joint}" for phase in _PHASE_ORDER for joint in PHASE_JOINTS]
    # Group C
    + [f"delta_load_release_{j}" for j in PHASE_JOINTS]
    + [f"delta_release_followthrough_{j}" for j in PHASE_JOINTS]
    # Group D
    + ["release_wrist_height", "load_timing_frac", "release_timing_frac"]
    # Group E
    + [f"{j}_vel_mean" for j in PHASE_JOINTS]
    + [f"{j}_vel_peak" for j in PHASE_JOINTS]
)

assert len(FEATURE_NAMES) == FEATURE_DIM, f"FEATURE_NAMES length {len(FEATURE_NAMES)} != {FEATURE_DIM}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fill_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr


def _joint_stats(series: np.ndarray) -> np.ndarray:
    """7 statistics for one joint's angle time series."""
    series = _fill_nan(series)
    n = len(series)
    p25_idx = max(0, int(0.25 * n) - 1)
    p75_idx = max(0, int(0.75 * n) - 1)
    return np.array([
        np.mean(series), np.std(series),
        np.min(series),  np.max(series),
        np.max(series) - np.min(series),
        series[p25_idx], series[p75_idx],
    ], dtype=np.float32)


def _angle_at(seq: ShotSequence, frame_idx: int, joint: str) -> float:
    """Return the angle of a joint at a specific frame, or 0.0 on failure."""
    if frame_idx < 0 or frame_idx >= len(seq.frames):
        return 0.0
    v = seq.frames[frame_idx].angles.get(joint, np.nan)
    return 0.0 if np.isnan(v) else float(v)


# ── Group A: global statistics (original 58) ──────────────────────────────────

def _global_stats(seq: ShotSequence) -> np.ndarray:
    joint_feats = np.concatenate([_joint_stats(seq.angle_series(j)) for j in JOINTS])
    global_feats = np.array([
        min(len(seq.frames) / 300.0, 1.0),
        float(seq.duration_sec),
    ], dtype=np.float32)
    return np.concatenate([joint_feats, global_feats])


# ── Groups B–E: phase-aware features (new 35) ─────────────────────────────────

def _phase_features(seq: ShotSequence) -> np.ndarray:
    """
    Detect key phases and extract groups B, C, D, E (35 dims total).
    Returns zeros on any failure so training never crashes.
    """
    from core.shot_phase_detector import ShotPhaseDetector, _shooting_side

    n = len(seq.frames)
    n_pj = len(PHASE_JOINTS)
    n_ph = len(_PHASE_ORDER)
    zeros = np.zeros(n_ph * n_pj + 2 * n_pj + 3 + 2 * n_pj, dtype=np.float32)

    if n < 5:
        return zeros

    try:
        kf = ShotPhaseDetector().detect(seq)
    except Exception:
        return zeros

    if not kf.phases:
        return zeros

    feats: List[float] = []

    # ── B: angle at each phase ────────────────────────────────────────────────
    for phase_name in _PHASE_ORDER:
        kframe = kf.phases.get(phase_name)
        fidx = kframe.frame_idx if kframe else -1
        for joint in PHASE_JOINTS:
            feats.append(_angle_at(seq, fidx, joint))

    # ── C: phase deltas ───────────────────────────────────────────────────────
    p2 = kf.phases.get("P2_load")
    p4 = kf.phases.get("P4_top")
    p7 = kf.phases.get("P7_followthrough")

    for joint in PHASE_JOINTS:
        feats.append(_angle_at(seq, p4.frame_idx, joint) - _angle_at(seq, p2.frame_idx, joint)
                     if p2 and p4 else 0.0)

    for joint in PHASE_JOINTS:
        feats.append(_angle_at(seq, p7.frame_idx, joint) - _angle_at(seq, p4.frame_idx, joint)
                     if p4 and p7 else 0.0)

    # ── D: release quality ────────────────────────────────────────────────────
    if p4 and p4.frame_idx < n:
        frame = seq.frames[p4.frame_idx]
        _, wrist_joint = _shooting_side(seq)
        wl = frame.world_landmarks.get(wrist_joint)
        if wl and not np.isnan(wl[1]):
            release_h = float(-wl[1])           # world Y↓, negate for height
        else:
            img = frame.landmarks.get(wrist_joint)
            release_h = float(1.0 - img[1]) if img else 0.5
    else:
        release_h = 0.5

    feats.append(release_h)
    feats.append(float(p2.frame_idx / n) if p2 else 0.4)
    feats.append(float(p4.frame_idx / n) if p4 else 0.7)

    # ── E: angular velocity ───────────────────────────────────────────────────
    for joint in PHASE_JOINTS:
        series = _fill_nan(seq.angle_series(joint))
        if len(series) > 1:
            vel = np.abs(np.diff(series))
            feats.extend([float(np.mean(vel)), float(np.max(vel))])
        else:
            feats.extend([0.0, 0.0])

    return np.array(feats, dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def featurize(seq: ShotSequence) -> np.ndarray:
    """
    Extract a 93-dimensional feature vector from a ShotSequence.

    Raises ValueError if the sequence has fewer than 5 frames.
    """
    if len(seq.frames) < 5:
        raise ValueError(f"Sequence too short ({len(seq.frames)} frames, need >= 5)")

    return np.concatenate([
        _global_stats(seq),    # 58 dims — global joint angle statistics
        _phase_features(seq),  # 35 dims — phase angles, deltas, release, velocity
    ]).astype(np.float32)
