"""
shot_featurizer.py
------------------
Convert a ShotSequence into a fixed-length 58-dim numpy feature vector
suitable for scikit-learn classifiers.
"""

import numpy as np
from typing import List

from core.pose_extractor import ShotSequence

JOINTS: List[str] = [
    "right_elbow",    "left_elbow",
    "right_shoulder", "left_shoulder",
    "right_knee",     "left_knee",
    "right_hip",      "left_hip",
]

# 8 joints x 7 stats + 2 global = 58 total
FEATURE_NAMES: List[str] = [
    f"{joint}_{stat}"
    for joint in JOINTS
    for stat in ["mean", "std", "min", "max", "range", "p25", "p75"]
] + ["frame_count_norm", "duration_sec"]


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs; return zeros if the entire array is NaN."""
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    indices = np.arange(len(arr))
    arr[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], arr[~nan_mask])
    return arr


def _joint_stats(series: np.ndarray) -> np.ndarray:
    """Return 7 statistics for one joint's angle time series."""
    series = _fill_nan(series)
    n = len(series)
    p25_idx = max(0, int(0.25 * n) - 1)
    p75_idx = max(0, int(0.75 * n) - 1)
    return np.array([
        np.mean(series),
        np.std(series),
        np.min(series),
        np.max(series),
        np.max(series) - np.min(series),
        series[p25_idx],
        series[p75_idx],
    ], dtype=np.float32)


def featurize(seq: ShotSequence) -> np.ndarray:
    """Extract a 58-dimensional feature vector from a ShotSequence.

    Raises ValueError if the sequence has fewer than 5 frames."""
    if len(seq.frames) < 5:
        raise ValueError(f"Sequence too short to featurize ({len(seq.frames)} frames, need >= 5)")

    joint_features = np.concatenate([
        _joint_stats(seq.angle_series(joint))
        for joint in JOINTS
    ])

    global_features = np.array([
        min(len(seq.frames) / 300.0, 1.0),
        float(seq.duration_sec),
    ], dtype=np.float32)

    return np.concatenate([joint_features, global_features])
