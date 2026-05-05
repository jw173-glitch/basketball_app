"""
shot_augmenter.py
-----------------
Augment ShotSequence objects to artificially grow a small made-shot dataset.

Two transforms applied in combination per synthetic sample:

  1. Time warp  — linearly resample to ±15% of original length.
                  Models slight pacing differences between shots.
  2. Angle jitter — add Gaussian noise (±3°) to every joint angle.
                    Models natural frame-to-frame variation.

Usage
-----
    from core.shot_augmenter import augment_sequence, n_augments_for

    n = n_augments_for(n_real=8, target=60)   # → 7 per sample
    extras = augment_sequence(seq, n=n)        # list of n ShotSequences
"""

import numpy as np
from typing import List

from core.pose_extractor import FrameData, ShotSequence


# ── Transforms ────────────────────────────────────────────────────────────────

def _time_warp(seq: ShotSequence, factor: float) -> ShotSequence:
    n_src = len(seq.frames)
    n_dst = max(5, int(round(n_src * factor)))
    joints = list(seq.frames[0].angles.keys()) if seq.frames else []

    old_x = np.linspace(0.0, 1.0, n_src)
    new_x = np.linspace(0.0, 1.0, n_dst)

    resampled: dict = {}
    for joint in joints:
        series = np.array([f.angles.get(joint, np.nan) for f in seq.frames], dtype=float)
        mask = np.isnan(series)
        if mask.all():
            series[:] = 0.0
        elif mask.any():
            series[mask] = np.interp(old_x[mask], old_x[~mask], series[~mask])
        resampled[joint] = np.interp(new_x, old_x, series)

    out = ShotSequence(fps=seq.fps)
    for i in range(n_dst):
        out.frames.append(FrameData(
            frame_idx=i, landmarks={},
            angles={j: float(resampled[j][i]) for j in joints},
        ))
    return out


def _jitter(seq: ShotSequence, noise_deg: float, rng: np.random.Generator) -> ShotSequence:
    out = ShotSequence(fps=seq.fps)
    for f in seq.frames:
        out.frames.append(FrameData(
            frame_idx=f.frame_idx, landmarks={},
            angles={j: v + float(rng.normal(0.0, noise_deg))
                    for j, v in f.angles.items()},
        ))
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def augment_sequence(
    seq: ShotSequence,
    n: int,
    noise_deg: float = 3.0,
    warp_range: tuple = (0.85, 1.15),
    seed: int = None,
) -> List[ShotSequence]:
    """Return n augmented copies of seq (original not included)."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        factor = float(rng.uniform(*warp_range))
        warped  = _time_warp(seq, factor)
        results.append(_jitter(warped, noise_deg, rng))
    return results


def n_augments_for(n_real: int, target: int = 60) -> int:
    """
    Augmented copies per real sample so total reaches ~target.

    n_real=6  → 9   (6 + 6×9=54 → ~60)
    n_real=12 → 4   (12 + 12×4=48 → ~60)
    n_real=25 → 1   (25 + 25×1=50 → minimal boost)
    n_real=40 → 0   (already enough)
    """
    if n_real <= 0:
        return 0
    return max(0, int(np.ceil(max(0, target - n_real) / n_real)))
