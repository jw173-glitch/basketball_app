"""
shot_phase_detector.py
----------------------
Detect key shot phases within one shooting motion using MediaPipe pose data.

Phase map (wrist-height curve over time):

  height
   |        P4 top (release)
   |       /|
   |      / |
   |  P2 /  |
   |   |/   |
   |  P1    | P7 follow-through end
   |        |____
   +-------------------> time

  P1  address       : ready stance before the loading dip
  P2  load          : wrist at lowest point before rising (max knee bend)
  P4  top           : wrist at highest point = release
  P7  followthrough : first valley after release where motion settles

Height source (priority order)
-------------------------------
1. MediaPipe world landmarks  — pose_world_landmarks.y in metres from hip centre.
   World Y points downward, so height_above_hip = -world_y.  This is the most
   accurate signal because it is camera-angle and distance independent.
2. Inverted image landmarks   — 1.0 - landmark.y (normalised 0-1).
   Falls back to this when world landmarks are not available (e.g. sequences
   loaded from stored JSON that predates the world_landmarks field).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from core.pose_extractor import PoseExtractor, ShotSequence


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class KeyFrame:
    name: str
    frame_idx: int           # index within the ShotSequence (0-based)
    timestamp_sec: float     # frame_idx / fps
    thumbnail: Optional[np.ndarray] = None  # BGR image, attached on request


@dataclass
class ShotKeyFrames:
    """All detected key frames for one shot, keyed by phase name."""
    phases: Dict[str, KeyFrame] = field(default_factory=dict)
    fps: float = 30.0

    def __getitem__(self, name: str) -> KeyFrame:
        return self.phases[name]

    def as_table(self) -> str:
        """Human-readable summary: phase → frame_idx → timestamp."""
        lines = [f"{'Phase':<20} {'Frame':>6} {'Time (s)':>9}"]
        lines.append("-" * 38)
        for kf in sorted(self.phases.values(), key=lambda k: k.frame_idx):
            lines.append(f"{kf.name:<20} {kf.frame_idx:>6} {kf.timestamp_sec:>9.3f}")
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fill_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr


def _smooth(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n < 5:
        return arr
    window = min(11, n if n % 2 == 1 else n - 1)
    return savgol_filter(arr, window_length=window, polyorder=2)


def _shooting_side(seq: ShotSequence) -> Tuple[str, str]:
    """Return (knee_joint, wrist_joint) for the side with greater range of motion."""
    r_std = np.nanstd(_fill_nan(seq.angle_series("right_knee")))
    l_std = np.nanstd(_fill_nan(seq.angle_series("left_knee")))
    if r_std >= l_std:
        return "right_knee", "right_wrist"
    return "left_knee", "left_wrist"


def _build_height_curve(seq: ShotSequence, wrist_joint: str) -> np.ndarray:
    """
    Build a smoothed wrist-height curve from a ShotSequence.

    Prefers MediaPipe world landmarks (metres, camera-independent).
    Falls back to inverted image-Y when world landmarks are absent.

    Returns an array of length len(seq.frames) where larger values = higher hand.
    """
    # Try world landmarks first (pose_world_landmarks.y, origin = hip centre, Y↓)
    world_ys = np.array([
        f.world_landmarks.get(wrist_joint, (np.nan, np.nan, np.nan))[1]
        for f in seq.frames
    ])
    has_world = not np.all(np.isnan(world_ys))

    if has_world:
        # World Y is positive-down, so negate to get height above hip
        raw = _fill_nan(-world_ys)
    else:
        # Fall back: invert image Y (0 = top of image → high position)
        image_ys = np.array([
            f.landmarks.get(wrist_joint, (0.5, 0.5, 0.0))[1]
            for f in seq.frames
        ])
        raw = _fill_nan(1.0 - image_ys)

    return _smooth(raw)


# ── Main class ────────────────────────────────────────────────────────────────

class ShotPhaseDetector:
    """
    Locate key phases within a single shot.

    Step 1 — Build wrist-height curve from MediaPipe landmarks (world coords
             preferred; image-Y fallback).
    Step 2 — Find peaks and valleys:
               P4 = global argmax  (wrist at highest → release)
               P2 = argmin before P4  (loading dip)
               P1 = P2 − pre_address_sec  (ready stance)
               P7 = first local valley after P4  (follow-through end)
    Step 3 — Convert frame index to timestamp:  t = frame_idx / fps
    Step 4 — Optionally save JPEG screenshots at each phase.
    """

    def __init__(
        self,
        pre_address_sec: float = 0.4,    # seconds before P2 to place P1
        followthrough_sec: float = 0.8,  # search window after P4 for P7
    ):
        self.pre_address_sec = pre_address_sec
        self.followthrough_sec = followthrough_sec

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, seq: ShotSequence) -> ShotKeyFrames:
        """
        Detect key phases from a pre-extracted ShotSequence.

        Uses world landmarks when available (requires PoseExtractor ≥ this
        version). Falls back to image-Y for sequences loaded from stored JSON.
        """
        fps = seq.fps
        n = len(seq.frames)
        if n < 5:
            return ShotKeyFrames(fps=fps)

        _, wrist_joint = _shooting_side(seq)
        height = _build_height_curve(seq, wrist_joint)

        # Step 2a — P4: global wrist peak = release point
        p4_idx = int(np.argmax(height))

        # Step 2b — P2: wrist valley before P4 = loading dip
        p2_idx = int(np.argmin(height[: p4_idx + 1]))

        # Step 2c — P1: ready stance, fixed offset before P2
        p1_idx = max(0, p2_idx - max(1, int(self.pre_address_sec * fps)))

        # Step 2d — P7: first local valley after P4 (follow-through end)
        search_end = min(n, p4_idx + int(self.followthrough_sec * fps) + 1)
        post_peak = height[p4_idx: search_end]
        valleys, _ = find_peaks(-post_peak, prominence=0.01)
        p7_idx = (
            p4_idx + int(valleys[0])
            if len(valleys) > 0
            else min(n - 1, p4_idx + int(self.followthrough_sec * fps))
        )

        def _kf(name: str, idx: int) -> KeyFrame:
            # Step 3 — frame index → timestamp
            return KeyFrame(name=name, frame_idx=idx, timestamp_sec=idx / fps)

        return ShotKeyFrames(
            fps=fps,
            phases={
                "P1_address":       _kf("P1_address",       p1_idx),
                "P2_load":          _kf("P2_load",           p2_idx),
                "P4_top":           _kf("P4_top",            p4_idx),
                "P7_followthrough": _kf("P7_followthrough",  p7_idx),
            },
        )

    def detect_from_video(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> ShotKeyFrames:
        """
        Run MediaPipe on a video clip and detect key phases in one call.

        Parameters
        ----------
        video_path  : path to the video file
        start_frame : first frame index to process (inclusive)
        end_frame   : last frame index to process (exclusive); None = until end

        Returns ShotKeyFrames whose frame_idx values are relative to start_frame
        so they can be passed directly to save_screenshots / attach_thumbnails
        with segment_start_frame=start_frame.
        """
        extractor = PoseExtractor(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if end_frame is None:
            end_frame = total

        # Seek to start_frame and extract only the clip
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        seq = ShotSequence(fps=fps)
        with extractor._make_pose() as pose:
            for abs_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                fd = extractor._process_frame(frame, pose, abs_idx - start_frame, annotate=False)
                if fd is not None:
                    seq.frames.append(fd)
        cap.release()

        return self.detect(seq)

    # ── Screenshot helpers ────────────────────────────────────────────────────

    def attach_thumbnails(
        self,
        video_path: str,
        key_frames: ShotKeyFrames,
        segment_start_frame: int = 0,
        thumb_width: int = 160,
    ) -> None:
        """Attach a resized BGR thumbnail to each KeyFrame. In-place."""
        cap = cv2.VideoCapture(video_path)
        for kf in key_frames.phases.values():
            abs_frame = segment_start_frame + kf.frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                scale = thumb_width / w
                kf.thumbnail = cv2.resize(frame, (thumb_width, int(h * scale)))
        cap.release()

    def save_screenshots(
        self,
        video_path: str,
        key_frames: ShotKeyFrames,
        output_dir: str,
        segment_start_frame: int = 0,
        jpeg_quality: int = 92,
    ) -> Dict[str, str]:
        """
        Step 4 — Save a full-resolution JPEG for each key phase.

        Returns dict mapping phase name → saved file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved: Dict[str, str] = {}
        cap = cv2.VideoCapture(video_path)
        for name, kf in key_frames.phases.items():
            abs_frame = segment_start_frame + kf.frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                path = os.path.join(output_dir, f"{name}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                saved[name] = path
        cap.release()
        return saved
