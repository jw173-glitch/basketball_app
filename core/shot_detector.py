"""
shot_detector.py
----------------
Automatically detect and segment individual shooting motions from a video.

All timing parameters are expressed in seconds so the detector works correctly
regardless of whether the video is normal speed, slow motion, or high frame rate.
Internally, seconds are multiplied by the video's FPS to get frame counts.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scipy.signal import find_peaks, savgol_filter

from core.pose_extractor import PoseExtractor, ShotSequence


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ShotSegment:
    """One detected shooting motion within a longer video."""
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration_sec: float
    thumbnail: Optional[np.ndarray] = None   # BGR frame near release point


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fill_nan_1d(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs with linear interpolation; return zeros if all NaN."""
    arr = arr.copy()
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr


def _smooth_1d(arr: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing; falls back gracefully for short arrays."""
    n = len(arr)
    if n < 5:
        return arr
    window = min(9, n if n % 2 == 1 else n - 1)
    return savgol_filter(arr, window_length=window, polyorder=2)


def _merge_overlapping(segments: List[ShotSegment]) -> List[ShotSegment]:
    """Merge any two segments whose frame ranges overlap."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start_frame)
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.start_frame <= prev.end_frame:
            # extend previous segment to cover both
            new_end = max(prev.end_frame, seg.end_frame)
            fps_est = prev.end_sec / prev.end_frame if prev.end_frame else 30.0
            merged[-1] = ShotSegment(
                start_frame=prev.start_frame,
                end_frame=new_end,
                start_sec=prev.start_sec,
                end_sec=new_end / fps_est,
                duration_sec=(new_end - prev.start_frame) / fps_est,
                thumbnail=prev.thumbnail,
            )
        else:
            merged.append(seg)
    return merged


# ── Main class ────────────────────────────────────────────────────────────────

class ShotDetector:
    """
    Detect individual shooting motions from a full video sequence.

    Algorithm:
      1. Extract the shooting-side knee angle curve from the pose sequence.
      2. Smooth the curve and find valleys (knee-flexion dips) using
         scipy.signal.find_peaks on the negated signal.
      3. Each valley marks the loading phase of one shot.
      4. The segment start is placed pre_shot_sec before the valley.
      5. The segment end is placed followthrough_sec after the wrist reaches
         its highest point following the valley (= release point).
      6. Segments shorter than min_shot_sec or longer than max_shot_sec
         are discarded; remaining overlapping segments are merged.

    All time parameters are in seconds; frame counts are derived via FPS so
    the detector works correctly for slow-motion and high-FPS videos.
    """

    def __init__(
        self,
        min_knee_drop_deg: float = 15.0,   # minimum knee-bend depth to qualify as a shot
        pre_shot_sec: float = 0.7,          # seconds to include before the knee valley
        followthrough_sec: float = 0.8,     # seconds after wrist peak to include
        min_between_shots_sec: float = 1.0, # minimum gap between two shot detections
        min_shot_sec: float = 0.5,          # discard segments shorter than this
        max_shot_sec: float = 5.0,          # discard segments longer than this
    ):
        self.min_knee_drop_deg = min_knee_drop_deg
        self.pre_shot_sec = pre_shot_sec
        self.followthrough_sec = followthrough_sec
        self.min_between_shots_sec = min_between_shots_sec
        self.min_shot_sec = min_shot_sec
        self.max_shot_sec = max_shot_sec

    # ── Public API ─────────────────────────────────────────────────────────────

    def detect_from_video(self, video_path: str) -> Tuple[ShotSequence, List[ShotSegment]]:
        """Extract pose from a full video, then detect shot segments.
        Returns the full ShotSequence and the list of detected segments so
        callers can slice the sequence without re-processing the video."""
        extractor = PoseExtractor(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        # No max_frames cap here — we need the full video for detection
        full_seq = extractor.process_video(video_path, annotate=False, max_frames=9999)
        segments = self.detect(full_seq)

        # Attach thumbnail (frame near release) from the video
        _attach_thumbnails(video_path, full_seq, segments)

        return full_seq, segments

    def detect(self, seq: ShotSequence) -> List[ShotSegment]:
        """Run shot detection on a pre-extracted ShotSequence.
        Returns a list of ShotSegment objects (sorted by start time)."""
        fps = seq.fps
        n = len(seq.frames)
        if n < 10:
            return []

        # Determine shooting side by whichever knee has more range of motion
        right_std = np.nanstd(_fill_nan_1d(seq.angle_series("right_knee")))
        left_std  = np.nanstd(_fill_nan_1d(seq.angle_series("left_knee")))
        knee_joint  = "right_knee"  if right_std >= left_std else "left_knee"
        wrist_joint = "right_wrist" if knee_joint == "right_knee" else "left_wrist"

        knee_curve  = _smooth_1d(_fill_nan_1d(seq.angle_series(knee_joint)))
        wrist_y     = _fill_nan_1d(np.array([
            f.landmarks.get(wrist_joint, (0.5, 0.5, 0.0))[1]
            for f in seq.frames
        ]))

        # Find knee valleys (negated so peaks become valleys)
        min_distance_frames = max(1, int(self.min_between_shots_sec * fps))
        valleys, _ = find_peaks(
            -knee_curve,
            prominence=self.min_knee_drop_deg,
            distance=min_distance_frames,
        )

        segments: List[ShotSegment] = []
        for valley_idx in valleys:
            pre_frames    = int(self.pre_shot_sec * fps)
            start_frame   = max(0, valley_idx - pre_frames)

            # Search for the wrist peak (highest point = smallest Y in image coords)
            search_limit  = min(n, valley_idx + int(3.0 * fps))
            wrist_window  = wrist_y[valley_idx:search_limit]
            if len(wrist_window) > 0:
                peak_offset   = int(np.argmin(wrist_window))
                release_frame = valley_idx + peak_offset
            else:
                release_frame = valley_idx + int(1.0 * fps)

            follow_frames = int(self.followthrough_sec * fps)
            end_frame     = min(n, release_frame + follow_frames)

            duration_sec  = (end_frame - start_frame) / fps
            if self.min_shot_sec <= duration_sec <= self.max_shot_sec:
                segments.append(ShotSegment(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_sec=start_frame / fps,
                    end_sec=end_frame / fps,
                    duration_sec=duration_sec,
                ))

        return _merge_overlapping(segments)

    def slice_sequence(self, seq: ShotSequence, seg: ShotSegment) -> ShotSequence:
        """Extract the frames of a ShotSegment from a full ShotSequence."""
        sliced = ShotSequence(fps=seq.fps)
        sliced.frames = seq.frames[seg.start_frame:seg.end_frame]
        return sliced


# ── Thumbnail helper ──────────────────────────────────────────────────────────

def _attach_thumbnails(
    video_path: str,
    seq: ShotSequence,
    segments: List[ShotSegment],
    thumb_width: int = 160,
) -> None:
    """Read one frame per segment (near the midpoint) and attach as thumbnail.
    Operates in-place on the segment list."""
    if not segments:
        return

    cap = cv2.VideoCapture(video_path)
    for seg in segments:
        mid_frame = (seg.start_frame + seg.end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            scale = thumb_width / w
            thumb = cv2.resize(frame, (thumb_width, int(h * scale)))
            seg.thumbnail = thumb
    cap.release()
