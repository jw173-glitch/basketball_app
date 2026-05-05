"""
pose_extractor.py
-----------------
Extract joint coordinates and joint angles frame-by-frame from video using MediaPipe Pose.
Compatible with MediaPipe new API (0.10+) and legacy API (<0.10).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import mediapipe as mp

# ── Version compatibility: unified import ─────────────────────────────────────
# MediaPipe 0.10+ moved solutions to mediapipe.python.solutions
try:
    from mediapipe.python.solutions import pose as mp_pose_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    try:
        from mediapipe.solutions import pose as mp_pose_module          # type: ignore
        from mediapipe.solutions import drawing_utils as mp_drawing     # type: ignore
    except ImportError:
        mp_pose_module = mp.solutions.pose                              # type: ignore
        mp_drawing     = mp.solutions.drawing_utils                     # type: ignore


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame_idx: int
    landmarks: Dict[str, Tuple[float, float, float]]        # name → (x, y, z) normalized image coords
    angles: Dict[str, float]                                # joint_name → degrees
    image: Optional[np.ndarray] = None                      # annotated frame (BGR)
    world_landmarks: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # world_landmarks: real-world coords in metres, origin = hip centre.
    # Y axis points downward, so height_above_hip = -world_y.


@dataclass
class ShotSequence:
    """Frame sequence of a complete shooting motion."""
    frames: List[FrameData] = field(default_factory=list)
    fps: float = 30.0

    def angle_series(self, joint: str) -> np.ndarray:
        """Return the angle time series for a given joint across all frames as a NumPy array.
        Frames where the joint was not detected are filled with NaN."""
        return np.array([f.angles.get(joint, np.nan) for f in self.frames])

    @property
    def duration_sec(self) -> float:
        """Total duration of the shot sequence in seconds, computed from frame count and FPS."""
        return len(self.frames) / self.fps


# ── Key joint definitions (all integer indices to avoid enum API differences) ─

LM = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow":    13,  "right_elbow":    14,
    "left_wrist":    15,  "right_wrist":    16,
    "left_hip":      23,  "right_hip":      24,
    "left_knee":     25,  "right_knee":     26,
    "left_ankle":    27,  "right_ankle":    28,
}

ANGLE_DEFINITIONS: Dict[str, Tuple[int, int, int]] = {
    "right_elbow":    (LM["right_shoulder"], LM["right_elbow"],    LM["right_wrist"]),
    "left_elbow":     (LM["left_shoulder"],  LM["left_elbow"],     LM["left_wrist"]),
    "right_shoulder": (LM["right_elbow"],    LM["right_shoulder"], LM["right_hip"]),
    "left_shoulder":  (LM["left_elbow"],     LM["left_shoulder"],  LM["left_hip"]),
    "right_knee":     (LM["right_hip"],      LM["right_knee"],     LM["right_ankle"]),
    "left_knee":      (LM["left_hip"],       LM["left_knee"],      LM["left_ankle"]),
    "right_hip":      (LM["right_shoulder"], LM["right_hip"],      LM["right_knee"]),
    "left_hip":       (LM["left_shoulder"],  LM["left_hip"],       LM["left_knee"]),
}

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


# ── Utility functions ─────────────────────────────────────────────────────────

def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at vertex b formed by points a-b-c, returning degrees in range 0-180."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


# ── Main class ────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Usage:
        extractor = PoseExtractor()
        sequence  = extractor.process_video("shot.mp4", annotate=True)
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """Initialize the extractor with MediaPipe confidence thresholds.
        Lower thresholds increase recall but may introduce noisy detections."""
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence

    def _make_pose(self):
        """Instantiate a MediaPipe Pose context manager using the stored confidence settings."""
        return mp_pose_module.Pose(
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        max_frames: int = 300,
    ) -> ShotSequence:
        """Extract pose data from all frames of a video file."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sequence = ShotSequence(fps=fps)

        with self._make_pose() as pose:
            idx = 0
            while cap.isOpened() and idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_data = self._process_frame(frame, pose, idx, annotate)
                if frame_data is not None:
                    sequence.frames.append(frame_data)
                idx += 1

        cap.release()
        return sequence

    # ── Internal methods ──────────────────────────────────────────────────────

    def _process_frame(self, frame, pose, idx, annotate) -> Optional[FrameData]:
        """Run MediaPipe on a single BGR frame, compute joint angles, and optionally draw landmarks.
        Returns None if no pose is detected in the frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks.landmark

        # Joint coordinates (normalized image coords)
        landmarks = {
            LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z)
            for i, lm in enumerate(lms)
            if i < len(LANDMARK_NAMES)
        }

        # World coordinates (metres, origin = hip centre, Y points down)
        world_landmarks: Dict[str, Tuple[float, float, float]] = {}
        if result.pose_world_landmarks:
            wlms = result.pose_world_landmarks.landmark
            world_landmarks = {
                LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z)
                for i, lm in enumerate(wlms)
                if i < len(LANDMARK_NAMES)
            }

        # Joint angles (integer indices, compatible with all versions)
        angles = {}
        for joint_name, (ai, bi, ci) in ANGLE_DEFINITIONS.items():
            try:
                a = np.array([lms[ai].x, lms[ai].y, lms[ai].z])
                b = np.array([lms[bi].x, lms[bi].y, lms[bi].z])
                c = np.array([lms[ci].x, lms[ci].y, lms[ci].z])
                angles[joint_name] = _calc_angle(a, b, c)
            except (IndexError, AttributeError):
                angles[joint_name] = float("nan")

        # Visualization annotation
        annotated = None
        if annotate:
            annotated = frame.copy()
            try:
                mp_drawing.draw_landmarks(
                    annotated,
                    result.pose_landmarks,
                    mp_pose_module.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
                )
            except Exception:
                pass  # annotation failure does not affect data extraction

        return FrameData(
            frame_idx=idx,
            landmarks=landmarks,
            angles=angles,
            image=annotated,
            world_landmarks=world_landmarks,
        )
