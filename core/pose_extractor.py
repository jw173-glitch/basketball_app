"""
pose_extractor.py
-----------------
Extract joint coordinates and joint angles frame-by-frame from video using MediaPipe Pose.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame_idx: int
    landmarks: Dict[str, Tuple[float, float, float]]   # name → (x, y, z)
    angles: Dict[str, float]                            # joint_name → degrees
    image: Optional[np.ndarray] = None                 # annotated frame (BGR)


@dataclass
class ShotSequence:
    """Frame sequence of a complete shooting motion."""
    frames: List[FrameData] = field(default_factory=list)
    fps: float = 30.0

    # Convenience accessor: angle time series
    def angle_series(self, joint: str) -> np.ndarray:
        return np.array([f.angles.get(joint, np.nan) for f in self.frames])

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.fps


# ── Key Joint Definitions ─────────────────────────────────────────────────────

MP_POSE = mp.solutions.pose.PoseLandmark

# Joint triplets used for shot analysis (vertex in the middle)
ANGLE_DEFINITIONS = {
    "right_elbow":   (MP_POSE.RIGHT_SHOULDER, MP_POSE.RIGHT_ELBOW,   MP_POSE.RIGHT_WRIST),
    "left_elbow":    (MP_POSE.LEFT_SHOULDER,  MP_POSE.LEFT_ELBOW,    MP_POSE.LEFT_WRIST),
    "right_shoulder":(MP_POSE.RIGHT_ELBOW,    MP_POSE.RIGHT_SHOULDER, MP_POSE.RIGHT_HIP),
    "left_shoulder": (MP_POSE.LEFT_ELBOW,     MP_POSE.LEFT_SHOULDER,  MP_POSE.LEFT_HIP),
    "right_knee":    (MP_POSE.RIGHT_HIP,      MP_POSE.RIGHT_KNEE,    MP_POSE.RIGHT_ANKLE),
    "left_knee":     (MP_POSE.LEFT_HIP,       MP_POSE.LEFT_KNEE,     MP_POSE.LEFT_ANKLE),
    "right_hip":     (MP_POSE.RIGHT_SHOULDER, MP_POSE.RIGHT_HIP,     MP_POSE.RIGHT_KNEE),
    "left_hip":      (MP_POSE.LEFT_SHOULDER,  MP_POSE.LEFT_HIP,      MP_POSE.LEFT_KNEE),
}

LANDMARK_NAMES = [lm.name.lower() for lm in MP_POSE]


# ── Utility Functions ─────────────────────────────────────────────────────────

def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate the angle at vertex b formed by points a-b-c. Returns degrees 0-180."""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _lm_to_array(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


# ── Main Class ────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    Usage:
        extractor = PoseExtractor()
        sequence  = extractor.process_video("shot.mp4", annotate=True)
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence

    # ── Public API ────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        max_frames: int = 300,
    ) -> ShotSequence:
        """
        Extract pose data from all frames in a video file.

        Parameters
        ----------
        video_path : Path to the video file
        annotate   : Whether to draw skeleton overlay on frames (for visualization)
        max_frames : Maximum number of frames to process (prevents long videos)

        Returns
        -------
        ShotSequence
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sequence = ShotSequence(fps=fps)

        with self.mp_pose.Pose(
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        ) as pose:
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

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, annotate: bool = True) -> Optional[FrameData]:
        """Process a single image frame (for real-time use)."""
        with self.mp_pose.Pose(
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        ) as pose:
            return self._process_frame(frame, pose, frame_idx, annotate)

    # ── Internal Methods ────────────────────────────────────────────────────────

    def _process_frame(self, frame, pose, idx, annotate) -> Optional[FrameData]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks.landmark

        # Joint coordinates
        landmarks = {
            LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z)
            for i, lm in enumerate(lms)
        }

        # Joint angles
        angles = {}
        for joint_name, (a_lm, b_lm, c_lm) in ANGLE_DEFINITIONS.items():
            a = _lm_to_array(lms[a_lm.value])
            b = _lm_to_array(lms[b_lm.value])
            c = _lm_to_array(lms[c_lm.value])
            angles[joint_name] = _calc_angle(a, b, c)

        # Visualization overlay
        annotated = None
        if annotate:
            annotated = frame.copy()
            self.mp_draw.draw_landmarks(
                annotated,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2),
            )

        return FrameData(frame_idx=idx, landmarks=landmarks, angles=angles, image=annotated)
