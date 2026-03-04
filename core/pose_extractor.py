"""
pose_extractor.py
-----------------
使用 MediaPipe Pose 从视频中逐帧提取关节坐标和关节角度。
兼容 MediaPipe 新版（0.10+）和旧版（<0.10）API。
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import mediapipe as mp

# ── 版本兼容：统一导入 ─────────────────────────────────────────────────────────
# MediaPipe 0.10+ 把 solutions 移到 mediapipe.python.solutions
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


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame_idx: int
    landmarks: Dict[str, Tuple[float, float, float]]   # name → (x, y, z)
    angles: Dict[str, float]                            # joint_name → degrees
    image: Optional[np.ndarray] = None                 # annotated frame (BGR)


@dataclass
class ShotSequence:
    """一次完整投篮动作的帧序列"""
    frames: List[FrameData] = field(default_factory=list)
    fps: float = 30.0

    def angle_series(self, joint: str) -> np.ndarray:
        return np.array([f.angles.get(joint, np.nan) for f in self.frames])

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.fps


# ── 关键关节定义（全部用整数索引，避免枚举API差异）────────────────────────────

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


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算 a-b-c 三点夹角（b 为顶点），返回度数 0-180。"""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


# ── 主类 ──────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    用法：
        extractor = PoseExtractor()
        sequence  = extractor.process_video("shot.mp4", annotate=True)
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence

    def _make_pose(self):
        return mp_pose_module.Pose(
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        )

    # ── 公开 API ───────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        max_frames: int = 300,
    ) -> ShotSequence:
        """从视频文件提取所有帧的姿态数据。"""
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

    # ── 内部方法 ───────────────────────────────────────────────────────────────

    def _process_frame(self, frame, pose, idx, annotate) -> Optional[FrameData]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks.landmark

        # 关节坐标
        landmarks = {
            LANDMARK_NAMES[i]: (lm.x, lm.y, lm.z)
            for i, lm in enumerate(lms)
            if i < len(LANDMARK_NAMES)
        }

        # 关节角度（整数索引，兼容所有版本）
        angles = {}
        for joint_name, (ai, bi, ci) in ANGLE_DEFINITIONS.items():
            try:
                a = np.array([lms[ai].x, lms[ai].y, lms[ai].z])
                b = np.array([lms[bi].x, lms[bi].y, lms[bi].z])
                c = np.array([lms[ci].x, lms[ci].y, lms[ci].z])
                angles[joint_name] = _calc_angle(a, b, c)
            except (IndexError, AttributeError):
                angles[joint_name] = float("nan")

        # 可视化标注
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
                pass  # 标注失败不影响数据提取

        return FrameData(frame_idx=idx, landmarks=landmarks, angles=angles, image=annotated)
