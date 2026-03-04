"""
pose_extractor.py
-----------------
使用 MediaPipe Pose 从视频中逐帧提取关节坐标和关节角度。
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


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

    # 便捷访问：角度时间序列
    def angle_series(self, joint: str) -> np.ndarray:
        return np.array([f.angles.get(joint, np.nan) for f in self.frames])

    @property
    def duration_sec(self) -> float:
        return len(self.frames) / self.fps


# ── 关键关节定义 ───────────────────────────────────────────────────────────────

MP_POSE = mp.solutions.pose.PoseLandmark

# 投篮分析用到的关节三元组 (顶点在中间)
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


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """计算 a-b-c 三点夹角（b 为顶点），返回度数 0-180。"""
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _lm_to_array(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


# ── 主类 ──────────────────────────────────────────────────────────────────────

class PoseExtractor:
    """
    用法：
        extractor = PoseExtractor()
        sequence  = extractor.process_video("shot.mp4", annotate=True)
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self._det_conf = min_detection_confidence
        self._trk_conf = min_tracking_confidence

    # ── 公开 API ───────────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        annotate: bool = True,
        max_frames: int = 300,
    ) -> ShotSequence:
        """
        从视频文件提取所有帧的姿态数据。

        Parameters
        ----------
        video_path : 视频文件路径
        annotate   : 是否在帧上画出骨骼（用于可视化）
        max_frames : 最多处理帧数（防止超长视频）

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
        """处理单帧图像（供实时使用）。"""
        with self.mp_pose.Pose(
            min_detection_confidence=self._det_conf,
            min_tracking_confidence=self._trk_conf,
        ) as pose:
            return self._process_frame(frame, pose, frame_idx, annotate)

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
        }

        # 关节角度
        angles = {}
        for joint_name, (a_lm, b_lm, c_lm) in ANGLE_DEFINITIONS.items():
            a = _lm_to_array(lms[a_lm.value])
            b = _lm_to_array(lms[b_lm.value])
            c = _lm_to_array(lms[c_lm.value])
            angles[joint_name] = _calc_angle(a, b, c)

        # 可视化标注
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
