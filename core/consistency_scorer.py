"""
consistency_scorer.py
----------------------
基于 DTW（动态时间规整）计算两段投篮动作的一致性分数。
支持：单次对比、多次对比取平均、关节级别细粒度分析。
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from core.pose_extractor import ShotSequence


# ── 配置：各关节权重 ───────────────────────────────────────────────────────────

JOINT_WEIGHTS: Dict[str, float] = {
    "right_elbow":    0.25,   # 出手肘部最关键
    "right_shoulder": 0.20,
    "right_wrist":    0.00,   # MediaPipe 不直接给 wrist angle，先置0
    "right_knee":     0.20,
    "right_hip":      0.15,
    "left_elbow":     0.10,
    "left_shoulder":  0.05,
    "left_knee":      0.05,
}


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class JointScore:
    joint: str
    dtw_distance: float       # 原始 DTW 距离（越小越好）
    score: float              # 归一化后 0-100（越高越好）
    weight: float
    is_most_inconsistent: bool = False


@dataclass
class ConsistencyReport:
    overall_score: float                  # 0-100
    joint_scores: List[JointScore]
    most_inconsistent_joint: str
    most_inconsistent_phase: str          # "起跳" / "上升" / "出手" / "随球"
    feedback: List[str]                   # 自然语言反馈列表

    @property
    def grade(self) -> str:
        if self.overall_score >= 85:
            return "A"
        elif self.overall_score >= 70:
            return "B"
        elif self.overall_score >= 55:
            return "C"
        else:
            return "D"


# ── DTW 核心 ──────────────────────────────────────────────────────────────────

def _dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """标准 DTW 距离，处理 NaN。"""
    # 替换 NaN 为线性插值
    s1 = _fill_nan(s1)
    s2 = _fill_nan(s2)

    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return np.zeros_like(arr)
    indices = np.arange(len(arr))
    arr[nan_mask] = np.interp(indices[nan_mask], indices[~nan_mask], arr[~nan_mask])
    return arr


def _normalize_score(dtw_dist: float, scale: float = 200.0) -> float:
    """将 DTW 距离转换为 0-100 分（指数衰减）。"""
    return float(100.0 * np.exp(-dtw_dist / scale))


# ── 投篮阶段检测 ──────────────────────────────────────────────────────────────

def detect_shot_phases(seq: ShotSequence) -> Dict[str, Tuple[int, int]]:
    """
    简单地把序列四等分为四个阶段。
    进阶版可以用膝盖角度曲线检测起跳点。
    """
    n = len(seq.frames)
    q = n // 4
    return {
        "起跳": (0, q),
        "上升": (q, 2 * q),
        "出手": (2 * q, 3 * q),
        "随球": (3 * q, n),
    }


def find_most_inconsistent_phase(
    ref: ShotSequence,
    user: ShotSequence,
    joint: str,
) -> str:
    """找出最不一致的阶段。"""
    phases = detect_shot_phases(user)
    worst_phase, worst_dist = "出手", -1.0

    ref_series = ref.angle_series(joint)
    user_series = user.angle_series(joint)

    for phase_name, (start, end) in phases.items():
        # 按比例切 ref
        r_start = int(start / len(user.frames) * len(ref.frames))
        r_end = int(end / len(user.frames) * len(ref.frames))
        dist = _dtw_distance(ref_series[r_start:r_end], user_series[start:end])
        if dist > worst_dist:
            worst_dist = dist
            worst_phase = phase_name

    return worst_phase


# ── 主类 ──────────────────────────────────────────────────────────────────────

class ConsistencyScorer:
    """
    用法：
        scorer = ConsistencyScorer()
        report = scorer.compare(reference_sequence, user_sequence)
    """

    def __init__(self, joint_weights: Dict[str, float] = None, dtw_scale: float = 200.0):
        self.weights = joint_weights or JOINT_WEIGHTS
        self.dtw_scale = dtw_scale

    def compare(self, ref: ShotSequence, user: ShotSequence) -> ConsistencyReport:
        """
        对比参考动作和用户动作，生成完整报告。
        """
        joint_scores = []
        active_joints = [j for j, w in self.weights.items() if w > 0]

        for joint in active_joints:
            ref_series = ref.angle_series(joint)
            user_series = user.angle_series(joint)
            dist = _dtw_distance(ref_series, user_series)
            score = _normalize_score(dist, self.dtw_scale)
            joint_scores.append(JointScore(
                joint=joint,
                dtw_distance=dist,
                score=score,
                weight=self.weights[joint],
            ))

        # 加权总分
        total_w = sum(js.weight for js in joint_scores)
        overall = sum(js.score * js.weight for js in joint_scores) / (total_w + 1e-8)

        # 最差关节
        worst = min(joint_scores, key=lambda js: js.score)
        worst.is_most_inconsistent = True

        # 最差阶段
        worst_phase = find_most_inconsistent_phase(ref, user, worst.joint)

        # 生成反馈
        feedback = self._generate_feedback(joint_scores, worst, worst_phase)

        return ConsistencyReport(
            overall_score=round(overall, 1),
            joint_scores=sorted(joint_scores, key=lambda js: js.score),
            most_inconsistent_joint=worst.joint,
            most_inconsistent_phase=worst_phase,
            feedback=feedback,
        )

    def compare_multiple(
        self,
        shots: List[ShotSequence],
    ) -> Tuple[ConsistencyReport, List[ConsistencyReport]]:
        """
        对多次投篮互相比较，返回综合报告和逐对报告。
        进阶功能：分析球员自身动作稳定性。
        """
        if len(shots) < 2:
            raise ValueError("至少需要2次投篮才能比较")

        pair_reports = []
        for i in range(len(shots) - 1):
            report = self.compare(shots[i], shots[i + 1])
            pair_reports.append(report)

        # 取平均分作为综合报告（简化版）
        avg_score = np.mean([r.overall_score for r in pair_reports])
        worst_joint_counts: Dict[str, int] = {}
        for r in pair_reports:
            worst_joint_counts[r.most_inconsistent_joint] = \
                worst_joint_counts.get(r.most_inconsistent_joint, 0) + 1
        most_common_worst = max(worst_joint_counts, key=worst_joint_counts.get)

        summary = ConsistencyReport(
            overall_score=round(float(avg_score), 1),
            joint_scores=pair_reports[0].joint_scores,   # 用第一次的关节分
            most_inconsistent_joint=most_common_worst,
            most_inconsistent_phase=pair_reports[0].most_inconsistent_phase,
            feedback=[f"综合 {len(shots)} 次投篮分析：平均一致性 {avg_score:.1f} 分"],
        )
        return summary, pair_reports

    # ── 反馈文本生成 ──────────────────────────────────────────────────────────

    def _generate_feedback(
        self,
        joint_scores: List[JointScore],
        worst: JointScore,
        worst_phase: str,
    ) -> List[str]:
        feedback = []
        joint_cn = {
            "right_elbow": "右肘",
            "left_elbow": "左肘",
            "right_shoulder": "右肩",
            "left_shoulder": "左肩",
            "right_knee": "右膝",
            "left_knee": "左膝",
            "right_hip": "右髋",
            "left_hip": "左髋",
        }

        # 整体评价
        avg = np.mean([js.score for js in joint_scores])
        if avg >= 85:
            feedback.append("✅ 整体动作非常稳定，继续保持！")
        elif avg >= 70:
            feedback.append("👍 整体动作较好，有小幅提升空间。")
        elif avg >= 55:
            feedback.append("⚠️ 动作存在一定波动，需要针对性练习。")
        else:
            feedback.append("❌ 动作一致性较低，建议回到基本功训练。")

        # 最差关节
        cn_name = joint_cn.get(worst.joint, worst.joint)
        feedback.append(f"🎯 最不稳定的关节是「{cn_name}」，在「{worst_phase}」阶段偏差最大。")

        # 关节级别建议
        for js in joint_scores:
            if js.score < 60:
                cn = joint_cn.get(js.joint, js.joint)
                feedback.append(f"📌 {cn} 一致性仅 {js.score:.0f} 分，需要重点关注。")

        return feedback
