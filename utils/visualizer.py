"""
visualizer.py
-------------
Generate all visualization charts: angle curves, joint radar, skeleton frame animation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import io

from core.pose_extractor import ShotSequence
from core.consistency_scorer import ConsistencyReport, JointScore


# Color scheme
COLOR_REF  = "#4A90D9"   # Blue = Reference
COLOR_USER = "#E84855"   # Red = User
COLOR_OK   = "#2ECC71"   # Green = Good
COLOR_WARN = "#F39C12"   # Orange = Warning
COLOR_BAD  = "#E74C3C"   # Red = Poor

JOINT_CN = {
    "right_elbow":    "R. Elbow",
    "left_elbow":     "L. Elbow",
    "right_shoulder": "R. Shoulder",
    "left_shoulder":  "L. Shoulder",
    "right_knee":     "R. Knee",
    "left_knee":      "L. Knee",
    "right_hip":      "R. Hip",
    "left_hip":       "L. Hip",
}


def _score_color(score: float) -> str:
    if score >= 75:
        return COLOR_OK
    elif score >= 55:
        return COLOR_WARN
    else:
        return COLOR_BAD


# ── 1. Angle Comparison Curves ───────────────────────────────────────────────────

def plot_angle_curves(
    ref: ShotSequence,
    user: ShotSequence,
    joints: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot reference vs user angle time curves.
    """
    if joints is None:
        joints = ["right_elbow", "right_shoulder", "right_knee", "right_hip"]

    n_joints = len(joints)
    fig, axes = plt.subplots(n_joints, 1, figsize=figsize, sharex=False)
    if n_joints == 1:
        axes = [axes]

    fig.patch.set_facecolor("#1A1A2E")

    for ax, joint in zip(axes, joints):
        ref_series  = ref.angle_series(joint)
        user_series = user.angle_series(joint)

        ref_t  = np.linspace(0, 100, len(ref_series))
        user_t = np.linspace(0, 100, len(user_series))

        ax.set_facecolor("#16213E")
        ax.plot(ref_t,  ref_series,  color=COLOR_REF,  lw=2.0, label="Reference", alpha=0.9)
        ax.plot(user_t, user_series, color=COLOR_USER, lw=2.0, label="Your Shot", alpha=0.9, linestyle="--")
        ax.fill_between(ref_t, ref_series, alpha=0.1, color=COLOR_REF)

        ax.set_ylabel(f"{JOINT_CN.get(joint, joint)}\nAngle (°)", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#444")
        ax.grid(alpha=0.2, color="white")

        if joint == joints[0]:
            ax.legend(loc="upper right", facecolor="#1A1A2E", labelcolor="white", fontsize=8)

    axes[-1].set_xlabel("Motion Progress (%)", color="white", fontsize=9)
    fig.suptitle("Joint Angle Comparison", color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ── 2. Joint Score Radar Chart ───────────────────────────────────────────────

def plot_radar(report: ConsistencyReport, figsize: Tuple[int, int] = (6, 6)) -> Figure:
    """Plot a radar chart of per-joint scores."""
    js_list = [js for js in report.joint_scores if js.weight > 0]
    labels  = [JOINT_CN.get(js.joint, js.joint) for js in js_list]
    scores  = [js.score for js in js_list]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    scores_plot = scores + scores[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    ax.plot(angles, scores_plot, color=COLOR_USER, lw=2)
    ax.fill(angles, scores_plot, color=COLOR_USER, alpha=0.25)

    # Reference line (100 pts)
    ax.plot(angles, [100] * (N + 1), color=COLOR_REF, lw=1.5, linestyle="--", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="gray", fontsize=7)
    ax.spines["polar"].set_color("#444")
    ax.grid(color="#444", alpha=0.5)

    ax.set_title("Per-Joint Consistency", color="white", fontsize=12, fontweight="bold", pad=15)
    return fig


# ── 3. Overall Score Card ───────────────────────────────────────────────────

def plot_score_card(report: ConsistencyReport, figsize: Tuple[int, int] = (6, 4)) -> Figure:
    """Display overall score and grade in large text."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.axis("off")

    color = _score_color(report.overall_score)

    ax.text(0.5, 0.72, f"{report.overall_score:.1f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=72, fontweight="bold", color=color)

    ax.text(0.5, 0.42, f"Grade {report.grade}", transform=ax.transAxes,
            ha="center", va="center", fontsize=24, color="white", alpha=0.85)

    ax.text(0.5, 0.20, f"Most unstable joint: {JOINT_CN.get(report.most_inconsistent_joint, '')}  |  Worst phase: {report.most_inconsistent_phase}",
            transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#AAAAAA")

    return fig


# ── 4. Joint Score Bar Chart ─────────────────────────────────────────────────────

def plot_joint_bars(report: ConsistencyReport, figsize: Tuple[int, int] = (8, 4)) -> Figure:
    js_list = sorted([js for js in report.joint_scores if js.weight > 0],
                     key=lambda x: x.score, reverse=True)
    labels = [JOINT_CN.get(js.joint, js.joint) for js in js_list]
    scores = [js.score for js in js_list]
    colors = [_score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    bars = ax.barh(labels, scores, color=colors, edgecolor="none", height=0.55)
    ax.set_xlim(0, 110)
    ax.axvline(x=75, color="white", linestyle="--", alpha=0.3, lw=1)
    ax.set_xlabel("Score", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", color="white", fontsize=9)

    ax.set_title("Per-Joint Scores", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ── 5. Export Skeleton Comparison Video Frames (GIF) ──────────────────────

def export_skeleton_frames(
    ref: ShotSequence,
    user: ShotSequence,
    output_path: str,
    max_frames: int = 60,
) -> str:
    """
    Export reference vs user skeleton-annotated frames side by side, saved as MP4.
    Returns the output path.
    """
    ref_frames  = [f.image for f in ref.frames  if f.image is not None][:max_frames]
    user_frames = [f.image for f in user.frames if f.image is not None][:max_frames]

    if not ref_frames or not user_frames:
        return ""

    h = max(ref_frames[0].shape[0], user_frames[0].shape[0])
    w = ref_frames[0].shape[1] + user_frames[0].shape[1] + 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, 15, (w, h))

    n = min(len(ref_frames), len(user_frames))
    for i in range(n):
        rf = cv2.resize(ref_frames[i],  (ref_frames[0].shape[1],  h))
        uf = cv2.resize(user_frames[i], (user_frames[0].shape[1], h))
        sep = np.zeros((h, 10, 3), dtype=np.uint8)

        # Labels
        cv2.putText(rf, "Reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(uf, "Your Shot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)

        combined = np.hstack([rf, sep, uf])
        out.write(combined)

    out.release()
    return output_path


# ── Utility: Figure → bytes (for Streamlit) ───────────────────────────────────

def fig_to_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()
