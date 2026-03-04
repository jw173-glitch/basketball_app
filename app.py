"""
app.py  ──  Basketball Shot Consistency Analyzer
Run with: streamlit run app.py
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from core.pose_extractor import PoseExtractor
from core.consistency_scorer import ConsistencyScorer
from utils.visualizer import (
    plot_angle_curves,
    plot_radar,
    plot_score_card,
    plot_joint_bars,
    export_skeleton_frames,
    fig_to_bytes,
    JOINT_CN,
)


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏀 Shot Consistency Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F0F1A; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #F0A500; }
    .score-big { font-size: 80px; font-weight: bold; text-align: center; }
    .stMetric label { color: #AAAAAA !important; }
    .stMetric div { color: #FFFFFF !important; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Cache: Initialize Models ─────────────────────────────────────────────────────
@st.cache_resource
def load_extractor():
    return PoseExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache_resource
def load_scorer():
    return ConsistencyScorer()


# ── Utility Functions ─────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    """Save an UploadedFile to a temp file and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def process_video_with_progress(extractor, path: str, label: str):
    """Process video and show progress bar."""
    bar = st.progress(0, text=f"Processing {label}…")
    seq = extractor.process_video(path, annotate=True, max_frames=200)
    bar.progress(100, text=f"✅ {label} done ({len(seq.frames)} frames)")
    time.sleep(0.3)
    bar.empty()
    return seq


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Basketball.png/240px-Basketball.png", width=80)
    st.title("🏀 Shot Analyzer")
    st.markdown("---")

    st.subheader("📁 Upload Videos")
    ref_file  = st.file_uploader("Reference shot (pro / standard)", type=["mp4", "mov", "avi", "mkv"])
    user_file = st.file_uploader("Your shooting video",             type=["mp4", "mov", "avi", "mkv"])

    st.markdown("---")
    st.subheader("⚙️ Analysis Settings")
    dtw_scale = st.slider("DTW Sensitivity", 50, 500, 200, 50,
                           help="Lower = stricter, Higher = more lenient")
    show_joints = st.multiselect(
        "Show joint curves",
        options=list(JOINT_CN.keys()),
        default=["right_elbow", "right_shoulder", "right_knee"],
        format_func=lambda x: JOINT_CN.get(x, x),
    )

    analyze_btn = st.button("🚀 Start Analysis", type="primary", use_container_width=True,
                            disabled=(ref_file is None or user_file is None))

    if ref_file is None or user_file is None:
        st.info("Please upload both videos first")

    st.markdown("---")
    st.caption("v1.0 · MediaPipe + DTW")


# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🏀 Basketball Shot Consistency Analyzer")
st.markdown("Upload a reference video and your shooting video to get joint angle comparison and consistency scores.")

if not analyze_btn:
    # Empty state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nUpload a reference shooting video (pro player / standard form)")
    with col2:
        st.info("**Step 2**\n\nUpload your shooting video (side angle at 45° works best)")
    with col3:
        st.info("**Step 3**\n\nClick 'Start Analysis' for scores and feedback")
    st.stop()


# ── Analysis Pipeline ──────────────────────────────────────────────────────────
extractor = load_extractor()
scorer    = ConsistencyScorer(dtw_scale=dtw_scale)

with st.spinner("Saving uploaded files…"):
    ref_path  = save_upload(ref_file)
    user_path = save_upload(user_file)

col_ref, col_user = st.columns(2)
with col_ref:
    ref_seq  = process_video_with_progress(extractor, ref_path,  "Reference video")
with col_user:
    user_seq = process_video_with_progress(extractor, user_path, "Your video")

if len(ref_seq.frames) < 5 or len(user_seq.frames) < 5:
    st.error("⚠️ Not enough pose frames detected. Make sure the video shows a full-body view.")
    st.stop()

with st.spinner("Calculating consistency scores…"):
    report = scorer.compare(ref_seq, user_seq)


# ── Results Display ─────────────────────────────────────────────────────────

st.markdown("---")
st.header("📊 Analysis Results")

# Row 1: Overall score + key metrics
col_score, col_metrics = st.columns([1, 2])

with col_score:
    fig_card = plot_score_card(report)
    st.image(fig_to_bytes(fig_card), use_container_width=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("Score",     f"{report.overall_score:.1f} / 100")
    m2.metric("Grade",     report.grade)
    m3.metric("Frames Analyzed", f"{len(user_seq.frames)} frames")

    m4, m5 = st.columns(2)
    m4.metric("Most Unstable Joint", JOINT_CN.get(report.most_inconsistent_joint, ""))
    m5.metric("Worst Phase",     report.most_inconsistent_phase)

    st.markdown("**📝 Analysis Feedback**")
    for fb in report.feedback:
        st.markdown(f"- {fb}")

st.markdown("---")

# Row 2: Radar + bar chart
col_radar, col_bar = st.columns(2)
with col_radar:
    st.subheader("Joint Radar Chart")
    fig_radar = plot_radar(report)
    st.image(fig_to_bytes(fig_radar), use_container_width=True)

with col_bar:
    st.subheader("Per-Joint Scores")
    fig_bar = plot_joint_bars(report)
    st.image(fig_to_bytes(fig_bar), use_container_width=True)

st.markdown("---")

# Row 3: Angle curves
st.subheader("📈 Joint Angle Comparison Curves")
if show_joints:
    fig_curves = plot_angle_curves(ref_seq, user_seq, joints=show_joints)
    st.image(fig_to_bytes(fig_curves), use_container_width=True)
else:
    st.info("Please select joints to display from the sidebar")

st.markdown("---")

# Row 4: Skeleton video (optional)
with st.expander("🎬 Skeleton Annotated Comparison Video (optional, slow to generate)", expanded=False):
    if st.button("Generate Skeleton Video"):
        with st.spinner("Generating…"):
            out_path = tempfile.mktemp(suffix=".mp4")
            result   = export_skeleton_frames(ref_seq, user_seq, out_path, max_frames=60)
        if result:
            with open(result, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Not enough skeleton frames to generate video.")

st.markdown("---")

# Joint score detail table
with st.expander("📋 Detailed Joint Score Table", expanded=False):
    import pandas as pd
    rows = []
    for js in report.joint_scores:
        if js.weight > 0:
            rows.append({
                "Joint":    JOINT_CN.get(js.joint, js.joint),
                "Score":    f"{js.score:.1f}",
                "DTW Dist.": f"{js.dtw_distance:.1f}",
                "Weight":    f"{js.weight:.0%}",
                "Status":    "⚠️ Needs work" if js.is_most_inconsistent else ("✅ Good" if js.score >= 75 else "📌 Attention"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Clean up temp files
try:
    os.unlink(ref_path)
    os.unlink(user_path)
except Exception:
    pass
