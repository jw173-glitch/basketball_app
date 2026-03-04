"""
app.py  ──  Basketball Shot Consistency Analyzer
Run with: streamlit run app.py

Key design: ALL expensive work (pose extraction, scoring) is stored in
st.session_state so that slider interactions never trigger a re-run of
the heavy pipeline.
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

sys.path.insert(0, os.path.dirname(__file__))

from core.pose_extractor import PoseExtractor, ShotSequence
from core.consistency_scorer import ConsistencyScorer
from utils.visualizer import (
    plot_angle_curves,
    plot_radar,
    plot_score_card,
    plot_joint_bars,
    export_skeleton_frames,
    fig_to_bytes,
    JOINT_LABELS,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏀 Shot Consistency Analyzer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0F0F1A; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #F0A500; }
    .stMetric label { color: #AAAAAA !important; }
    .stMetric div  { color: #FFFFFF !important; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
# Persists heavy results across every slider drag / page re-run.
defaults = {
    "ref_seq":           None,
    "user_seq":          None,
    "videos_processed":  False,
    "last_ref_name":     None,
    "last_user_name":    None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_extractor():
    return PoseExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def run_pose_extraction(extractor, path: str, label: str) -> ShotSequence:
    bar = st.progress(0, text=f"Processing {label}…")
    seq = extractor.process_video(path, annotate=True, max_frames=300)
    bar.progress(100, text=f"✅ {label} done — {len(seq.frames)} frames detected")
    time.sleep(0.3)
    bar.empty()
    return seq


def trim_sequence(seq: ShotSequence, start_pct: float, end_pct: float) -> ShotSequence:
    n = len(seq.frames)
    start_idx = int(start_pct / 100 * n)
    end_idx   = int(end_pct   / 100 * n)
    trimmed = ShotSequence(fps=seq.fps)
    trimmed.frames = seq.frames[start_idx:end_idx]
    return trimmed


def frame_to_jpeg_bytes(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Basketball.png/240px-Basketball.png",
        width=80,
    )
    st.title("🏀 Shot Analyzer")
    st.markdown("---")

    st.subheader("📁 Upload Videos")
    ref_file  = st.file_uploader("Reference shot (pro / standard form)", type=["mp4", "mov", "avi", "mkv"])
    user_file = st.file_uploader("Your shot",                            type=["mp4", "mov", "avi", "mkv"])

    st.markdown("---")
    st.subheader("⚙️ Settings")
    dtw_scale = st.slider(
        "DTW Sensitivity", 50, 500, 200, 50,
        help="Lower = stricter scoring. Higher = more lenient.",
    )

    ALL_JOINTS = list(JOINT_LABELS.keys())
    DEFAULT_JOINTS = [
        "right_elbow", "right_shoulder", "right_knee", "right_hip",
        "left_elbow", "left_shoulder", "left_knee", "left_hip",
    ]
    # Default raw weights (before normalization)
    DEFAULT_WEIGHTS_RAW = {
        "right_elbow":    25,
        "right_shoulder": 20,
        "right_knee":     20,
        "right_hip":      15,
        "left_elbow":     10,
        "left_shoulder":   5,
        "left_knee":       5,
        "left_hip":        0,
    }

    st.markdown("**🦴 Joints & Weights**")
    st.caption("Uncheck joints that weren't detected. Drag sliders to adjust importance.")

    active_joints  = []
    raw_weights    = {}

    for joint in ALL_JOINTS:
        checked = st.checkbox(
            JOINT_LABELS[joint],
            value=(joint in DEFAULT_JOINTS),
            key=f"joint_cb_{joint}",
        )
        if checked:
            active_joints.append(joint)
            raw_weights[joint] = st.slider(
                f"Weight",
                min_value=1, max_value=100,
                value=DEFAULT_WEIGHTS_RAW.get(joint, 10),
                step=1,
                key=f"joint_w_{joint}",
                label_visibility="collapsed",
            )

    if not active_joints:
        st.warning("Select at least one joint.")

    # Show live normalized weights
    if active_joints:
        total_raw = sum(raw_weights.values())
        normalized = {j: raw_weights[j] / total_raw for j in active_joints}
        with st.expander("📊 Normalized weights", expanded=False):
            for j in active_joints:
                pct = normalized[j] * 100
                st.caption(f"{JOINT_LABELS[j]}: **{pct:.1f}%**")

    st.markdown("---")
    st.markdown("**📈 Angle Curves**")
    show_joints = st.multiselect(
        "Show curves for",
        options=active_joints if active_joints else ALL_JOINTS,
        default=[j for j in ["right_elbow", "right_shoulder", "right_knee"] if j in (active_joints or ALL_JOINTS)],
        format_func=lambda x: JOINT_LABELS.get(x, x),
    )

    # If the user swaps files, invalidate cached sequences
    ref_name  = ref_file.name  if ref_file  else None
    user_name = user_file.name if user_file else None
    if (ref_name  != st.session_state.last_ref_name or
        user_name != st.session_state.last_user_name):
        st.session_state.videos_processed = False
        st.session_state.ref_seq          = None
        st.session_state.user_seq         = None
        st.session_state.last_ref_name    = ref_name
        st.session_state.last_user_name   = user_name

    process_btn = st.button(
        "🚀 Load & Process Videos", type="primary", use_container_width=True,
        disabled=(ref_file is None or user_file is None),
    )

    if ref_file is None or user_file is None:
        st.info("Upload both videos to get started.")

    st.markdown("---")
    st.caption("v1.2 · MediaPipe + DTW")


# ── Landing page ──────────────────────────────────────────────────────────────
st.title("🏀 Basketball Shot Consistency Analyzer")
st.markdown("Upload a reference shot and your own shot to get a consistency score and joint-level feedback.")

if not st.session_state.videos_processed and not process_btn:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1**\n\nUpload a reference video (pro player or standard form)")
    c2.info("**Step 2**\n\nUpload your shooting video *(side/45° angle works best)*")
    c3.info("**Step 3**\n\nClick **Load & Process Videos**, trim if needed, then hit **Run Analysis**")
    st.stop()


# ── Pose extraction — runs ONCE, result cached in session_state ───────────────
if process_btn and not st.session_state.videos_processed:
    extractor = load_extractor()

    with st.spinner("Saving uploaded files…"):
        ref_path  = save_upload(ref_file)
        user_path = save_upload(user_file)

    col_r, col_u = st.columns(2)
    with col_r:
        ref_seq  = run_pose_extraction(extractor, ref_path,  "Reference video")
    with col_u:
        user_seq = run_pose_extraction(extractor, user_path, "Your video")

    try:
        os.unlink(ref_path)
        os.unlink(user_path)
    except Exception:
        pass

    if len(ref_seq.frames) < 5 or len(user_seq.frames) < 5:
        st.error("⚠️ Not enough pose frames detected. Make sure your full body is visible.")
        st.stop()

    # ✅ Store in session_state — survives all future re-runs
    st.session_state.ref_seq          = ref_seq
    st.session_state.user_seq         = user_seq
    st.session_state.videos_processed = True

# Safety guard
if not st.session_state.videos_processed:
    st.stop()

# Pull from session_state — these are always available from here on
ref_seq  = st.session_state.ref_seq
user_seq = st.session_state.user_seq


# ── Trim section ──────────────────────────────────────────────────────────────
# This section re-runs freely on every slider drag — it's fast (no ML work).
st.markdown("---")
st.header("✂️ Trim to Shooting Action")
st.markdown(
    "Drag the sliders to isolate **only the shooting motion** — from the start of the dip "
    "to the end of follow-through. The frame preview updates live as you drag."
)

def show_trim_ui(seq: ShotSequence, label: str, slider_key: str):
    """Render slider + live start/end frame previews. Returns (start_pct, end_pct)."""
    total = len(seq.frames)
    st.subheader(label)

    trim_range = st.slider(
        f"Keep frames (%) — {label}",
        min_value=0, max_value=100,
        value=(0, 100), step=1,
        key=slider_key,
        help=f"Total frames detected: {total}",
    )

    start_idx = max(0,         int(trim_range[0] / 100 * total))
    end_idx   = min(total - 1, int(trim_range[1] / 100 * total) - 1)
    kept      = max(0, end_idx - start_idx + 1)

    st.caption(f"Frame **{start_idx}** → **{end_idx}** of {total}  |  **{kept} frames selected**")

    prev_l, prev_r = st.columns(2)
    with prev_l:
        st.markdown("**▶ Start frame**")
        sf = seq.frames[start_idx]
        if sf.image is not None:
            st.image(frame_to_jpeg_bytes(sf.image),
                     caption=f"Frame {start_idx}", use_container_width=True)
        else:
            st.info("No skeleton frame available")
    with prev_r:
        st.markdown("**⏹ End frame**")
        ef = seq.frames[end_idx]
        if ef.image is not None:
            st.image(frame_to_jpeg_bytes(ef.image),
                     caption=f"Frame {end_idx}", use_container_width=True)
        else:
            st.info("No skeleton frame available")

    return trim_range

trim_col1, trim_col2 = st.columns(2)
with trim_col1:
    ref_trim  = show_trim_ui(ref_seq,  "Reference Video", "ref_trim")
with trim_col2:
    user_trim = show_trim_ui(user_seq, "Your Video",      "user_trim")

ref_seq_trimmed  = trim_sequence(ref_seq,  *ref_trim)
user_seq_trimmed = trim_sequence(user_seq, *user_trim)

if len(ref_seq_trimmed.frames) < 5 or len(user_seq_trimmed.frames) < 5:
    st.warning("⚠️ Selection is too short (< 5 frames). Please widen the slider range.")
    st.stop()

st.success(
    f"✅ Reference: **{len(ref_seq_trimmed.frames)} frames** selected  |  "
    f"Your shot: **{len(user_seq_trimmed.frames)} frames** selected"
)


# ── Run Analysis button ───────────────────────────────────────────────────────
st.markdown("---")
run_analysis = st.button("📊 Run Analysis", type="primary")

if not run_analysis:
    st.info("Happy with the trim? Click **Run Analysis** to get your consistency score.")
    st.stop()


# ── Scoring ───────────────────────────────────────────────────────────────────
# Normalize the user's raw slider values to sum to 1.0
from core.consistency_scorer import JOINT_WEIGHTS as DEFAULT_WEIGHTS

if active_joints:
    total_raw      = sum(raw_weights.values())
    custom_weights = {j: raw_weights[j] / total_raw for j in active_joints}
else:
    custom_weights = None

scorer = ConsistencyScorer(joint_weights=custom_weights, dtw_scale=dtw_scale)
with st.spinner("Calculating consistency score…"):
    report = scorer.compare(ref_seq_trimmed, user_seq_trimmed)


# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("📊 Results")

col_score, col_metrics = st.columns([1, 2])
with col_score:
    st.image(fig_to_bytes(plot_score_card(report)), use_container_width=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Score",   f"{report.overall_score:.1f} / 100")
    m2.metric("Grade",           report.grade)
    m3.metric("Frames Analyzed", len(user_seq_trimmed.frames))

    m4, m5 = st.columns(2)
    m4.metric("Most Inconsistent Joint", JOINT_LABELS.get(report.most_inconsistent_joint, ""))
    m5.metric("Worst Phase",             report.most_inconsistent_phase)

    st.markdown("**📝 Feedback**")
    for fb in report.feedback:
        st.markdown(f"- {fb}")

st.markdown("---")

col_radar, col_bar = st.columns(2)
with col_radar:
    st.subheader("Joint Consistency Radar")
    st.image(fig_to_bytes(plot_radar(report)), use_container_width=True)
with col_bar:
    st.subheader("Joint Scores")
    st.image(fig_to_bytes(plot_joint_bars(report)), use_container_width=True)

st.markdown("---")

st.subheader("📈 Joint Angle Comparison")
if show_joints:
    fig_curves = plot_angle_curves(ref_seq_trimmed, user_seq_trimmed, joints=show_joints)
    st.image(fig_to_bytes(fig_curves), use_container_width=True)
else:
    st.info("Select joints in the sidebar to display angle curves.")

st.markdown("---")

with st.expander("🎥 Skeleton Overlay Video (optional — slow to generate)", expanded=False):
    if st.button("Generate skeleton video"):
        with st.spinner("Rendering…"):
            out_path = tempfile.mktemp(suffix=".mp4")
            result   = export_skeleton_frames(ref_seq_trimmed, user_seq_trimmed, out_path, max_frames=60)
        if result:
            with open(result, "rb") as f:
                st.video(f.read())
        else:
            st.warning("Not enough annotated frames to generate video.")

st.markdown("---")

with st.expander("📋 Detailed Joint Score Table", expanded=False):
    import pandas as pd
    rows = []
    for js in report.joint_scores:
        if js.weight > 0:
            rows.append({
                "Joint":    JOINT_LABELS.get(js.joint, js.joint),
                "Score":    f"{js.score:.1f}",
                "DTW Dist": f"{js.dtw_distance:.1f}",
                "Weight":   f"{js.weight:.0%}",
                "Status":   "⚠️ Needs work" if js.is_most_inconsistent else ("✅ Good" if js.score >= 75 else "📌 Watch"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)