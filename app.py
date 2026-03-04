"""
app.py  ──  篮球投篮一致性分析器
运行方式：streamlit run app.py
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

# ── 路径设置 ──────────────────────────────────────────────────────────────────
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


# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏀 投篮一致性分析器",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ────────────────────────────────────────────────────────────────
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


# ── 缓存：初始化模型 ──────────────────────────────────────────────────────────
@st.cache_resource
def load_extractor():
    return PoseExtractor(min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache_resource
def load_scorer():
    return ConsistencyScorer()


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def save_upload(uploaded_file) -> str:
    """把 UploadedFile 保存到临时文件，返回路径。"""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    return tmp.name


def process_video_with_progress(extractor, path: str, label: str):
    """处理视频并显示进度条。"""
    bar = st.progress(0, text=f"正在处理 {label}…")
    seq = extractor.process_video(path, annotate=True, max_frames=200)
    bar.progress(100, text=f"✅ {label} 处理完成（{len(seq.frames)} 帧）")
    time.sleep(0.3)
    bar.empty()
    return seq


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Basketball.png/240px-Basketball.png", width=80)
    st.title("🏀 投篮分析器")
    st.markdown("---")

    st.subheader("📁 上传视频")
    ref_file  = st.file_uploader("参考动作（标准/职业球员）", type=["mp4", "mov", "avi", "mkv"])
    user_file = st.file_uploader("你的投篮视频",             type=["mp4", "mov", "avi", "mkv"])

    st.markdown("---")
    st.subheader("⚙️ 分析设置")
    dtw_scale = st.slider("DTW 灵敏度", 50, 500, 200, 50,
                           help="越小=越严格，越大=越宽松")
    show_joints = st.multiselect(
        "显示关节曲线",
        options=list(JOINT_CN.keys()),
        default=["right_elbow", "right_shoulder", "right_knee"],
        format_func=lambda x: JOINT_CN.get(x, x),
    )

    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True,
                            disabled=(ref_file is None or user_file is None))

    if ref_file is None or user_file is None:
        st.info("请先上传两段视频")

    st.markdown("---")
    st.caption("v1.0 · MediaPipe + DTW")


# ── 主体 ──────────────────────────────────────────────────────────────────────
st.title("🏀 篮球投篮一致性分析器")
st.markdown("上传参考视频和你的投篮视频，获取关节角度对比和一致性评分。")

if not analyze_btn:
    # 空状态展示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\n上传参考动作视频（职业球员 / 标准动作）")
    with col2:
        st.info("**Step 2**\n\n上传你的投篮视频（侧面45°拍摄效果最佳）")
    with col3:
        st.info("**Step 3**\n\n点击「开始分析」获取得分和反馈")
    st.stop()


# ── 分析流程 ──────────────────────────────────────────────────────────────────
extractor = load_extractor()
scorer    = ConsistencyScorer(dtw_scale=dtw_scale)

with st.spinner("保存上传文件…"):
    ref_path  = save_upload(ref_file)
    user_path = save_upload(user_file)

col_ref, col_user = st.columns(2)
with col_ref:
    ref_seq  = process_video_with_progress(extractor, ref_path,  "参考视频")
with col_user:
    user_seq = process_video_with_progress(extractor, user_path, "你的视频")

if len(ref_seq.frames) < 5 or len(user_seq.frames) < 5:
    st.error("⚠️ 未能检测到足够的姿态帧，请确认视频中有完整的全身画面。")
    st.stop()

with st.spinner("计算一致性分数…"):
    report = scorer.compare(ref_seq, user_seq)


# ── 结果展示 ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.header("📊 分析结果")

# 第一行：总分 + 关键指标
col_score, col_metrics = st.columns([1, 2])

with col_score:
    fig_card = plot_score_card(report)
    st.image(fig_to_bytes(fig_card), use_container_width=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("总分",     f"{report.overall_score:.1f} / 100")
    m2.metric("等级",     report.grade)
    m3.metric("分析帧数", f"{len(user_seq.frames)} 帧")

    m4, m5 = st.columns(2)
    m4.metric("最不稳定关节", JOINT_CN.get(report.most_inconsistent_joint, ""))
    m5.metric("最差阶段",     report.most_inconsistent_phase)

    st.markdown("**📝 分析反馈**")
    for fb in report.feedback:
        st.markdown(f"- {fb}")

st.markdown("---")

# 第二行：雷达图 + 条形图
col_radar, col_bar = st.columns(2)
with col_radar:
    st.subheader("关节雷达图")
    fig_radar = plot_radar(report)
    st.image(fig_to_bytes(fig_radar), use_container_width=True)

with col_bar:
    st.subheader("各关节得分")
    fig_bar = plot_joint_bars(report)
    st.image(fig_to_bytes(fig_bar), use_container_width=True)

st.markdown("---")

# 第三行：角度曲线
st.subheader("📈 关节角度对比曲线")
if show_joints:
    fig_curves = plot_angle_curves(ref_seq, user_seq, joints=show_joints)
    st.image(fig_to_bytes(fig_curves), use_container_width=True)
else:
    st.info("请在左侧选择要显示的关节")

st.markdown("---")

# 第四行：骨骼视频（可选）
with st.expander("🎥 骨骼标注对比视频（可选，生成较慢）", expanded=False):
    if st.button("生成骨骼视频"):
        with st.spinner("生成中…"):
            out_path = tempfile.mktemp(suffix=".mp4")
            result   = export_skeleton_frames(ref_seq, user_seq, out_path, max_frames=60)
        if result:
            with open(result, "rb") as f:
                st.video(f.read())
        else:
            st.warning("骨骼帧不足，无法生成视频。")

st.markdown("---")

# 关节得分详情表
with st.expander("📋 详细关节得分表", expanded=False):
    import pandas as pd
    rows = []
    for js in report.joint_scores:
        if js.weight > 0:
            rows.append({
                "关节":    JOINT_CN.get(js.joint, js.joint),
                "得分":    f"{js.score:.1f}",
                "DTW距离": f"{js.dtw_distance:.1f}",
                "权重":    f"{js.weight:.0%}",
                "状态":    "⚠️ 需改善" if js.is_most_inconsistent else ("✅ 良好" if js.score >= 75 else "📌 关注"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# 清理临时文件
try:
    os.unlink(ref_path)
    os.unlink(user_path)
except Exception:
    pass
