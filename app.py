"""
app.py — Traffic Vision System
Run with: streamlit run app.py
"""

import time
import cv2
import streamlit as st
from PIL import Image

from core.config import (
    APP_TITLE, APP_ICON,
    SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES,
    YOLO_CONFIDENCE_THRESHOLD,
)
from core.pipeline import TrafficVisionPipeline
from utils.helpers import (
    uploaded_file_to_bgr,
    save_uploaded_video,
    bgr_to_rgb,
    resize_for_display,
    frame_generator,
    get_video_properties,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #1a1a2e;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    .app-header p {
        margin: 0.25rem 0 0 0;
        opacity: 0.7;
        font-size: 0.85rem;
    }

    /* Stat cards */
    .stat-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .stat-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stat-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1;
    }
    .stat-card .label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }
    .stat-card.highlight .value { color: #e84545; }

    /* Task badges */
    .badge {
        display: inline-block;
        background: #f0f4ff;
        border: 1px solid #c7d7ff;
        color: #3355cc;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 2px;
    }

    /* Section label */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
        margin: 1rem 0 0.5rem 0;
    }

    /* Image panels */
    .panel-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #444;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Button */
    .stButton > button {
        background: #1a1a2e;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-size: 0.9rem;
    }
    .stButton > button:hover {
        background: #2d2d4e;
        color: white;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline_key" not in st.session_state:
    st.session_state.pipeline_key = None
    st.session_state.pipeline = None


def _get_pipeline(use_finetuned: bool, confidence: float) -> TrafficVisionPipeline:
    key = f"{use_finetuned}_{confidence:.2f}"
    if st.session_state.pipeline_key != key:
        st.session_state.pipeline = TrafficVisionPipeline(
            use_finetuned=use_finetuned,
            yolo_confidence=confidence,
        )
        st.session_state.pipeline_key = key
    return st.session_state.pipeline


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {APP_ICON} {APP_TITLE}")
    st.markdown("<small>CSCI435 · UOWD · Spring 2026</small>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**Model Settings**")
    use_finetuned = st.toggle("Use fine-tuned model", value=False)
    confidence = st.slider("Detection confidence", 0.10, 0.95, YOLO_CONFIDENCE_THRESHOLD, 0.05)

    st.divider()
    st.markdown("**Active CV Tasks**")
    for task in ["Edge Detection", "Object Detection", "Face Detection", "Object Tracking"]:
        st.markdown(f'<span class="badge">✓ {task}</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**Input Mode**")
    input_mode = st.radio("", ["Upload Image", "Upload Video", "Webcam"], label_visibility="collapsed")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🚦 Traffic Vision System</h1>
    <p>Real-time traffic scene analysis · Vehicle detection · Lane edge mapping · Pedestrian tracking</p>
</div>
""", unsafe_allow_html=True)

pipeline = _get_pipeline(use_finetuned, confidence)


# ── Image mode ────────────────────────────────────────────────────────────────
if input_mode == "Upload Image":
    uploaded = st.file_uploader(
        "Upload a traffic image", type=SUPPORTED_IMAGE_TYPES, label_visibility="collapsed"
    )

    if uploaded:
        frame_bgr = uploaded_file_to_bgr(uploaded)
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="panel-label">Original</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(resize_for_display(frame_bgr)), width=480)

        with st.spinner("Analysing scene…"):
            t0 = time.perf_counter()
            result = pipeline.process_image(frame_bgr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        with col2:
            st.markdown('<div class="panel-label">Annotated Output</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(resize_for_display(result.annotated_frame)), width=480)

        with st.expander("🛣️ Edge Detection Mask — Lane & Road Boundaries"):
            st.image(resize_for_display(result.edge_mask), width=480, clamp=True)

        st.divider()
        st.markdown('<div class="section-label">Scene Analysis Results</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card highlight">
                <div class="value">{result.vehicle_count}</div>
                <div class="label">Vehicles</div>
            </div>
            <div class="stat-card">
                <div class="value">{result.pedestrian_count}</div>
                <div class="label">Pedestrians</div>
            </div>
            <div class="stat-card">
                <div class="value">{result.face_count}</div>
                <div class="label">Faces</div>
            </div>
            <div class="stat-card">
                <div class="value">{elapsed_ms:.0f}ms</div>
                <div class="label">Inference Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if result.detections:
            st.markdown('<div class="section-label">Detected Objects</div>', unsafe_allow_html=True)
            rows = [
                {"Label": d["label"], "Confidence": f"{d['confidence']:.1%}", "BBox": str(d["bbox"])}
                for d in result.detections
            ]
            st.dataframe(rows, use_container_width=False)


# ── Video mode ────────────────────────────────────────────────────────────────
elif input_mode == "Upload Video":
    uploaded = st.file_uploader(
        "Upload a traffic video", type=SUPPORTED_VIDEO_TYPES, label_visibility="collapsed"
    )

    if uploaded:
        video_path = save_uploaded_video(uploaded)
        props = get_video_properties(video_path)

        st.info(f"📹 {props['width']}×{props['height']} · {props['fps']:.1f} FPS · {props['frame_count']} frames")

        skip = st.slider("Process every N frames", min_value=1, max_value=10, value=3)

        if st.button("▶️ Analyse Video"):
            pipeline.reset_tracker()
            frame_ph = st.empty()
            summary_ph = st.empty()
            progress = st.progress(0)
            total = max(props["frame_count"] // skip, 1)

            for i, (_, frame) in enumerate(frame_generator(video_path, skip_frames=skip)):
                result = pipeline.process_frame(frame)
                frame_ph.image(bgr_to_rgb(resize_for_display(result.annotated_frame)), width=480)
                summary_ph.markdown(f"```\n{result.summary()}\n```")
                progress.progress(min((i + 1) / total, 1.0))

            st.success("✅ Analysis complete.")


# ── Webcam mode ───────────────────────────────────────────────────────────────
elif input_mode == "Webcam":
    st.info("📷 Click **Take Photo** to capture and analyse a frame from your camera.")
    img_file = st.camera_input("Capture Frame")

    if img_file:
        frame_bgr = uploaded_file_to_bgr(img_file)
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="panel-label">Captured Frame</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(resize_for_display(frame_bgr)), width=480)

        with st.spinner("Analysing…"):
            t0 = time.perf_counter()
            result = pipeline.process_image(frame_bgr)
            elapsed_ms = (time.perf_counter() - t0) * 1000

        with col2:
            st.markdown('<div class="panel-label">Annotated Output</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(resize_for_display(result.annotated_frame)), width=480)

        st.markdown(f"```\n{result.summary()}\nInference: {elapsed_ms:.0f}ms\n```")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<small style='color:#aaa'>CSCI435 · Computer Vision Algorithms and Systems · University of Wollongong in Dubai · Spring 2026</small>",
    unsafe_allow_html=True,
)
