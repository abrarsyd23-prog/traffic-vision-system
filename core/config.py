"""
config.py — Central configuration for the Traffic Vision System.
All thresholds, paths, and model settings live here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_DEFAULT_MODEL = "yolov8n.pt"
YOLO_FINETUNED_MODEL = os.path.join(MODELS_DIR, "yolov8n_traffic.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_IOU_THRESHOLD = 0.45
YOLO_INPUT_SIZE = 416

# Traffic-relevant COCO classes to highlight (others still shown but dimmed)
TRAFFIC_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign"}

# ── Edge Detection ────────────────────────────────────────────────────────────
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
CANNY_APERTURE_SIZE = 3
CANNY_L2_GRADIENT = True

# ── Face Detection ────────────────────────────────────────────────────────────
FACE_MIN_DETECTION_CONFIDENCE = 0.60
FACE_DETECTION_MAX_WIDTH = 960

# ── Object Tracking ───────────────────────────────────────────────────────────
TRACKER_TYPE = "CSRT"
TRACKER_LABEL = "Tracked Vehicle"

# ── Visualization ─────────────────────────────────────────────────────────────
BBOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
LABEL_PADDING = 4
EDGE_OVERLAY_ALPHA = 0.6

# Traffic class colors (BGR)
CLASS_COLORS = {
    "car":           (220, 80,  20),
    "truck":         (180, 40,  40),
    "bus":           (40,  120, 220),
    "motorcycle":    (40,  180, 80),
    "bicycle":       (60,  200, 120),
    "person":        (200, 80,  200),
    "traffic light": (20,  200, 200),
    "stop sign":     (20,  40,  220),
}
DEFAULT_COLOR = (100, 100, 100)

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE = "Traffic Vision System"
APP_ICON = "🚦"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
MAX_DISPLAY_WIDTH = 720
