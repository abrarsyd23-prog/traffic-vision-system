"""
helpers.py — Frame conversion, video I/O, and display utilities.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import SUPPORTED_IMAGE_TYPES, SUPPORTED_VIDEO_TYPES, MAX_DISPLAY_WIDTH

# Maximum resolution for inference — prevents bbox coordinate mismatch on large images
INFERENCE_MAX_WIDTH = 1280


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    bgr = pil_to_bgr(image)
    # Resize large images before inference to keep bbox coords consistent
    h, w = bgr.shape[:2]
    if w > INFERENCE_MAX_WIDTH:
        scale = INFERENCE_MAX_WIDTH / w
        bgr = cv2.resize(bgr, (INFERENCE_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def get_video_properties(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    props = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def frame_generator(video_path: str, skip_frames: int = 1):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip_frames == 0:
                yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def resize_for_display(frame: np.ndarray, max_width: int = MAX_DISPLAY_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lstrip(".").lower() in SUPPORTED_IMAGE_TYPES


def is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lstrip(".").lower() in SUPPORTED_VIDEO_TYPES
