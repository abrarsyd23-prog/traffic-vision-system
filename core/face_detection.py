"""
face_detection.py — MediaPipe face detection for pedestrian and driver identification.
Detector is instantiated once at module level for performance.
"""

import cv2
import numpy as np
import mediapipe as mp

from core.config import (
    FACE_MIN_DETECTION_CONFIDENCE,
    FACE_DETECTION_MAX_WIDTH,
    BBOX_THICKNESS,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LABEL_PADDING,
)

_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(
    min_detection_confidence=FACE_MIN_DETECTION_CONFIDENCE,
    model_selection=1,
)

FACE_COLOR_BGR = (30, 130, 220)


def detect_faces(frame: np.ndarray) -> list:
    orig_h, orig_w = frame.shape[:2]

    if orig_w > FACE_DETECTION_MAX_WIDTH:
        proc_w = FACE_DETECTION_MAX_WIDTH
        proc_h = int(orig_h * (FACE_DETECTION_MAX_WIDTH / orig_w))
        proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        proc_frame = frame

    rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
    results = _detector.process(rgb)

    faces = []
    if not results.detections:
        return faces

    for detection in results.detections:
        b = detection.location_data.relative_bounding_box
        x1 = max(int(b.xmin * orig_w), 0)
        y1 = max(int(b.ymin * orig_h), 0)
        x2 = min(int((b.xmin + b.width) * orig_w), orig_w)
        y2 = min(int((b.ymin + b.height) * orig_h), orig_h)
        faces.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(detection.score[0]),
        })
    return faces


def draw_faces(frame: np.ndarray, faces: list) -> np.ndarray:
    output = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        label = f"Face {face['confidence']:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), FACE_COLOR_BGR, BBOX_THICKNESS)
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )
        bg_y1 = max(y1 - text_h - LABEL_PADDING * 2, 0)
        cv2.rectangle(
            output, (x1, bg_y1), (x1 + text_w + LABEL_PADDING, y1),
            FACE_COLOR_BGR, -1,
        )
        cv2.putText(
            output, label,
            (x1 + LABEL_PADDING // 2, y1 - LABEL_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
            (255, 255, 255), LABEL_THICKNESS,
        )
    return output


def run_face_detection(frame: np.ndarray) -> tuple:
    faces = detect_faces(frame)
    annotated = draw_faces(frame, faces)
    return faces, annotated
