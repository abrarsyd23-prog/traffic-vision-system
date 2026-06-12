"""
object_detection.py — YOLOv8 object detection with traffic-class highlighting.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

from core.config import (
    YOLO_DEFAULT_MODEL,
    YOLO_FINETUNED_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_INPUT_SIZE,
    BBOX_THICKNESS,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    LABEL_PADDING,
    CLASS_COLORS,
    DEFAULT_COLOR,
)

_model_cache: dict = {}


def _get_model(use_finetuned: bool) -> YOLO:
    model_path = (
        YOLO_FINETUNED_MODEL
        if use_finetuned and os.path.exists(YOLO_FINETUNED_MODEL)
        else YOLO_DEFAULT_MODEL
    )
    if model_path not in _model_cache:
        _model_cache[model_path] = YOLO(model_path)
    return _model_cache[model_path]


def detect_objects(
    frame: np.ndarray,
    use_finetuned: bool = False,
    confidence: float = YOLO_CONFIDENCE_THRESHOLD,
) -> list:
    model = _get_model(use_finetuned)
    results = model.predict(
        source=frame,
        conf=confidence,
        iou=YOLO_IOU_THRESHOLD,
        imgsz=YOLO_INPUT_SIZE,
        verbose=False,
    )
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
            })
    return detections


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['confidence']:.2f}"
        color = CLASS_COLORS.get(det["label"], DEFAULT_COLOR)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )
        bg_y1 = max(y1 - text_h - LABEL_PADDING * 2, 0)
        cv2.rectangle(output, (x1, bg_y1), (x1 + text_w + LABEL_PADDING, y1), color, -1)
        cv2.putText(
            output, label,
            (x1 + LABEL_PADDING // 2, y1 - LABEL_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
            (255, 255, 255), LABEL_THICKNESS,
        )
    return output


def run_object_detection(
    frame: np.ndarray,
    use_finetuned: bool = False,
    confidence: float = YOLO_CONFIDENCE_THRESHOLD,
) -> tuple:
    detections = detect_objects(frame, use_finetuned, confidence)
    annotated = draw_detections(frame, detections)
    return detections, annotated
