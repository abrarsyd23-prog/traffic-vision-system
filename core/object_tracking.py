"""
object_tracking.py — CSRT vehicle tracker for following a selected vehicle across frames.
"""

import cv2
import numpy as np

from core.config import TRACKER_TYPE, TRACKER_LABEL, BBOX_THICKNESS, LABEL_FONT_SCALE, LABEL_THICKNESS, LABEL_PADDING

TRACKER_COLOR_BGR = (20, 160, 80)

_TRACKER_CONSTRUCTORS = {
    "CSRT": cv2.TrackerCSRT_create,
    "KCF":  cv2.TrackerKCF_create,
}


def create_tracker() -> cv2.Tracker:
    constructor = _TRACKER_CONSTRUCTORS.get(TRACKER_TYPE)
    if constructor is None:
        raise ValueError(f"Unsupported tracker: {TRACKER_TYPE}")
    return constructor()


def initialise_tracker(tracker: cv2.Tracker, frame: np.ndarray, bbox: tuple) -> None:
    tracker.init(frame, bbox)


def update_tracker(tracker: cv2.Tracker, frame: np.ndarray) -> tuple:
    success, box = tracker.update(frame)
    if success:
        return True, tuple(map(int, box))
    return False, None


def draw_tracking_result(
    frame: np.ndarray,
    bbox,
    success: bool,
    is_image_mode: bool = False,
) -> np.ndarray:
    output = frame.copy()
    if is_image_mode:
        return output
    if success and bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(output, (x, y), (x + w, y + h), TRACKER_COLOR_BGR, BBOX_THICKNESS)
        (text_w, text_h), _ = cv2.getTextSize(
            TRACKER_LABEL, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
        )
        bg_y1 = max(y - text_h - LABEL_PADDING * 2, 0)
        cv2.rectangle(output, (x, bg_y1), (x + text_w + LABEL_PADDING, y), TRACKER_COLOR_BGR, -1)
        cv2.putText(
            output, TRACKER_LABEL,
            (x + LABEL_PADDING // 2, y - LABEL_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
            (255, 255, 255), LABEL_THICKNESS,
        )
    else:
        cv2.putText(
            output, "Tracking lost",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2,
        )
    return output


def xyxy_to_xywh(bbox: tuple) -> tuple:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(bbox: tuple) -> tuple:
    x, y, w, h = bbox
    return x, y, x + w, y + h
