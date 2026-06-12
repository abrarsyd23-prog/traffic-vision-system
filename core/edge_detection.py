"""
edge_detection.py — Canny edge detection for lane and road boundary mapping.
"""

import cv2
import numpy as np

from core.config import (
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    CANNY_APERTURE_SIZE,
    CANNY_L2_GRADIENT,
    EDGE_OVERLAY_ALPHA,
)


def detect_edges(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)
    edges = cv2.Canny(
        blurred,
        threshold1=CANNY_THRESHOLD_LOW,
        threshold2=CANNY_THRESHOLD_HIGH,
        apertureSize=CANNY_APERTURE_SIZE,
        L2gradient=CANNY_L2_GRADIENT,
    )
    return edges


def overlay_edges(frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_colored[:, :, 0] = 0
    edge_colored[:, :, 2] = 0
    return cv2.addWeighted(frame, 1.0, edge_colored, EDGE_OVERLAY_ALPHA, gamma=0)


def run_edge_detection(frame: np.ndarray) -> tuple:
    edges = detect_edges(frame)
    overlay = overlay_edges(frame, edges)
    return edges, overlay
