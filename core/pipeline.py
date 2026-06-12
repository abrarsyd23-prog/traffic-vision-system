"""
pipeline.py — Unified 4-task traffic vision pipeline.
All CV tasks run on the same frame and results are merged into one annotated output.

Video mode auto-tracking: when processing video frames, the tracker automatically
locks onto the highest-confidence vehicle detection. If tracking is lost, it
re-acquires a target on the next frame.
"""

import numpy as np

from core.edge_detection import run_edge_detection
from core.object_detection import run_object_detection
from core.face_detection import run_face_detection
from core.object_tracking import (
    create_tracker,
    initialise_tracker,
    update_tracker,
    draw_tracking_result,
    xyxy_to_xywh,
)

# Case-insensitive vehicle and pedestrian class sets
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "motorbike", "van", "tram"}
PEDESTRIAN_CLASSES = {"person", "pedestrian"}


class PipelineResult:
    def __init__(self, annotated_frame, edge_mask, detections, faces, tracking_bbox, tracking_success):
        self.annotated_frame = annotated_frame
        self.edge_mask = edge_mask
        self.detections = detections
        self.faces = faces
        self.tracking_bbox = tracking_bbox
        self.tracking_success = tracking_success

    @property
    def vehicle_count(self):
        return sum(1 for d in self.detections if d["label"].lower() in VEHICLE_CLASSES)

    @property
    def pedestrian_count(self):
        return sum(1 for d in self.detections if d["label"].lower() in PEDESTRIAN_CLASSES)

    @property
    def face_count(self):
        return len(self.faces)

    def summary(self) -> str:
        lines = [
            f"🚗 Vehicles detected: {self.vehicle_count}",
            f"🚶 Pedestrians detected: {self.pedestrian_count}",
            f"👤 Faces detected: {self.face_count}",
            f"🎯 Tracking: {'Active — following vehicle' if self.tracking_success else 'Acquiring target…'}",
        ]
        return "\n".join(lines)


class TrafficVisionPipeline:
    def __init__(self, use_finetuned: bool = False, yolo_confidence: float = 0.40):
        self.use_finetuned = use_finetuned
        self.yolo_confidence = yolo_confidence
        self._tracker = None
        self._tracking_active = False

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image(self, frame: np.ndarray) -> PipelineResult:
        """Run all CV tasks on a single image (no tracking)."""
        return self._run(frame, use_tracker=False)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run all CV tasks on a video frame (auto-tracking enabled)."""
        return self._run(frame, use_tracker=True)

    def initialise_tracking(self, frame: np.ndarray, bbox_xyxy: tuple) -> None:
        """Start tracking the object inside the given (x1, y1, x2, y2) box."""
        self._tracker = create_tracker()
        initialise_tracker(self._tracker, frame, xyxy_to_xywh(bbox_xyxy))
        self._tracking_active = True

    def reset_tracker(self) -> None:
        """Stop tracking and clear tracker state."""
        self._tracker = None
        self._tracking_active = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select_tracking_target(self, detections: list) -> dict | None:
        """Pick the highest-confidence vehicle detection as the tracking target."""
        vehicles = [d for d in detections if d["label"].lower() in VEHICLE_CLASSES]
        if not vehicles:
            return None
        return max(vehicles, key=lambda d: d["confidence"])

    def _run(self, frame: np.ndarray, use_tracker: bool) -> PipelineResult:
        edge_mask, _ = run_edge_detection(frame)
        detections, obj_frame = run_object_detection(frame, self.use_finetuned, self.yolo_confidence)
        faces, face_frame = run_face_detection(obj_frame)

        tracking_success = False
        tracking_bbox = None

        if use_tracker:
            if self._tracking_active and self._tracker is not None:
                # Continue following the current target
                tracking_success, tracking_bbox = update_tracker(self._tracker, frame)
                if not tracking_success:
                    # Lost the target — re-acquire on the next frame
                    self.reset_tracker()
            else:
                # Auto-acquire: lock onto the best vehicle detection
                target = self._select_tracking_target(detections)
                if target is not None:
                    self.initialise_tracking(frame, target["bbox"])
                    tracking_success = True
                    tracking_bbox = xyxy_to_xywh(target["bbox"])

        final_frame = draw_tracking_result(
            face_frame, tracking_bbox, tracking_success,
            is_image_mode=not use_tracker,
        )

        return PipelineResult(final_frame, edge_mask, detections, faces, tracking_bbox, tracking_success)
