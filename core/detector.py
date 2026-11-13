"""
Detecção (YOLOv8s / OpenVINO) + rastreio local (ByteTrack) — Sentinel-Trace.
"""

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

from config.settings import (
    DETECTOR_WEIGHTS, DETECTOR_FALLBACK, DETECT_IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD,
    TARGET_CLASSES, TRACKER_CONFIG_PATH, MIN_BOX_W, MIN_BOX_H,
)


class PersonDetector:
    def __init__(self):
        if DETECTOR_WEIGHTS.exists():
            self.model, self.backend = YOLO(str(DETECTOR_WEIGHTS), task="detect"), "OpenVINO"
        else:
            self.model, self.backend = YOLO(str(DETECTOR_FALLBACK)), "PyTorch"

    def detect(self, frame: np.ndarray):
        return self.model.predict(frame, classes=TARGET_CLASSES, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                                  imgsz=DETECT_IMGSZ, verbose=False)[0].boxes.cpu().numpy()


class CameraTracker:
    """ByteTrack independente por janela de câmera; buffer de oclusão em segundos."""

    def __init__(self, sample_fps: float):
        cfg = YAML.load(TRACKER_CONFIG_PATH)
        cfg["track_buffer"] = max(1, int(round(float(cfg.pop("track_buffer_sec", 8.0)) * sample_fps)))
        self.tracker = BYTETracker(IterableSimpleNamespace(**cfg))

    @property
    def has_active(self) -> bool:
        return bool(self.tracker.tracked_stracks or self.tracker.lost_stracks)

    def update(self, boxes, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        dets = []
        for x1, y1, x2, y2, tid, score, _cls, _idx in self.tracker.update(boxes, frame):
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            if (x2 - x1) >= MIN_BOX_W and (y2 - y1) >= MIN_BOX_H:
                dets.append({"local_id": int(tid), "bbox": [x1, y1, x2, y2], "confidence": float(score)})
        return dets
