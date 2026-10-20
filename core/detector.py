"""
Módulo de Deteção e Rastreio Local - Sentinel-Trace.
Utiliza YOLOv8 com ByteTrack para travamento espacial contínuo intra-câmara.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import YOLO_MODEL_NAME, CONF_THRESHOLD, IOU_THRESHOLD, TARGET_CLASSES


class PersonDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME, conf_thresh: float = CONF_THRESHOLD):
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.target_classes = TARGET_CLASSES

    def track(self, frame: np.ndarray):
        """
        Executa deteção e rastreamento local (ByteTrack).
        Retorna bbox, id local consistente e recorte corporal (crop).
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=self.target_classes,
            conf=self.conf_thresh,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        frame_h, frame_w = frame.shape[:2]

        for box in boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            local_id = int(box.id[0].item())

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)

            if (x2 - x1) <= 10 or (y2 - y1) <= 10:
                continue

            crop = frame[y1:y2, x1:x2].copy()

            detections.append({
                "local_id": local_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop": crop
            })

        return detections