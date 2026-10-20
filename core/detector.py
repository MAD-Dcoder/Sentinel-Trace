"""
Módulo de Deteção de Pessoas - Sentinel-Trace.
Encapsula o YOLOv8 com foco em pessoas e extração de crops para ReID.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from config.settings import YOLO_MODEL_NAME, CONF_THRESHOLD, IOU_THRESHOLD, TARGET_CLASSES


class PersonDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME, conf_thresh: float = CONF_THRESHOLD):
        """
        Inicializa o modelo YOLOv8 para deteção de pessoas.
        """
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.target_classes = TARGET_CLASSES

    def detect(self, frame: np.ndarray):
        """
        Processa um frame de vídeo e extrai bounding boxes e recortes corporais.

        Retorna:
            detections (list): Lista de dicionários com bbox [x1, y1, x2, y2], confiança e recorte corporal (crop).
        """
        results = self.model.predict(
            source=frame,
            classes=self.target_classes,
            conf=self.conf_thresh,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        detections = []
        if not results:
            return detections

        boxes = results[0].boxes
        frame_h, frame_w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())

            # Garantir que as coordenadas fiquem dentro dos limites do frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)

            # Validar se o crop possui dimensões válidas
            if (x2 - x1) <= 5 or (y2 - y1) <= 5:
                continue

            crop = frame[y1:y2, x1:x2].copy()

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop": crop
            })

        return detections