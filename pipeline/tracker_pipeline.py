"""
Módulo do Pipeline de Processamento de Vídeo - Sentinel-Trace.
Integra Deteção, Rastreio Global e Persistência Forense para streams e ficheiros de CCTV.
"""

from datetime import datetime, timezone
import cv2
from core.detector import PersonDetector
from core.reid import FeatureExtractor
from core.tracker import GlobalTracker
from db.database import ForensicDatabase


class VideoTrackingPipeline:
    def __init__(self, camera_id: str, db: ForensicDatabase = None, tracker: GlobalTracker = None):
        self.camera_id = camera_id
        self.detector = PersonDetector()
        self.extractor = FeatureExtractor()
        self.tracker = tracker if tracker else GlobalTracker()
        self.db = db if db else ForensicDatabase()

    def process_frame(self, frame, local_track_counter: int = 1):
        """
        Processa um único frame de vídeo:
        1. Deteta indivíduos.
        2. Extrai embeddings corporais.
        3. Associa a uma identidade global.
        4. Regista na base de dados forense.
        5. Desenha anotações visuais no frame.
        """
        detections = self.detector.detect(frame)
        annotated_frame = frame.copy()

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            crop = det["crop"]
            embedding = self.extractor.extract(crop)
            local_id = local_track_counter + idx

            global_id = self.tracker.update_track(
                camera_id=self.camera_id,
                local_track_id=local_id,
                embedding=embedding,
                bbox=bbox,
                crop=crop
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            self.db.save_event(
                global_id=global_id,
                camera_id=self.camera_id,
                local_track_id=local_id,
                timestamp=timestamp,
                bbox=bbox,
                embedding=embedding
            )

            # Desenha bounding box e legenda
            x1, y1, x2, y2 = bbox
            label = f"ID Global: {global_id} | {det['confidence']:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return annotated_frame, len(detections)