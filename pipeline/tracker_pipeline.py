"""
Pipeline de Processamento de Vídeo Integrado - Sentinel-Trace.
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

    def process_frame(self, frame):
        detections = self.detector.track(frame)
        annotated_frame = frame.copy()
        timestamp = datetime.now(timezone.utc).isoformat()

        for det in detections:
            local_id = det["local_id"]
            bbox = det["bbox"]
            crop = det["crop"]

            embedding = self.extractor.extract(crop)
            global_id = self.tracker.resolve_global_identity(
                camera_id=self.camera_id,
                local_id=local_id,
                embedding=embedding,
                bbox=bbox
            )

            self.db.save_event(
                global_id=global_id,
                camera_id=self.camera_id,
                local_track_id=local_id,
                timestamp=timestamp,
                bbox=bbox,
                embedding=embedding
            )

            x1, y1, x2, y2 = bbox
            label = f"ID: {global_id} (Local #{local_id})"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return annotated_frame, len(detections)