"""
Pipeline de Processamento Integrado - Sentinel-Trace.
Calcula Dwell Time (Tempo de Permanência) e dispara alertas visuais para Loss Prevention.
"""

from datetime import datetime, timezone
import time
import cv2
from core.detector import PersonDetector
from core.reid import FeatureExtractor
from core.tracker import GlobalTracker
from db.database import ForensicDatabase


class VideoTrackingPipeline:
    def __init__(self, camera_id: str, db: ForensicDatabase = None, tracker: GlobalTracker = None, dwell_alert_seconds: float = 8.0):
        self.camera_id = camera_id
        self.detector = PersonDetector()
        self.extractor = FeatureExtractor()
        self.tracker = tracker if tracker else GlobalTracker()
        self.db = db if db else ForensicDatabase()
        self.dwell_alert_seconds = dwell_alert_seconds

        # Regista o momento em que cada ID Global entrou nesta câmara: {global_id: timestamp_entrada}
        self.first_seen_in_cam = {}

    def process_frame(self, frame, video_time_sec: float = None):
        raw_detections = self.detector.track(frame)
        annotated_frame = frame.copy()

        for det in raw_detections:
            det["embedding"] = self.extractor.extract(det["crop"])

        resolved = self.tracker.update_frame(self.camera_id, raw_detections)
        timestamp = datetime.now(timezone.utc).isoformat()
        current_clock = video_time_sec if video_time_sec is not None else time.time()

        for det in resolved:
            gid = det["global_id"]
            local_id = det["local_id"]
            bbox = det["bbox"]

            # Cálculo do Dwell Time (Tempo de Permanência na câmara)
            if gid not in self.first_seen_in_cam:
                self.first_seen_in_cam[gid] = current_clock
            dwell_time = max(0.0, current_clock - self.first_seen_in_cam[gid])

            self.db.save_event(
                global_id=gid,
                camera_id=self.camera_id,
                local_track_id=local_id,
                timestamp=timestamp,
                bbox=bbox,
                embedding=det["embedding"]
            )

            x1, y1, x2, y2 = bbox
            is_suspicious = dwell_time >= self.dwell_alert_seconds

            # Muda de Verde (Normal) para Vermelho (Alerta de Permanência LP)
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            text_color = (255, 255, 255) if is_suspicious else (0, 0, 0)
            status_tag = " [ALERTA LP]" if is_suspicious else ""
            label = f"Alvo #{gid} | {dwell_time:.1f}s{status_tag}"

            # Desenha a caixa e o cabeçalho limpo (sem #local)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            box_width = max(170, len(label) * 9)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 26)), (x1 + box_width, y1), color, -1)
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                text_color,
                2
            )

        return annotated_frame, len(resolved)