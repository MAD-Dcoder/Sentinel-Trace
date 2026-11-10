"""
Módulo de Ingestão e Processamento de Streams de Vídeo - Sentinel-Trace.
Sincronizado com o relógio real para manter a velocidade original do vídeo.
"""

from pathlib import Path
import time
import cv2
from pipeline.tracker_pipeline import VideoTrackingPipeline
from core.tracker import GlobalTracker
from db.database import ForensicDatabase


class VideoStreamProcessor:
    def __init__(self, source, camera_id: str = "Cam_01", frame_skip: int = 2, tracker: GlobalTracker = None, db: ForensicDatabase = None):
        self.source = str(source)
        self.camera_id = camera_id
        self.frame_skip = max(1, frame_skip)
        self.pipeline = VideoTrackingPipeline(camera_id=self.camera_id, db=db, tracker=tracker)

    def run(self, display: bool = True):
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tempo ideal (em segundos) que cada frame processado deve durar na tela
        target_frame_duration = self.frame_skip / fps

        print(f"\n[VARREDURA LP] Iniciando câmara: {self.camera_id} ({Path(self.source).name})")
        print(f"[INFO] Resolução original: {width}x{height} | FPS nativo: {fps:.1f} | Frames: {total_frames}")

        frame_count = 0

        try:
            while cap.isOpened():
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                # Redimensiona para 720p se for maior, garantindo inferência rápida na CPU
                if width > 800:
                    scale = 800.0 / width
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

                video_time_sec = frame_count / fps

                annotated_frame, _ = self.pipeline.process_frame(frame=frame, video_time_sec=video_time_sec)

                if display:
                    cv2.imshow(f"Sentinel-Trace | {self.camera_id}", annotated_frame)

                    # Desconta o tempo que a IA gastou do tempo de espera do frame!
                    elapsed = time.perf_counter() - t_start
                    remaining_ms = int(max(1, (target_frame_duration - elapsed) * 1000))

                    if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
                        print("[INFO] Câmara pulada pelo investigador.")
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[OK] {self.camera_id} finalizada ({frame_count} frames lidos).")