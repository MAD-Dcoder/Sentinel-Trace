"""
Módulo de Ingestão e Processamento de Streams de Vídeo - Sentinel-Trace.
Executa o pipeline em ficheiros de vídeo locais (data/) ou feeds RTSP contínuos.
"""

from pathlib import Path
import cv2
from pipeline.tracker_pipeline import VideoTrackingPipeline


class VideoStreamProcessor:
    def __init__(self, source, camera_id: str = "Cam_01", frame_skip: int = 2):
        """
        Inicializa o processador de vídeo.

        Args:
            source (str | int | Path): Caminho para ficheiro de vídeo, URL RTSP ou índice da webcam (0).
            camera_id (str): Identificador da câmara para auditoria forense.
            frame_skip (int): Processa 1 quadro a cada 'frame_skip' para optimizar a taxa de FPS.
        """
        self.source = str(source)
        self.camera_id = camera_id
        self.frame_skip = max(1, frame_skip)
        self.pipeline = VideoTrackingPipeline(camera_id=self.camera_id)

    def run(self, display: bool = True, output_path: str = None):
        """
        Inicia a leitura e processamento contínuo do vídeo.
        Pressione a tecla 'q' na janela do OpenCV para encerrar.
        """
        cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[INFO] Conectado à fonte: {self.source}")
        print(f"[INFO] Resolução: {width}x{height} | FPS nativo: {fps:.1f} | Total de frames: {total_frames}")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps / self.frame_skip, (width, height))

        frame_count = 0
        local_track_counter = 1

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] Fim da transmissão ou ficheiro concluído.")
                    break

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                # Processa detecção, extração corporal e persistência forense
                annotated_frame, detections_count = self.pipeline.process_frame(frame=frame)

                # Gravação em ficheiro de auditoria
                if writer:
                    writer.write(annotated_frame)

                # Exibição em janela OpenCV
                if display:
                    cv2.imshow(f"Sentinel-Trace | {self.camera_id}", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] Processamento interrompido pelo utilizador.")
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            print(f"[INFO] Concluído. Total de frames lidos: {frame_count}")