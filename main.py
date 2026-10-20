"""
Ponto de entrada do Sentinel-Trace para testes operacionais.
"""

from pathlib import Path
from pipeline.video_stream import VideoStreamProcessor
from config.settings import DATA_DIR

def run_test():
    # Procura se existe algum vídeo de teste na pasta data/
    video_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.avi"))

    if video_files:
        test_video = video_files[0]
        print(f"Vídeo de teste detectado na pasta data: {test_video.name}")
        processor = VideoStreamProcessor(
            source=test_video,
            camera_id="Cam_01_Armazem",
            frame_skip=2
        )
    else:
        print("Nenhum ficheiro de vídeo encontrado em data/. Tentando abrir a webcam padrão (índice 0)...")
        processor = VideoStreamProcessor(
            source=0,
            camera_id="Cam_01_Webcam",
            frame_skip=2
        )

    # Executa o processamento com exibição visual na tela
    processor.run(display=True)

if __name__ == "__main__":
    run_test()