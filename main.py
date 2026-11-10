"""
Motor de Varredura Multi-Câmara Sob Demanda - Sentinel-Trace.
Analisa todos os vídeos na pasta data/ mantendo a identidade global do suspeito entre câmaras.
"""

from pipeline.video_stream import VideoStreamProcessor
from core.tracker import GlobalTracker
from db.database import ForensicDatabase
from config.settings import DATA_DIR, DB_DIR
from utils.inspector import export_investigation_summary


def run_multi_camera_investigation():
    extensions = ("*.mp4", "*.MOV", "*.mov", "*.avi")
    video_files = []
    for ext in extensions:
        video_files.extend(DATA_DIR.glob(ext))

    # Filtra para pegar apenas os clipes das câmaras (ignora raw_1, raw_2 e sample_cctv)
    ignore_prefixes = ("raw_", "sample_")
    video_files = [f for f in set(video_files) if not f.name.lower().startswith(ignore_prefixes)]
    video_files = sorted(video_files, key=lambda x: x.name)

    if not video_files:
        print("Nenhum vídeo encontrado na pasta data/!")
        return

    # Limpa o banco da execução anterior para não misturar horários e gerar Dwell Time falso
    db_file = DB_DIR / "sentinel.db"
    if db_file.exists():
        db_file.unlink()
        print("🧹 Banco de dados anterior limpo para iniciar nova sessão de investigação.")

    print(f"🔍 Iniciando Varredura Sentinel-Trace em {len(video_files)} câmara(s)...")

    shared_tracker = GlobalTracker(similarity_threshold=0.72)
    shared_db = ForensicDatabase()

    for idx, video_path in enumerate(video_files, start=1):
        # Se o arquivo já começa com 'cam_', usa o próprio nome; senão, adiciona o prefixo
        stem = video_path.stem
        cam_name = stem if stem.lower().startswith("cam_") else f"Cam_0{idx}_{stem}"

        processor = VideoStreamProcessor(
            source=video_path,
            camera_id=cam_name,
            frame_skip=2,
            tracker=shared_tracker,
            db=shared_db
        )
        processor.run(display=True)

    print("\n✅ Varredura Multi-Câmara concluída! Gerando Raio-X Forense...")
    export_investigation_summary()


if __name__ == "__main__":
    run_multi_camera_investigation()