"""
Script utilitário para descarregar vídeo de teste de CCTV/pedestres diretamente para a pasta data/.
"""

import urllib.request
from config.settings import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)
dest_path = DATA_DIR / "sample_cctv.mp4"

# Amostra pública de teste de deteção de pessoas da Intel IoT Devkit
VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"

print(f"A descarregar vídeo de teste de vigilância...")
print(f"Destino: {dest_path}")

try:
    urllib.request.urlretrieve(VIDEO_URL, dest_path)
    print("\n✅ Vídeo descarregado com sucesso na pasta data/ (sample_cctv.mp4)!")
except Exception as e:
    print(f"\n❌ Erro ao descarregar: {e}")