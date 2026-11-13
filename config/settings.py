"""
Configurações globais do Sentinel-Trace (busca forense sob demanda, alvo primeiro).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
# SENTINEL_CASOS / SENTINEL_CONHECIMENTO permitem apontar para outro lugar (ex.: testes, outro disco)
CASES_DIR = Path(os.environ.get("SENTINEL_CASOS", BASE_DIR / "casos"))   # Uma pasta por investigação (banco + crops = dado biométrico)
KNOWLEDGE_DB = Path(os.environ.get("SENTINEL_CONHECIMENTO", BASE_DIR / "db" / "conhecimento.db"))   # Aprendizado entre casos (sem imagens)
CAMERAS_CONFIG_PATH = BASE_DIR / "config" / "cameras.yaml"
TRACKER_CONFIG_PATH = BASE_DIR / "config" / "warehouse_tracker.yaml"
OSD_FONT_PATH = BASE_DIR / "config" / "osd_font.npz"   # "Fonte" aprendida do relógio gravado na imagem
ARCHIVE_INDEX = DATA_DIR / ".sentinel_acervo.json"     # Cache do que já foi lido de cada vídeo

# --- Leitura de vídeo ------------------------------------------------------
SAMPLE_FPS = 4.0               # Frames analisados por segundo de vídeo (pessoa andando ~1.4 m/s => ~35 cm entre amostras)
MOTION_WIDTH = 640             # Resolução da miniatura usada para checar movimento
MOTION_MIN_PIXELS = 20         # Pixels mudando (na miniatura, já sem tremor) para considerar movimento
MOTION_IGNORE_BOTTOM = 0.14    # Faixa inferior ignorada (relógio/marca d'água do VMS)
MOTION_HOLD_S = 2.0            # Mantém a IA ligada por N s depois do último movimento
IDLE_DETECT_EVERY = 3          # Sem movimento mas com pessoa rastreada (parada no bin): detecta 1 a cada N amostras

# --- Detecção --------------------------------------------------------------
# OpenVINO (otimizado para CPU Intel): ~2.2x mais rápido que PyTorch e com recall igual ou maior nos testes.
DETECTOR_WEIGHTS = MODELS_DIR / "yolov8s_1280_openvino_model"
DETECTOR_FALLBACK = BASE_DIR / "yolov8s.pt"   # usado se o modelo OpenVINO não existir
DETECT_IMGSZ = (736, 1280)
CONF_THRESHOLD = 0.20
IOU_THRESHOLD = 0.45
MIN_BOX_W, MIN_BOX_H = 8, 14
TARGET_CLASSES = [0]

# --- ReID ------------------------------------------------------------------
REID_WEIGHTS = MODELS_DIR / "osnet_ain_x1_0_msmt17.pth"
REID_WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal&export=download&confirm=t"
REID_IMG_SIZE = (256, 128)
REID_BATCH = 32

# --- Tracklets -------------------------------------------------------------
TRACKLET_MIN_DETECTIONS = 3    # Descarta rastros com menos de ~0.75 s de evidência
TRACKLET_TOPK = 10             # ReID roda SÓ nestes K melhores crops de cada pessoa
TRACKLET_TOPK_MIN_GAP_S = 0.4  # Espaçamento mínimo entre crops escolhidos (diversidade de pose)

# --- Caçada ----------------------------------------------------------------
ANCHOR_WINDOW_S = 20           # Janela padrão (±s) ao redor do horário informado na câmera de início
HUNT_BACK_S = 10               # Cada rodada olha um pouco antes do último avistamento...
HUNT_FORWARD_S = 90            # ...e até N segundos depois
HUNT_MAX_CAMERAS = 20          # Teto de câmeras por rodada (controle de custo)
SEARCH_ACCEPT_PROB = 0.80      # Modo automático: aceita candidato com prob >= este valor
SEARCH_MIN_PROB = 0.15         # Baixo de propósito: entre câmeras o match certo costuma ficar em 40-55%
SEARCH_TOP_N = 6
QUERY_MAX_EMBEDDINGS = 40
