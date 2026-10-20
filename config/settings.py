"""
Configurações globais do Sentinel-Trace.
Define limiares de confiança, pesos dos modelos e parâmetros de extração.
"""

from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

# Pesos e modelos
YOLO_MODEL_NAME = "yolov8n.pt"  # Modelo nano para maior agilidade e baixo consumo
REID_MODEL_NAME = "osnet_x0_25"  # Arquitetura OSNet leve e eficaz

# Limiares de Deteção e Tracking
CONF_THRESHOLD = 0.50  # Confiança mínima do YOLOv8
IOU_THRESHOLD = 0.45   # Limiar de IoU para NMS
REID_SIMILARITY_THRESHOLD = 0.65  # Limiar de afinidade por cosseno entre câmaras

# Dimensões de entrada para extração ReID (Altura x Largura padrão OSNet)
REID_IMG_SIZE = (256, 128)

# Classes COCO (0 representa 'person')
TARGET_CLASSES = [0]