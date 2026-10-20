"""
Módulo de Extração de Embeddings (Person Re-Identification) - Sentinel-Trace.
Converte recortes corporais em vetores normalizados de 512 dimensões.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from config.settings import REID_IMG_SIZE


class FeatureExtractor:
    def __init__(self, device: str = None):
        """
        Inicializa o extrator de características corporais com backbone leve e pré-treinado.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Arquitetura baseada em ResNet18 modificada para extração direta de vetores (512-d)
        # Permite execução veloz tanto em CPU local quanto com aceleração por GPU
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove a última camada de classificação para reter o pool de características puro
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()

        # Normalização padrão para modelos treinados em ImageNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(REID_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Recebe um recorte de imagem BGR (OpenCV) e extrai o vetor normalizado L2.

        Retorna:
            embedding (np.ndarray): Vetor 1D de tamanho 512 com norma unitária.
        """
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        # Converte BGR para RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
            # Achata o tensor para vetor unidimensional de 512 posições
            features = features.view(features.size(0), -1).cpu().numpy()[0]

        # Normalização L2 para viabilizar busca rápida via afinidade por cosseno
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features