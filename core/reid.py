"""
Módulo de Extração Biométrica e Atributos Visuais (Part-Based ReID) - Sentinel-Trace.
Projetado para câmaras de galpão logístico (baixa resolução, ângulo alto e EPIs).
Divide o corpo em Cabeça, Tronco e Pernas, extraindo cor (HSV), textura profunda e biotipo.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from config.settings import REID_IMG_SIZE


class FeatureExtractor:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Usamos as camadas intermediárias da ResNet18 (capturam melhor texturas de roupas, cabelo e acessórios)
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(REID_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_part_color_hist(self, part_bgr: np.ndarray) -> np.ndarray:
        """
        Extrai assinatura de cor invariante à iluminação (HSV) de uma parte específica do corpo.
        Essencial para diferenciar colete, boné, tom de pele e cor de roupa em câmaras distantes.
        """
        if part_bgr is None or part_bgr.size == 0:
            return np.zeros(48, dtype=np.float32)

        hsv = cv2.cvtColor(part_bgr, cv2.COLOR_BGR2HSV)
        # Histograma de Matiz (Cor pura), Saturação e Brilho
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Gera um vetor híbrido de 512 dimensões combinando:
        - Fatiamento Horizontal (Cabeça/Boné, Tronco/Colete, Pernas/Calçado)
        - Textura neural (ResNet)
        - Proporção física (Biotipo)
        """
        if crop is None or crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        h, w = crop.shape[:2]

        # 1. Fatiamento Corporal (Recorta o fundo das bordas laterais em 15% para focar só na pessoa)
        margin_x = int(w * 0.15)
        core_crop = crop[:, margin_x:max(margin_x + 1, w - margin_x)] if w > 20 else crop

        head_part = core_crop[0:max(1, int(h * 0.22)), :]              # Cabeça, cabelo, boné
        torso_part = core_crop[int(h * 0.22):max(2, int(h * 0.65)), :]  # Camisa, colete, braços/tatuagens
        legs_part = core_crop[int(h * 0.65):h, :]                       # Calça, bermuda, calçado

        hist_head = self._extract_part_color_hist(head_part)    # 48 dims
        hist_torso = self._extract_part_color_hist(torso_part)  # 48 dims
        hist_legs = self._extract_part_color_hist(legs_part)    # 48 dims

        # 2. Extração de textura profunda (padrões de roupa, cabelo longo/curto, silhueta)
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            deep_feat = self.model(tensor).view(1, -1).cpu().numpy()[0]  # 512 dims
            # Centraliza e normaliza a feature profunda para não esmagar as diferenças
            deep_feat = deep_feat - np.mean(deep_feat)
            d_norm = np.linalg.norm(deep_feat)
            if d_norm > 0:
                deep_feat /= d_norm

        # Reduzimos a parte profunda para 368 posições e somamos os 144 das 3 partes do corpo = 512 dims cravados
        deep_subset = deep_feat[:368] * 0.45  # Peso de 45% para textura geral
        color_parts = np.concatenate([
            hist_head * 0.25,   # Cabeça/Boné/Cabelo
            hist_torso * 0.55,  # Tronco/Colete tem o maior peso discriminativo
            hist_legs * 0.35    # Pernas/Calça
        ])

        combined = np.concatenate([color_parts, deep_subset]).astype(np.float32)

        # Normalização final L2
        final_norm = np.linalg.norm(combined)
        if final_norm > 0:
            combined /= final_norm

        return combined