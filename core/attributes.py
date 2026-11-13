"""
Atributos visuais simples (cor da camisa e da calça) — Sentinel-Trace.

Servem para o investigador DESCREVER o alvo ("camisa preta, calça jeans") e o
Sentinel ordenar primeiro quem bate com a descrição. Não decidem sozinhos:
iluminação e câmera mudam cores, então é uma pista, não uma prova.
"""

import cv2
import numpy as np

COLORS = ["preto", "branco", "cinza", "vermelho", "marrom", "laranja", "amarelo", "verde", "azul", "roxo/rosa"]


def _color_names(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # Limiares medidos nos vídeos do galpão: preto ~V32, marrom ~V55/S93/H8 (CCTV escurece tudo)
    out = np.full(h.shape, "cinza", dtype=object)
    out[(v < 40) | ((v < 60) & (s < 60))] = "preto"
    out[(s < 50) & (v >= 185)] = "branco"
    c = (s >= 60) & (v >= 40)
    warm = (h < 22) | (h >= 165)
    out[c & warm & (v < 110)] = "marrom"
    out[c & warm & (v >= 110) & ((h < 8) | (h >= 165))] = "vermelho"
    out[c & (h >= 8) & (h < 22) & (v >= 110)] = "laranja"
    out[c & (h >= 22) & (h < 35)] = "amarelo"
    out[c & (h >= 35) & (h < 85)] = "verde"
    out[c & (h >= 85) & (h < 130)] = "azul"
    out[c & (h >= 130) & (h < 165)] = "roxo/rosa"
    return out


def _dominant(region: np.ndarray) -> str | None:
    if region is None or region.size == 0 or min(region.shape[:2]) < 4:
        return None
    names = _color_names(cv2.cvtColor(region, cv2.COLOR_BGR2HSV)).ravel()
    vals, counts = np.unique(names, return_counts=True)
    return str(vals[np.argmax(counts)])


def crop_attributes(crop: np.ndarray) -> dict:
    h, w = crop.shape[:2]
    cx0, cx1 = int(w * 0.25), int(w * 0.75)            # faixa central: menos fundo/prateleira
    return {
        "camisa": _dominant(crop[int(h * 0.20):int(h * 0.50), cx0:cx1]),
        "calca": _dominant(crop[int(h * 0.60):int(h * 0.90), cx0:cx1]),
    }


def tracklet_attributes(crops: list) -> dict:
    """Votação entre os melhores crops do tracklet."""
    votes = {"camisa": {}, "calca": {}}
    for c in crops:
        for k, v in crop_attributes(c).items():
            if v:
                votes[k][v] = votes[k].get(v, 0) + 1
    return {k: (max(v, key=v.get) if v else None) for k, v in votes.items()}


def description_match(attrs: dict, wanted: dict) -> float:
    """1.0 = bate tudo que foi descrito, 0.0 = nada; campos não descritos são ignorados."""
    keys = [k for k in ("camisa", "calca") if wanted.get(k)]
    if not keys:
        return 0.5
    return sum(1.0 for k in keys if attrs.get(k) == wanted[k]) / len(keys)
