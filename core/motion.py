"""
Filtro de movimento — Sentinel-Trace.

Comparar dois frames em miniatura (640 px, tons de cinza) custa ~2 ms; rodar a IA de
detecção custa ~270 ms. Em galpão, boa parte do tempo o corredor está vazio: nesses
frames a IA nem é chamada.

Calibrado nos vídeos reais: na câmera de teto 4K uma pessoa andando muda só ~0.03% da
imagem, e câmeras presas em estante tremem; por isso a comparação é anti-tremor e o limiar é uma contagem ABSOLUTA de pixels (≈ um passo de uma pessoa
pequena), e não uma porcentagem grande. Depois de qualquer movimento a IA continua
ligada por alguns segundos (histerese), para não perder quem desacelera.
"""

import cv2
import numpy as np

from config.settings import MOTION_WIDTH, MOTION_MIN_PIXELS, MOTION_HOLD_S, MOTION_IGNORE_BOTTOM

_K5 = np.ones((5, 5), np.uint8)
_K2 = np.ones((2, 2), np.uint8)


class MotionGate:
    def __init__(self, sample_fps: float):
        self.prev = None
        self.hold_frames = max(1, int(round(MOTION_HOLD_S * sample_fps)))
        self.since_motion = 10 ** 9

    def has_motion(self, frame: np.ndarray) -> bool:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (MOTION_WIDTH, int(h * MOTION_WIDTH / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        gray = gray[: int(gray.shape[0] * (1 - MOTION_IGNORE_BOTTOM))]   # ignora a faixa do relógio/OSD
        prev, self.prev = self.prev, gray
        if prev is None:
            moving = True
        else:
            # Anti-tremor: um pixel só "mudou" se não puder ser explicado por um deslocamento de
            # até 2 px da imagem anterior (câmera presa em estante vibra; prateleira tem muita borda).
            g = gray.astype(np.int16)
            changed = ((g - cv2.dilate(prev, _K5)) > 22) | ((cv2.erode(prev, _K5).astype(np.int16) - g) > 22)
            changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, _K2)
            moving = np.count_nonzero(changed) >= MOTION_MIN_PIXELS
        self.since_motion = 0 if moving else self.since_motion + 1
        return self.since_motion <= self.hold_frames
