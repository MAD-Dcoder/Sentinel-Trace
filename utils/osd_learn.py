"""
Aprende a "fonte" do relógio OSD a partir de UM vídeo cujo horário de início se conhece.

    python -m utils.osd_learn data/cam_02.mp4 "2026-11-11 07:53:10.75"

Rotula cada célula de dígito com o valor verdadeiro em frames longe das viradas de
segundo (meio do segundo), acumula as máscaras e grava a média por dígito em
config/osd_font.npz. Precisa de >= 10 s de vídeo para ver todos os dígitos 0-9 nas unidades.
"""

import sys
from datetime import datetime, timedelta

import cv2
import numpy as np

from config.settings import OSD_FONT_PATH
from core.osd_clock import cell_masks, SHIFT, CELL_W


def learn(video: str, start: datetime, seconds: float = 14.0):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    acc = {d: [] for d in range(10)}
    for i in range(int(seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        true = start + timedelta(seconds=i / fps)
        frac = true.microsecond / 1e6
        if not 0.3 <= frac <= 0.7:           # perto da virada: rótulo incerto, ignora
            continue
        label = true.strftime("%d%m%Y%H%M%S")
        for cell, ch in zip(cell_masks(frame), label):
            acc[int(ch)].append(cell[:, SHIFT:SHIFT + CELL_W])
    cap.release()
    missing = [d for d, v in acc.items() if not v]
    if missing:
        sys.exit(f"Faltaram exemplos dos dígitos {missing}; use um trecho mais longo.")
    font = {f"d{d}": (np.mean(v, axis=0) > 0.5).astype(np.float32) for d, v in acc.items()}
    np.savez_compressed(OSD_FONT_PATH, **font)
    print(f"Fonte do relógio salva em {OSD_FONT_PATH} | exemplos por dígito: "
          + ", ".join(f"{d}:{len(v)}" for d, v in acc.items()))


if __name__ == "__main__":
    learn(sys.argv[1], datetime.fromisoformat(sys.argv[2].replace(" ", "T")))
