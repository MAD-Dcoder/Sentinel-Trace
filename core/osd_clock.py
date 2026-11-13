"""
Leitor do relógio gravado na imagem (OSD) — Sentinel-Trace.

Exportações do Smart Client trazem data/hora desenhadas no vídeo ("11/11/2026 07:53:11").
O Sentinel lê esse relógio sozinho para descobrir o horário REAL do frame 0 de cada
exportação — sem o investigador digitar nada em config/cameras.yaml.

Como funciona (sem OCR externo, sem internet):
  - o relógio fica sempre nas mesmas posições da imagem; cada dígito ocupa uma "célula";
  - cada célula é binarizada (texto branco) e comparada com a "fonte" aprendida
    (config/osd_font.npz: um molde por dígito 0-9);
  - o resultado só é aceito se formar data/hora válida e todas as células baterem bem.

Precisão do horário de início: o Sentinel procura o frame exato em que o segundo vira
(ex.: :11 -> :12). Nesse frame o relógio vale exatamente o novo segundo, o que dá o
início com erro de ~1 frame (< 0,05 s), melhor que a leitura manual (±0,3 s).

Para uma instalação com outro layout/fonte de relógio, reaprenda com
    python -m utils.osd_learn <video> "AAAA-MM-DD HH:MM:SS.s"
"""

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from config.settings import OSD_FONT_PATH

NORM_W, NORM_H = 1920, 1080          # o layout é descrito numa tela 1920x1080
# Esquerda de cada célula de dígito: DD MM AAAA HH MM SS (14 dígitos)
CELL_X = [1543, 1564, 1600, 1621, 1656, 1677, 1698, 1718, 1769, 1791, 1822, 1842, 1874, 1894]
CELL_Y0, CELL_Y1, CELL_W = 957, 986, 23
SHIFT = 2                             # tolerância de alinhamento (px)
MIN_SCORE = 0.55                      # semelhança mínima por dígito


def _norm(frame: np.ndarray) -> np.ndarray:
    if frame.shape[1] != NORM_W or frame.shape[0] != NORM_H:
        frame = cv2.resize(frame, (NORM_W, NORM_H), interpolation=cv2.INTER_AREA)
    return frame


def cell_masks(frame: np.ndarray) -> list[np.ndarray]:
    """Máscara (0/1) do texto branco de cada célula, com margem para o ajuste de alinhamento."""
    f = _norm(frame)
    out = []
    for x in CELL_X:
        c = f[CELL_Y0:CELL_Y1, x - SHIFT:x + CELL_W + SHIFT]
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[..., 1]
        out.append(((gray > 190) & (sat < 70)).astype(np.float32))
    return out


def _score(cell: np.ndarray, tpl: np.ndarray) -> float:
    """IoU do texto, testando pequenos deslocamentos horizontais."""
    best = 0.0
    for dx in range(0, 2 * SHIFT + 1):
        c = cell[:, dx:dx + CELL_W]
        inter = float((c * tpl).sum())
        union = float(np.maximum(c, tpl).sum())
        best = max(best, inter / union if union else 0.0)
    return best


class OSDClock:
    def __init__(self, font_path: Path = OSD_FONT_PATH):
        self.templates = None
        if Path(font_path).exists():
            data = np.load(font_path)
            self.templates = {int(k[1:]): data[k] for k in data.files if k.startswith("d")}

    @property
    def ready(self) -> bool:
        return bool(self.templates) and len(self.templates) == 10

    def read(self, frame: np.ndarray) -> tuple[datetime | None, float]:
        """Retorna (data/hora lida, confiança mínima entre os dígitos) ou (None, conf)."""
        if not self.ready:
            return None, 0.0
        digits, worst = [], 1.0
        for cell in cell_masks(frame):
            scores = {d: _score(cell, t) for d, t in self.templates.items()}
            d = max(scores, key=scores.get)
            digits.append(str(d))
            worst = min(worst, scores[d])
        if worst < MIN_SCORE:
            return None, worst
        s = "".join(digits)
        try:
            return datetime.strptime(s, "%d%m%Y%H%M%S"), worst
        except ValueError:
            return None, worst

    def start_time(self, path: Path, probe_s: float = 6.0) -> tuple[datetime | None, str]:
        """
        Horário real do frame 0 de um vídeo, pelo relógio da imagem.
        Procura viradas de segundo nos primeiros segundos; em cada virada, início = novo
        segundo - (frame / fps). Usa a mediana das viradas (robusto a um frame mal lido).
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, "não abriu o vídeo"
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        n = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), probe_s * fps))
        readings, flips = [], 0
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            t, _conf = self.read(frame)
            if t is not None:
                if readings and readings[-1][0] == i - 1 and (t - readings[-1][1]) == timedelta(seconds=1):
                    flips += 1
                readings.append((i, t))
                if flips >= 3:            # 3 viradas bastam (mediana robusta); poupa decodificação 4K
                    break
        cap.release()
        if len(readings) < 3:
            return None, "relógio não reconhecido (layout/fonte diferente?)"

        starts = []
        for (i0, t0), (i1, t1) in zip(readings, readings[1:]):
            if i1 == i0 + 1 and (t1 - t0) == timedelta(seconds=1):
                starts.append(t1 - timedelta(seconds=i1 / fps))
        if starts:
            base = min(starts)
            med = sorted((s - base).total_seconds() for s in starts)[len(starts) // 2]
            return base + timedelta(seconds=med), f"relógio lido ({len(starts)} viradas de segundo)"
        # Sem virada observada: estimativa pelo meio do segundo (±0,5 s)
        i, t = readings[0]
        return t + timedelta(seconds=0.5 - i / fps), "relógio lido (±0,5 s)"
