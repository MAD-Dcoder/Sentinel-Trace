"""
Tracklets — Sentinel-Trace.

Um TRACKLET é o trecho contínuo de uma pessoa numa câmera. Durante a leitura, ele só
guarda os recortes candidatos (com nota de qualidade). A assinatura de ReID — a parte
cara — é calculada NO FINAL e apenas para os K melhores recortes (nítidos, grandes,
sem corte de borda, sem carrinho na frente), espaçados no tempo para variar a pose.
"""

from dataclasses import dataclass, field
import numpy as np

from config.settings import TRACKLET_TOPK, TRACKLET_TOPK_MIN_GAP_S


def crop_quality(bbox, conf: float, frame_shape) -> float:
    """0..1: confiança × tamanho × não-cortado × proporção de corpo inteiro."""
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    bh, bw = y2 - y1, max(1, x2 - x1)
    size = min(1.0, bh / 200.0)
    touches_border = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
    body = 1.0 if 1.6 <= bh / bw <= 4.5 else 0.6     # meio-corpo atrás de carrinho fica "achatado"
    return float(conf * size * (0.5 if touches_border else 1.0) * body)


@dataclass(eq=False)   # comparação por identidade (há arrays numpy nos campos)
class Sample:
    frame_idx: int
    ts: float                       # horário real (epoch, segundos)
    bbox: list
    quality: float
    crop: np.ndarray | None = None
    embedding: np.ndarray | None = None


@dataclass
class Tracklet:
    camera_id: str
    local_id: int
    n: int = 0
    t_start: float = 0.0
    t_end: float = 0.0
    pool: list = field(default_factory=list)   # recortes candidatos (podado continuamente)

    def add(self, s: Sample):
        if self.n == 0:
            self.t_start = s.ts
        self.t_end = s.ts
        self.n += 1
        self.pool.append(s)
        if len(self.pool) > TRACKLET_TOPK * 3:
            self.pool = select_topk(self.pool)

    def finalize(self) -> list:
        """Fixa os K melhores recortes; retorna-os para o ReID calcular as assinaturas."""
        self.pool = select_topk(self.pool)
        return self.pool

    @property
    def embeddings(self) -> np.ndarray:
        e = [s.embedding for s in self.pool if s.embedding is not None]
        return np.stack(e) if e else np.zeros((0, 512), np.float32)

    @property
    def mean_embedding(self) -> np.ndarray:
        e = self.embeddings
        if not len(e):
            return np.zeros(512, np.float32)
        wts = np.array([s.quality for s in self.pool if s.embedding is not None], np.float32) + 1e-6
        m = (e * wts[:, None]).sum(0)
        return (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)


def select_topk(samples: list, k: int = TRACKLET_TOPK, min_gap: float = TRACKLET_TOPK_MIN_GAP_S) -> list:
    ranked = sorted(samples, key=lambda s: s.quality, reverse=True)
    chosen = []
    for s in ranked:
        if all(abs(s.ts - c.ts) >= min_gap for c in chosen):
            chosen.append(s)
            if len(chosen) == k:
                break
    if len(chosen) < min(k, len(samples)):   # tracklet curto: completa sem a regra de espaçamento
        chosen += [s for s in ranked if s not in chosen][: k - len(chosen)]
    return sorted(chosen, key=lambda s: s.ts)
