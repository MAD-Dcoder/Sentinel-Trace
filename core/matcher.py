"""
Busca do alvo e calibração de probabilidade — Sentinel-Trace.

1) set_score: similaridade entre DOIS CONJUNTOS de crops (tracklet × tracklet ou
   galeria do alvo × tracklet) = média dos 25% melhores pares de cosseno. Robusto a
   poses ruins e a meio-corpo, sem depender de um único frame bom.

2) Calibrator: transforma set_score em P(mesma pessoa) com regressão logística, usando
   pares que sabemos a resposta SEM rótulo humano:
     - impostores: dois tracklets visíveis AO MESMO TEMPO na mesma câmera (com certeza
       são pessoas diferentes);
     - genuínos: primeira metade × segunda metade do mesmo tracklet longo;
   mais todos os rótulos humanos (confirmações/rejeições) já salvos na sessão.
   Atenção: genuínos intra-câmera são mais "fáceis" que entre câmeras, então a curva
   tende a SUBESTIMAR matches entre câmeras (nos vídeos de teste, matches corretos
   cross-camera ficaram em 40-55%). Use o RANKING + margem; a probabilidade só fica
   confiável depois de acumular rótulos humanos entre câmeras.

3) TargetSearch: parte do tracklet âncora, ranqueia todos os outros tracklets, aceita
   o melhor (humano ou --auto), expande a galeria do alvo com os crops aceitos e repete.
   Restrição física: o alvo não pode estar em dois tracklets simultâneos da mesma câmera.
"""

import numpy as np

from config.settings import SEARCH_ACCEPT_PROB, SEARCH_MIN_PROB, QUERY_MAX_EMBEDDINGS

DEFAULT_CALIB = {"a": 14.0, "b": -9.0, "source": "padrão (sem dados suficientes)"}


def set_score(A: np.ndarray, B: np.ndarray) -> float:
    if not len(A) or not len(B):
        return 0.0
    sims = (A @ B.T).ravel()
    k = max(1, len(sims) // 4)
    return float(np.mean(np.partition(sims, -k)[-k:]))


def overlap_s(a: dict, b: dict) -> float:
    return min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])


def key(t: dict) -> tuple:
    return (t["camera_id"], t["local_id"])


class Calibrator:
    def __init__(self, a=DEFAULT_CALIB["a"], b=DEFAULT_CALIB["b"], source=DEFAULT_CALIB["source"], stats=None):
        self.a, self.b, self.source, self.stats = a, b, source, stats or {}

    def prob(self, score: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.a * score + self.b))))

    def as_dict(self) -> dict:
        return {"a": round(self.a, 3), "b": round(self.b, 3), "source": self.source, **self.stats}

    @classmethod
    def fit(cls, tracklets: list[dict], labels: list[tuple] = ()) -> "Calibrator":
        """labels: [(score, mesma_pessoa)] das decisões humanas de TODOS os casos (KnowledgeDB)."""
        gen, imp = [], []

        for t in tracklets:   # genuínos: metades do mesmo tracklet longo
            ts = np.array(t["top_ts"])
            if len(ts) >= 6 and (ts[-1] - ts[0]) >= 2.0:
                mid = np.median(ts)
                A, B = t["top"][ts <= mid], t["top"][ts > mid]
                if len(A) >= 2 and len(B) >= 2:
                    gen.append(set_score(A, B))

        for i, a in enumerate(tracklets):   # impostores: co-visíveis na mesma câmera
            for b in tracklets[i + 1:]:
                if a["camera_id"] == b["camera_id"] and overlap_s(a, b) >= 0.5:
                    imp.append(set_score(a["top"], b["top"]))

        n_human = len(labels)
        for score, same in labels:   # rótulos humanos (peso dobrado: são o caso difícil, entre câmeras)
            (gen if same else imp).extend([score, score])

        stats = {"n_genuinos": len(gen), "n_impostores": len(imp), "n_rotulos_humanos": n_human}
        if gen:
            stats["genuino_p50"] = round(float(np.median(gen)), 3)
        if imp:
            stats["impostor_p50"] = round(float(np.median(imp)), 3)
            stats["impostor_p95"] = round(float(np.percentile(imp, 95)), 3)
        if len(gen) < 5 or len(imp) < 5:
            return cls(stats=stats)

        from sklearn.linear_model import LogisticRegression
        x = np.array(gen + imp)[:, None]
        y = np.array([1] * len(gen) + [0] * len(imp))
        # class_weight balanceado = prior neutro (50/50); C baixo = regularização com poucas amostras
        lr = LogisticRegression(class_weight="balanced", C=10.0).fit(x, y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]), "auto-calibrado na sessão", stats)


class TargetSearch:
    def __init__(self, tracklets: list[dict], anchor: tuple, calibrator: Calibrator):
        self.by_key = {key(t): t for t in tracklets}
        if anchor not in self.by_key:
            raise KeyError(f"Tracklet âncora {anchor} não existe nesta sessão.")
        self.calib = calibrator
        self.accepted = [anchor]
        self.rejected: set[tuple] = set()

    def query(self) -> np.ndarray:
        anchor = self.by_key[self.accepted[0]]["top"]
        if len(self.accepted) == 1:
            return anchor
        rest = np.concatenate([self.by_key[k]["top"] for k in self.accepted[1:]])
        room = max(1, QUERY_MAX_EMBEDDINGS - len(anchor))
        if len(rest) > room:                       # âncora inteira + amostra uniforme dos confirmados
            rest = rest[np.linspace(0, len(rest) - 1, room).astype(int)]
        return np.concatenate([anchor, rest])

    def _conflicts(self, t: dict) -> bool:
        """O alvo não pode estar em dois tracklets ao mesmo tempo na mesma câmera."""
        return any(self.by_key[k]["camera_id"] == t["camera_id"] and overlap_s(self.by_key[k], t) > 0.3
                   for k in self.accepted)

    def ranked(self) -> list[dict]:
        Q = self.query()
        rows = []
        for k, t in self.by_key.items():
            if k in self.accepted or k in self.rejected or self._conflicts(t):
                continue
            s = set_score(Q, t["top"])
            rows.append({"key": k, "tracklet": t, "score": s, "prob": self.calib.prob(s)})
        rows.sort(key=lambda r: r["score"], reverse=True)
        # margem = distância para o melhor concorrente co-visível na mesma câmera
        # (se dois candidatos simultâneos têm score parecido, a decisão é ambígua)
        for r in rows:
            rivals = [o["score"] for o in rows if o is not r and o["key"][0] == r["key"][0]
                      and overlap_s(o["tracklet"], r["tracklet"]) > 0.3]
            r["margin"] = r["score"] - max(rivals) if rivals else None
        return rows

    def candidates(self, n: int) -> list[dict]:
        return [r for r in self.ranked() if r["prob"] >= SEARCH_MIN_PROB][:n]

    def accept(self, k: tuple):
        self.accepted.append(k)
        # Quem estava co-visível com o aceito na mesma câmera é, com certeza, outra pessoa
        t = self.by_key[k]
        for k2, t2 in self.by_key.items():
            if k2 not in self.accepted and t2["camera_id"] == t["camera_id"] and overlap_s(t, t2) > 0.3:
                self.rejected.add(k2)

    def reject(self, k: tuple):
        self.rejected.add(k)

    def auto_step(self) -> dict | None:
        c = self.candidates(1)
        if c and c[0]["prob"] >= SEARCH_ACCEPT_PROB:
            return c[0]
        return None
