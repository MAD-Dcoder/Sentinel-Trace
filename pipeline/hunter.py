"""
Motor de caçada sob demanda — Sentinel-Trace.

Fluxo (nada roda sem o investigador pedir):
  1. Investigação: o investigador diz câmera de início (ex.: catraca), horário e,
     opcionalmente, como o associado estava vestido.
  2. Localizar: o Sentinel varre SÓ aquela câmera, SÓ ±N segundos daquele horário,
     e mostra as pessoas encontradas — primeiro as que batem com a descrição.
  3. Alvo: o investigador escolhe quem é. Isso vira a âncora (o "target").
  4. Caçada em rodadas: a cada rodada o Sentinel varre só as câmeras prováveis
     (vizinhas aprendidas, ou as escolhidas pelo investigador) na janela de tempo logo
     após o último avistamento, ranqueia candidatos, e o investigador confirma.
     Cada confirmação: alarga a "memória visual" do alvo, ensina a calibração e
     registra a passagem câmera A -> câmera B (o mapa do galpão se monta sozinho).

Economia de processamento:
  - lê só a janela pedida (nunca o vídeo inteiro);
  - filtro de movimento: sem movimento, não chama a IA de detecção;
  - ReID só nos 10 melhores recortes de cada pessoa;
  - cache: trecho já varrido no caso nunca é processado de novo;
  - detector em OpenVINO (otimizado para CPU Intel).
"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    CASES_DIR, KNOWLEDGE_DB, SAMPLE_FPS, IDLE_DETECT_EVERY, TRACKLET_MIN_DETECTIONS,
    HUNT_BACK_S, HUNT_FORWARD_S, HUNT_MAX_CAMERAS, SEARCH_TOP_N, SEARCH_ACCEPT_PROB,
)
from core.attributes import tracklet_attributes, description_match
from core.detector import PersonDetector, CameraTracker
from core.matcher import Calibrator, TargetSearch, key
from core.motion import MotionGate
from core.reid import ReIDExtractor
from core.tracklet import Tracklet, Sample, crop_quality
from core.video import VideoArchive
from db.database import CaseDB, KnowledgeDB


class Engine:
    """Modelos carregados uma única vez e reaproveitados por todos os casos."""

    def __init__(self):
        self.archive = VideoArchive()
        self.detector = PersonDetector()
        self.reid = ReIDExtractor()
        self.knowledge = KnowledgeDB(KNOWLEDGE_DB)


def crops_strip(crops: list, h: int = 160, max_n: int = 5) -> np.ndarray:
    crops = [c for c in crops if c is not None and c.size][:max_n]
    if not crops:
        return np.zeros((h, h // 2, 3), np.uint8)
    tiles = [cv2.resize(c, (max(1, int(c.shape[1] * h / c.shape[0])), h)) for c in crops]
    return np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT) for t in tiles])


def _uncovered(want: tuple, covered: list[tuple], min_len: float = 1.0) -> list[tuple]:
    """Partes de [a, b] que ainda não foram varridas neste caso."""
    a, b = want
    gaps, cur = [], a
    for c0, c1 in sorted(covered):
        if c1 <= cur or c0 >= b:
            continue
        if c0 > cur:
            gaps.append((cur, c0))
        cur = max(cur, c1)
    if cur < b:
        gaps.append((cur, b))
    return [(x, y) for x, y in gaps if y - x >= min_len]


class Case:
    def __init__(self, engine: Engine, name: str):
        self.engine, self.name = engine, name
        self.dir = CASES_DIR / name
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        self.db = CaseDB(self.dir / "sentinel.db")
        for cam in engine.archive.cameras.values():
            self.db.save_camera(cam)
        self.stats = {"frames_lidos": 0, "frames_com_ia": 0, "segundos": 0.0}

    # ---- varredura de uma janela ------------------------------------------
    def scan(self, camera_id: str, t0: datetime, t1: datetime, progress=None) -> int:
        """Varre [t0, t1] numa câmera, pulando o que já foi varrido. Retorna nº de tracklets novos."""
        new = 0
        for a, b in _uncovered((t0.timestamp(), t1.timestamp()), self.db.coverage(camera_id)):
            new += self._scan_gap(camera_id, datetime.fromtimestamp(a), datetime.fromtimestamp(b), progress)
        return new

    def _scan_gap(self, camera_id, t0, t1, progress) -> int:
        eng, tic = self.engine, time.perf_counter()
        tracker, gate = CameraTracker(SAMPLE_FPS), MotionGate(SAMPLE_FPS)
        base = self.db.next_local_id_base(camera_id)
        trks: dict[int, Tracklet] = {}
        read = detected = idle = 0
        total = max(1, int((t1 - t0).total_seconds() * SAMPLE_FPS))

        for frame_idx, when, frame in eng.archive.read_window(camera_id, t0, t1, SAMPLE_FPS):
            read += 1
            moving = gate.has_motion(frame)
            if not moving:
                idle += 1
            # Sem movimento e ninguém rastreado: corredor vazio, pula a IA.
            # Sem movimento mas com alguém rastreado (parado no bin): detecta de vez em quando.
            if not moving and (not tracker.has_active or idle % IDLE_DETECT_EVERY):
                continue
            if moving:
                idle = 0
            detected += 1
            ts = when.timestamp()
            for d in tracker.update(eng.detector.detect(frame), frame):
                lid = base + d["local_id"]
                x1, y1, x2, y2 = d["bbox"]
                q = crop_quality(d["bbox"], d["confidence"], frame.shape)
                trks.setdefault(lid, Tracklet(camera_id, lid)).add(
                    Sample(frame_idx, ts, d["bbox"], q, frame[y1:y2, x1:x2].copy()))
                self.db.add_detection(camera_id, lid, frame_idx, ts, d["bbox"], d["confidence"], q)
            if progress:
                progress(min(1.0, read / total), f"{camera_id} {when:%H:%M:%S}")

        new = 0
        for lid, trk in trks.items():
            if trk.n < TRACKLET_MIN_DETECTIONS:
                continue
            top = trk.finalize()
            for s, e in zip(top, eng.reid.extract_batch([s.crop for s in top])):  # ReID só nos melhores
                s.embedding = e
            crops = [s.crop for s in top]
            rel = Path("crops") / f"{camera_id}__{lid}.jpg"
            cv2.imwrite(str(self.dir / rel), crops_strip(crops))
            self.db.save_tracklet(trk, str(rel), tracklet_attributes(crops))
            new += 1
        self.db.flush()
        self.db.add_coverage(camera_id, t0.timestamp(), t1.timestamp(), read, detected)
        self.stats["frames_lidos"] += read
        self.stats["frames_com_ia"] += detected
        self.stats["segundos"] += time.perf_counter() - tic
        return new

    # ---- passo 2: localizar pessoas na câmera de início --------------------
    def locate(self, camera_id: str, when: datetime, window_s: float, wanted: dict, progress=None) -> list[dict]:
        t0, t1 = when - timedelta(seconds=window_s), when + timedelta(seconds=window_s)
        self.scan(camera_id, t0, t1, progress)
        people = self.db.tracklets(camera_id, t0.timestamp(), t1.timestamp())
        for p in people:
            p["match_desc"] = description_match(p["attrs"], wanted)
            p["dist_s"] = 0.0 if p["t_start"] <= when.timestamp() <= p["t_end"] else \
                min(abs(p["t_start"] - when.timestamp()), abs(p["t_end"] - when.timestamp()))
        people.sort(key=lambda p: (-p["match_desc"], p["dist_s"]))
        return people


class Hunt:
    """Caçada a partir de uma âncora, em rodadas confirmadas pelo investigador."""

    def __init__(self, case: Case, anchor: tuple, wanted: dict | None = None, search_id: int | None = None):
        self.case, self.anchor, self.wanted = case, anchor, wanted or {}
        self.calib = Calibrator.fit(case.db.tracklets(), case.engine.knowledge.labels())
        self.search = TargetSearch(case.db.tracklets(), anchor, self.calib)
        self.step = 0
        if search_id is None:
            self.search_id = case.db.new_search(anchor, "investigador", self.calib.as_dict())
            case.db.save_hit(self.search_id, 0, *anchor, 1.0, 1.0, None, "ancora", "investigador")
        else:
            self.search_id = search_id
            _, hits = case.db.search_rows(search_id)
            for step, cam, lid, *_rest, decision, _by in hits:
                if decision == "aceito":
                    self.search.accepted.append((cam, lid))
                elif decision == "rejeitado":
                    self.search.rejected.add((cam, lid))
                self.step = max(self.step, step)

    @property
    def route(self) -> list[dict]:
        return sorted((self.search.by_key[k] for k in self.search.accepted), key=lambda t: t["t_start"])

    def suggest_cameras(self) -> list[str]:
        """Câmeras a varrer na próxima rodada: vizinhas aprendidas primeiro; sem mapa ainda, todas."""
        last = self.route[-1]
        learned = [c for c, _n, _dt in self.case.engine.knowledge.neighbors(last["camera_id"])]
        all_cams = list(self.case.engine.archive.cameras)
        if learned:
            return (learned + [last["camera_id"]])[:HUNT_MAX_CAMERAS]
        return all_cams[:HUNT_MAX_CAMERAS]

    def run_round(self, cameras: list[str], back_s=HUNT_BACK_S, forward_s=HUNT_FORWARD_S, progress=None) -> list[dict]:
        last = self.route[-1]
        t0 = datetime.fromtimestamp(last["t_start"] - back_s)
        t1 = datetime.fromtimestamp(last["t_end"] + forward_s)
        for i, cam in enumerate(cameras):
            sub = (lambda f, m, i=i: progress((i + f) / len(cameras), m)) if progress else None
            self.case.scan(cam, t0, t1, sub)
        # Recarrega tracklets (novos entraram) mantendo decisões já tomadas
        accepted, rejected = self.search.accepted, self.search.rejected
        self.calib = Calibrator.fit(self.case.db.tracklets(), self.case.engine.knowledge.labels())
        self.search = TargetSearch(self.case.db.tracklets(), self.anchor, self.calib)
        self.search.accepted, self.search.rejected = accepted, rejected
        return self.candidates()

    def candidates(self, n: int = SEARCH_TOP_N) -> list[dict]:
        cands = self.search.candidates(n)
        for c in cands:
            c["match_desc"] = description_match(c["tracklet"]["attrs"], self.wanted)
        return cands

    def decide(self, cand: dict, same: bool, by: str = "investigador"):
        self.step += 1
        k, t = cand["key"], cand["tracklet"]
        know = self.case.engine.knowledge
        if same:
            prev = self.route[-1]
            self.search.accept(k)
            if prev["camera_id"] != t["camera_id"]:   # aprende o mapa: de onde veio -> para onde foi
                know.add_transition(prev["camera_id"], t["camera_id"], t["t_start"] - prev["t_end"])
        else:
            self.search.reject(k)
        if by == "investigador":                       # decisões automáticas não ensinam (seria circular)
            know.add_label(cand["score"], same, t["camera_id"] != self.anchor[0])
        self.case.db.save_hit(self.search_id, self.step, *k, cand["score"], cand["prob"], cand["margin"],
                              "aceito" if same else "rejeitado", by)

    def auto_round(self, cameras: list[str], progress=None) -> int:
        """Modo automático (lote/testes): aceita enquanto prob >= limiar. Sempre revisar depois."""
        accepted = 0
        self.run_round(cameras, progress=progress)
        while True:
            c = self.candidates(1)
            if not c or c[0]["prob"] < SEARCH_ACCEPT_PROB:
                return accepted
            self.decide(c[0], True, by="auto")
            accepted += 1

    def close(self):
        for r in self.search.ranked()[:5]:   # auditoria: quem mais poderia ser
            self.case.db.save_hit(self.search_id, 999, *r["key"], r["score"], r["prob"], r["margin"], "nao_aceito", "-")
