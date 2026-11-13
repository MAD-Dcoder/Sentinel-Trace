"""
Persistência — Sentinel-Trace.

CaseDB  (casos/<caso>/sentinel.db) — tudo de UMA investigação:
  case_info   dados do caso (investigador, pessoa investigada, descrição do alvo)
  cameras     câmeras e horário real de início de cada gravação
  coverage    trechos (câmera, início, fim) já varridos — nunca processa o mesmo trecho 2x
  detections  caixas rastreadas (sem assinatura biométrica)
  tracklets   resumo por pessoa/câmera: horário, assinaturas dos top-K crops, cores
  searches    caçadas executadas (âncora) | hits: candidatos avaliados e decisão

KnowledgeDB (db/conhecimento.db) — o que o Sentinel APRENDE entre casos, sem imagens:
  labels       score × "mesma pessoa?" das decisões humanas (melhora a % de match)
  transitions  câmera A -> câmera B e tempo de trânsito (vira o mapa do galpão)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS case_info (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY, file TEXT, start_time TEXT, fps REAL, width INTEGER, height INTEGER, n_frames INTEGER
);
CREATE TABLE IF NOT EXISTS coverage (camera_id TEXT, t0 REAL, t1 REAL, frames_read INTEGER, frames_detected INTEGER);
CREATE TABLE IF NOT EXISTS detections (
    camera_id TEXT, local_id INTEGER, frame_idx INTEGER, ts REAL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, conf REAL, quality REAL
);
CREATE INDEX IF NOT EXISTS ix_det ON detections(camera_id, ts);
CREATE TABLE IF NOT EXISTS tracklets (
    camera_id TEXT, local_id INTEGER, t_start REAL, t_end REAL, n_detections INTEGER,
    mean_emb BLOB, top_embs BLOB, top_ts TEXT, crops_path TEXT, attrs TEXT,
    PRIMARY KEY (camera_id, local_id)
);
CREATE TABLE IF NOT EXISTS searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, anchor_camera TEXT,
    anchor_local_id INTEGER, mode TEXT, calibration TEXT
);
CREATE TABLE IF NOT EXISTS hits (
    search_id INTEGER, step INTEGER, camera_id TEXT, local_id INTEGER,
    set_score REAL, prob REAL, margin REAL, decision TEXT, decided_by TEXT
);
"""

KNOWLEDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (score REAL, same INTEGER, cross_camera INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS transitions (cam_from TEXT, cam_to TEXT, dt REAL, created_at TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CaseDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.executescript(CASE_SCHEMA)
        self._det = []

    # ---- caso -------------------------------------------------------------
    def set_info(self, info: dict):
        self.conn.executemany("INSERT OR REPLACE INTO case_info VALUES (?,?)",
                              [(k, json.dumps(v, ensure_ascii=False)) for k, v in info.items()])
        self.conn.commit()

    def info(self) -> dict:
        return {k: json.loads(v) for k, v in self.conn.execute("SELECT key, value FROM case_info")}

    def save_camera(self, cam):
        self.conn.execute("INSERT OR REPLACE INTO cameras VALUES (?,?,?,?,?,?,?)",
                          (cam.camera_id, cam.path.name, cam.start_time.isoformat(), cam.fps,
                           cam.width, cam.height, cam.n_frames))
        self.conn.commit()

    # ---- cobertura (cache de trechos varridos) ----------------------------
    def coverage(self, camera_id: str) -> list[tuple]:
        return list(self.conn.execute("SELECT t0, t1 FROM coverage WHERE camera_id=? ORDER BY t0", (camera_id,)))

    def add_coverage(self, camera_id, t0, t1, frames_read, frames_detected):
        self.conn.execute("INSERT INTO coverage VALUES (?,?,?,?,?)", (camera_id, t0, t1, frames_read, frames_detected))
        self.conn.commit()

    def next_local_id_base(self, camera_id: str) -> int:
        r = self.conn.execute("SELECT MAX(local_id) FROM tracklets WHERE camera_id=?", (camera_id,)).fetchone()[0]
        return ((r or 0) // 10000 + 1) * 10000

    # ---- detecções e tracklets --------------------------------------------
    def add_detection(self, camera_id, local_id, frame_idx, ts, bbox, conf, quality):
        self._det.append((camera_id, local_id, frame_idx, ts, *bbox, conf, quality))
        if len(self._det) >= 500:
            self.flush()

    def flush(self):
        if self._det:
            self.conn.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?)", self._det)
            self.conn.commit()
            self._det.clear()

    def save_tracklet(self, trk, crops_path: str, attrs: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO tracklets VALUES (?,?,?,?,?,?,?,?,?,?)",
            (trk.camera_id, trk.local_id, trk.t_start, trk.t_end, trk.n,
             trk.mean_embedding.astype(np.float32).tobytes(), trk.embeddings.astype(np.float32).tobytes(),
             json.dumps([round(s.ts, 2) for s in trk.pool]), crops_path, json.dumps(attrs, ensure_ascii=False)))
        self.conn.commit()

    def tracklets(self, camera_id: str | None = None, t0: float | None = None, t1: float | None = None) -> list[dict]:
        q, args = "SELECT * FROM tracklets WHERE 1=1", []
        if camera_id:
            q += " AND camera_id=?"; args.append(camera_id)
        if t0 is not None:
            q += " AND t_end>=?"; args.append(t0)
        if t1 is not None:
            q += " AND t_start<=?"; args.append(t1)
        out = []
        for cam, lid, a, b, n, mean_b, top_b, top_ts, crops, attrs in self.conn.execute(q + " ORDER BY t_start", args):
            out.append(dict(camera_id=cam, local_id=lid, t_start=a, t_end=b, n=n,
                            mean=np.frombuffer(mean_b, np.float32),
                            top=np.frombuffer(top_b, np.float32).reshape(-1, 512),
                            top_ts=json.loads(top_ts), crops_path=crops, attrs=json.loads(attrs or "{}")))
        return out

    # ---- caçadas ------------------------------------------------------------
    def new_search(self, anchor: tuple, mode: str, calibration: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO searches (created_at, anchor_camera, anchor_local_id, mode, calibration) VALUES (?,?,?,?,?)",
            (now_iso(), anchor[0], anchor[1], mode, json.dumps(calibration)))
        self.conn.commit()
        return cur.lastrowid

    def save_hit(self, search_id, step, camera_id, local_id, score, prob, margin, decision, decided_by):
        self.conn.execute("INSERT INTO hits VALUES (?,?,?,?,?,?,?,?,?)",
                          (search_id, step, camera_id, local_id, score, prob, margin, decision, decided_by))
        self.conn.commit()

    def search_rows(self, search_id: int):
        s = self.conn.execute("SELECT * FROM searches WHERE id=?", (search_id,)).fetchone()
        hits = list(self.conn.execute(
            "SELECT step, camera_id, local_id, set_score, prob, margin, decision, decided_by "
            "FROM hits WHERE search_id=? ORDER BY step, prob DESC", (search_id,)))
        return s, hits

    def latest_search_id(self) -> int | None:
        return self.conn.execute("SELECT MAX(id) FROM searches").fetchone()[0]

    def close(self):
        self.flush()
        self.conn.close()


class KnowledgeDB:
    def __init__(self, db_path: Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(KNOWLEDGE_SCHEMA)

    def add_label(self, score: float, same: bool, cross_camera: bool):
        self.conn.execute("INSERT INTO labels VALUES (?,?,?,?)", (score, int(same), int(cross_camera), now_iso()))
        self.conn.commit()

    def labels(self) -> list[tuple]:
        return list(self.conn.execute("SELECT score, same FROM labels"))

    def add_transition(self, cam_from: str, cam_to: str, dt: float):
        self.conn.execute("INSERT INTO transitions VALUES (?,?,?,?)", (cam_from, cam_to, dt, now_iso()))
        self.conn.commit()

    def neighbors(self, camera_id: str) -> list[tuple]:
        """[(câmera, nº de vezes vista em seguida, tempo de trânsito mediano)] — o mapa aprendido."""
        rows = self.conn.execute("SELECT cam_to, dt FROM transitions WHERE cam_from=?", (camera_id,)).fetchall()
        by = {}
        for cam, dt in rows:
            by.setdefault(cam, []).append(dt)
        return sorted(((c, len(d), float(np.median(d))) for c, d in by.items()), key=lambda r: -r[1])

    def edges(self) -> list[tuple]:
        return list(self.conn.execute(
            "SELECT cam_from, cam_to, COUNT(*), AVG(dt) FROM transitions GROUP BY cam_from, cam_to"))
