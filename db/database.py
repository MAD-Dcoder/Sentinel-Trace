"""
Módulo de Persistência Forense - Sentinel-Trace.
Armazena eventos de deteção, IDs globais e vetores biométricos em SQLite.
"""

import sqlite3
import numpy as np
from pathlib import Path
from config.settings import DB_DIR

DB_FILE = DB_DIR / "sentinel.db"


class ForensicDatabase:
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Cria as tabelas de auditoria caso ainda não existam."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    global_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    local_track_id INTEGER,
                    timestamp TEXT NOT NULL,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def save_event(self, global_id: int, camera_id: str, local_track_id: int, timestamp: str, bbox: list, embedding: np.ndarray):
        """Insere um evento forense no registo central."""
        embedding_blob = embedding.astype(np.float32).tobytes()
        x1, y1, x2, y2 = bbox

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO events (global_id, camera_id, local_track_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (global_id, camera_id, local_track_id, timestamp, x1, y1, x2, y2, embedding_blob))
            conn.commit()

    def get_events_by_global_id(self, global_id: int):
        """Recupera a linha cronológica completa de um indivíduo específico."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, camera_id, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                FROM events
                WHERE global_id = ?
                ORDER BY timestamp ASC
            """, (global_id,))
            return cursor.fetchall()