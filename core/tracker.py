"""
Módulo de Rastreio Cross-Camera - Sentinel-Trace.
Mapeia trajetórias locais persistentes a identidades corporativas globais.
"""

from datetime import datetime, timezone
import numpy as np
from config.settings import REID_SIMILARITY_THRESHOLD


class GlobalTracker:
    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold
        self.global_tracks = {}  # {global_id: {"embedding": np.ndarray, "history": list}}
        self.local_to_global_map = {}  # {(camera_id, local_id): global_id}
        self.next_global_id = 1

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def resolve_global_identity(self, camera_id: str, local_id: int, embedding: np.ndarray, bbox: list) -> int:
        track_key = (camera_id, local_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "local_track_id": local_id,
            "bbox": bbox
        }

        # 1. Se o ByteTrack já travou este indivíduo nesta câmara, mantém o mesmo Global ID estritamente
        if track_key in self.local_to_global_map:
            global_id = self.local_to_global_map[track_key]
            # Atualiza o vetor de forma suave
            self.global_tracks[global_id]["embedding"] = (
                0.90 * self.global_tracks[global_id]["embedding"] + 0.10 * embedding
            )
            norm = np.linalg.norm(self.global_tracks[global_id]["embedding"])
            if norm > 0:
                self.global_tracks[global_id]["embedding"] /= norm

            self.global_tracks[global_id]["history"].append(entry)
            return global_id

        # 2. Primeira vez que este local_id surge: busca na galeria global por afinidade de aparência
        best_id = None
        best_sim = -1.0
        for gid, data in self.global_tracks.items():
            sim = self.cosine_similarity(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= self.similarity_threshold and best_id is not None:
            assigned_id = best_id
            self.global_tracks[assigned_id]["history"].append(entry)
        else:
            assigned_id = self.next_global_id
            self.global_tracks[assigned_id] = {
                "embedding": embedding.copy(),
                "history": [entry]
            }
            self.next_global_id += 1

        # Fixa o mapeamento
        self.local_to_global_map[track_key] = assigned_id
        return assigned_id