"""
Módulo de Gestão de Rastreio Cross-Camera - Sentinel-Trace.
Mantém o registo de identidades globais e associa indivíduos entre câmaras distintas.
"""

from datetime import datetime
import numpy as np
from config.settings import REID_SIMILARITY_THRESHOLD


class GlobalTracker:
    def __init__(self, similarity_threshold: float = REID_SIMILARITY_THRESHOLD):
        """
        Gere a base de identidades globais observadas no sistema de vigilância.
        """
        self.similarity_threshold = similarity_threshold
        self.global_tracks = {}  # {global_id: {"embedding": np.ndarray, "history": list}}
        self.next_global_id = 1

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula a similaridade por cosseno entre dois vetores normalizados.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def update_track(self, camera_id: str, local_track_id: int, embedding: np.ndarray, bbox: list, crop: np.ndarray = None) -> int:
        """
        Recebe uma deteção com vetor corporal e atribui ou atualiza um ID global.
        """
        best_id = None
        best_sim = -1.0

        # Procura a identidade mais próxima na base global
        for gid, data in self.global_tracks.items():
            sim = self.cosine_similarity(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        timestamp = datetime.utcnow().isoformat()
        detection_entry = {
            "timestamp": timestamp,
            "camera_id": camera_id,
            "local_track_id": local_track_id,
            "bbox": bbox,
            "crop": crop
        }

        # Associa à identidade existente se ultrapassar o limiar de afinidade
        if best_sim >= self.similarity_threshold and best_id is not None:
            assigned_id = best_id
            # Atualização suave do vetor da pessoa com média móvel exponencial
            self.global_tracks[assigned_id]["embedding"] = (
                0.85 * self.global_tracks[assigned_id]["embedding"] + 0.15 * embedding
            )
            # Re-normalização L2
            norm = np.linalg.norm(self.global_tracks[assigned_id]["embedding"])
            if norm > 0:
                self.global_tracks[assigned_id]["embedding"] /= norm

            self.global_tracks[assigned_id]["history"].append(detection_entry)
        else:
            # Cria um novo indivíduo no circuito do Sentinel-Trace
            assigned_id = self.next_global_id
            self.global_tracks[assigned_id] = {
                "embedding": embedding.copy(),
                "history": [detection_entry]
            }
            self.next_global_id += 1

        return assigned_id