"""
Módulo de Rastreio Cross-Camera e Anti-Colisão - Sentinel-Trace.
"""

from datetime import datetime, timezone
import numpy as np


class GlobalTracker:
    def __init__(self, similarity_threshold: float = 0.72):
        self.similarity_threshold = similarity_threshold
        self.global_tracks = {}           # {global_id: {"embedding": np.ndarray, "history": list}}
        self.local_to_global = {}         # {(camera_id, local_id): global_id}
        self.next_global_id = 1

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        dot = float(np.dot(v1, v2))
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)

    def update_frame(self, camera_id: str, detections: list):
        """
        Garante que CADA pessoa visível no mesmo frame tenha um ID Global único.
        """
        used_global_ids_in_frame = set()
        timestamp = datetime.now(timezone.utc).isoformat()
        unmapped_detections = []

        # 1. Mantém quem já tinha rastro local ativo no ByteTrack
        for det in detections:
            key = (camera_id, det["local_id"])
            if key in self.local_to_global:
                gid = self.local_to_global[key]
                # Se por algum erro de oclusão o ID já está em uso por outra pessoa neste frame, desvincula
                if gid not in used_global_ids_in_frame:
                    det["global_id"] = gid
                    used_global_ids_in_frame.add(gid)
                    # Atualiza o vetor suavemente
                    self.global_tracks[gid]["embedding"] = (
                        0.90 * self.global_tracks[gid]["embedding"] + 0.10 * det["embedding"]
                    )
                    norm = np.linalg.norm(self.global_tracks[gid]["embedding"])
                    if norm > 0:
                        self.global_tracks[gid]["embedding"] /= norm
                else:
                    unmapped_detections.append(det)
            else:
                unmapped_detections.append(det)

        # 2. Para novos rastros locais, busca no banco APENAS entre IDs que NÃO estão na tela agora
        for det in unmapped_detections:
            emb = det["embedding"]
            best_gid = None
            best_sim = -1.0

            for gid, data in self.global_tracks.items():
                if gid in used_global_ids_in_frame:
                    continue  # REGRA DE OURO: A mesma pessoa não pode estar em 2 lugares ao mesmo tempo!

                sim = self.cosine_similarity(emb, data["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

            if best_sim >= self.similarity_threshold and best_gid is not None:
                assigned_gid = best_gid
            else:
                assigned_gid = self.next_global_id
                self.global_tracks[assigned_gid] = {
                    "embedding": emb.copy(),
                    "history": []
                }
                self.next_global_id += 1

            key = (camera_id, det["local_id"])
            self.local_to_global[key] = assigned_gid
            used_global_ids_in_frame.add(assigned_gid)
            det["global_id"] = assigned_gid

        # 3. Registra histórico
        for det in detections:
            gid = det["global_id"]
            self.global_tracks[gid]["history"].append({
                "timestamp": timestamp,
                "camera_id": camera_id,
                "local_track_id": det["local_id"],
                "bbox": det["bbox"]
            })

        return detections