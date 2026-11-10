"""
Módulo Inspetor de Auditoria (Raio-X Forense) - Sentinel-Trace.
Filtra ruídos de borda e calcula permanência exata e compatibilidade ReID entre câmaras.
"""

import sqlite3
import numpy as np
import pandas as pd
from config.settings import DB_DIR, BASE_DIR


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def export_investigation_summary(dwell_threshold: float = 8.0, min_frames_valid: int = 6) -> str:
    db_path = DB_DIR / "sentinel.db"
    if not db_path.exists():
        msg = "[AVISO] Banco sentinel.db ainda não existe."
        print(msg)
        return msg

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM events ORDER BY id ASC", conn)
    conn.close()

    if df.empty:
        msg = "[AVISO] O banco sentinel.db está vazio."
        print(msg)
        return msg

    df["dt"] = pd.to_datetime(df["timestamp"])

    # Filtra IDs que apareceram por menos de 'min_frames_valid' no total (descarta ruídos de borda de tela)
    valid_gids = [gid for gid, g in df.groupby("global_id") if len(g) >= min_frames_valid]

    lines = []
    lines.append("=" * 80)
    lines.append("🛡️  SENTINEL-TRACE — RAIO-X DA INVESTIGAÇÃO (LOG FORENSE LP)")
    lines.append("=" * 80)
    lines.append(f"Total de Capturas Gravadas : {len(df)}")
    lines.append(f"Alvos Relevantes Rastreados: {len(valid_gids)} -> {valid_gids} (Ruídos < {min_frames_valid} frames ocultados)")
    lines.append(f"Câmaras Analisadas         : {df['camera_id'].nunique()}")
    lines.append("-" * 80)

    for gid in valid_gids:
        p_df = df[df["global_id"] == gid]
        cams_visited = []
        for c in p_df["camera_id"]:
            if not cams_visited or cams_visited[-1] != c:
                cams_visited.append(c)

        lines.append(f"\n🎯 ALVO GLOBAL #{gid} (Total: {len(p_df)} frames | {len(cams_visited)} câmara(s))")
        lines.append(f"   Rota Descoberta: {' ➔ '.join(cams_visited)}")

        cam_embeddings = {}
        for cam_id, group in p_df.groupby("camera_id", sort=False):
            t_in = group["dt"].min()
            t_out = group["dt"].max()
            dwell = (t_out - t_in).total_seconds()
            local_ids = sorted(group["local_track_id"].unique().tolist())
            status = "🚨 [ALERTA LP - PERMANÊNCIA]" if dwell >= dwell_threshold else "✅ [Normal]"

            embs = [np.frombuffer(b, dtype=np.float32) for b in group["embedding"] if b is not None]
            if embs:
                cam_embeddings[cam_id] = np.mean(embs, axis=0)

            lines.append(
                f"   • {cam_id:<36} | Tempo: {dwell:>5.1f}s | Frames: {len(group):>3} | "
                f"Rastros: {local_ids} | {status}"
            )

        cam_keys = list(cam_embeddings.keys())
        if len(cam_keys) > 1:
            lines.append("   🔗 Confiança do Match Cross-Camera (ReID):")
            for i in range(len(cam_keys) - 1):
                c1, c2 = cam_keys[i], cam_keys[i + 1]
                sim = cosine_sim(cam_embeddings[c1], cam_embeddings[c2]) * 100
                lines.append(f"      ↳ Transição [{c1}] ➔ [{c2}]: {sim:.1f}% de compatibilidade visual")

    lines.append("\n" + "=" * 80)
    report_text = "\n".join(lines)

    print("\n" + report_text)
    out_file = BASE_DIR / "resumo_investigacao.txt"
    out_file.write_text(report_text, encoding="utf-8")
    print(f"\n📄 Relatório salvo automaticamente em: {out_file.name}")

    return report_text


if __name__ == "__main__":
    export_investigation_summary()