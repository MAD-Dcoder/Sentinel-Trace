"""
Relatório forense de uma caçada — Sentinel-Trace.

Rota do alvo em ordem cronológica real, probabilidade de cada trecho, quem decidiu,
candidatos rejeitados/não aceitos (auditoria) e prancha de evidências (JPG).
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from db.database import CaseDB


def hms(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def route_rows(case_dir: Path, search_id: int) -> tuple:
    db = CaseDB(case_dir / "sentinel.db")
    s, hits = db.search_rows(search_id)
    trk = {(t["camera_id"], t["local_id"]): t for t in db.tracklets()}
    info = db.info()
    db.close()
    rows = []
    for step, cam, lid, score, prob, margin, decision, by in hits:
        t = trk.get((cam, lid))
        if t is None:
            continue
        rows.append({
            "t_start": t["t_start"], "inicio": hms(t["t_start"]), "fim": hms(t["t_end"]), "camera": cam,
            "local_id": lid, "prob": "ÂNCORA" if decision == "ancora" else f"{prob:.0%}",
            "score": f"{score:.3f}", "margem": "-" if margin is None else f"{margin:+.3f}",
            "decisao": decision, "decidido_por": by, "crops": case_dir / t["crops_path"],
            "cores": f"camisa {t['attrs'].get('camisa') or '?'} / calça {t['attrs'].get('calca') or '?'}",
        })
    return s, rows, info


def export_search_report(case_dir: Path, search_id: int | None = None) -> str:
    if search_id is None:
        db = CaseDB(case_dir / "sentinel.db")
        search_id = db.latest_search_id()
        db.close()
    if search_id is None:
        return "Nenhuma caçada neste caso."
    s, rows, info = route_rows(case_dir, search_id)
    calib = json.loads(s[5])
    route = sorted((r for r in rows if r["decisao"] in ("ancora", "aceito")), key=lambda r: r["t_start"])
    rejected = [r for r in rows if r["decisao"] == "rejeitado"]
    others = [r for r in rows if r["decisao"] == "nao_aceito"]

    info.setdefault("investigado", info.pop("associado", None))   # casos antigos
    L = ["=" * 90, f"SENTINEL-TRACE — CASO {case_dir.name} — CAÇADA #{search_id}", "=" * 90]
    for k, nome in (("investigador", "Investigador"), ("investigado", "Investigado"),
                    ("descricao", "Descrição"), ("observacoes", "Observações")):
        if info.get(k):
            L.append(f"{nome:<13}: {info[k]}")
    L += [f"Âncora       : {s[2]} #{s[3]}",
          f"Câmeras      : {len({r['camera'] for r in route})} com o alvo",
          f"Calibração   : {calib.get('source')} (rótulos humanos acumulados: {calib.get('n_rotulos_humanos')})",
          "-" * 90, "ROTA DO ALVO (horário real):"]
    for r in route:
        L.append(f"  {r['inicio']}–{r['fim']}  {r['camera']:<24} #{r['local_id']:<6} prob {r['prob']:>6}  "
                 f"margem {r['margem']:>7}  {r['cores']:<28} decisão: {r['decidido_por']}")
    if rejected:
        L.append("\nREJEITADOS PELO INVESTIGADOR:")
        L += [f"  {r['inicio']}  {r['camera']:<24} #{r['local_id']:<6} prob {r['prob']}" for r in rejected]
    if others:
        L.append("\nMELHORES CANDIDATOS NÃO AVALIADOS (revisar se o alvo pode ter sido perdido):")
        L += [f"  {r['inicio']}  {r['camera']:<24} #{r['local_id']:<6} prob {r['prob']}" for r in others]
    L += ["", "prob = chance estimada de ser a mesma pessoa (melhora a cada decisão humana).",
          "margem = vantagem sobre o concorrente simultâneo na mesma câmera (perto de 0 = ambíguo).", "=" * 90]
    text = "\n".join(L)
    (case_dir / f"cacada_{search_id}.txt").write_text(text, encoding="utf-8")
    evidence_board(route, case_dir / f"cacada_{search_id}_evidencias.jpg")
    return text


def evidence_board(route: list[dict], out_path: Path, width=1500, h=150):
    rows = []
    for r in route:
        img = cv2.imread(str(r["crops"]))
        if img is None:
            continue
        img = cv2.resize(img, (max(1, int(img.shape[1] * h / img.shape[0])), h))
        canvas = np.full((h + 28, width, 3), 25, np.uint8)
        canvas[28:, :min(width, img.shape[1])] = img[:, :width]
        label = "ANCORA" if r["decisao"] == "ancora" else f"{r['prob']} ({r['decidido_por']})"
        cv2.putText(canvas, f"{r['inicio']}  {r['camera']} #{r['local_id']}  {label}", (6, 20), 0, 0.6, (0, 255, 255), 2)
        rows.append(canvas)
    if rows:
        cv2.imwrite(str(out_path), np.vstack(rows))
