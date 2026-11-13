"""
Laudo pericial (.docx) de uma busca de alvo — Sentinel-Trace.
"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor


def generate_forensic_report(search_id: int, anchor: str, route: list[dict], calibration: dict,
                             evidence_image: Path | None = None) -> BytesIO:
    """route: [{camera, local_id, inicio, fim, prob, score, margem, decidido_por}, ...] em ordem cronológica."""
    doc = Document()
    run = doc.add_heading(level=0).add_run("SENTINEL-TRACE — LAUDO DE BUSCA FORENSE")
    run.font.size, run.font.bold, run.font.color.rgb = Pt(18), True, RGBColor(15, 23, 42)

    meta = doc.add_paragraph()
    meta.add_run(f"Emissão: {datetime.now(timezone.utc):%d/%m/%Y %H:%M:%S} UTC\n").italic = True
    meta.add_run(f"Busca #{search_id} | Âncora: {anchor}\n").bold = True
    meta.add_run(f"Câmeras com o alvo: {len({r['camera'] for r in route})}\n")

    doc.add_heading("1. Método", level=1)
    doc.add_paragraph(
        "O alvo foi indicado pelo investigador na câmera âncora. As demais aparições foram ranqueadas por "
        "similaridade corporal (ReID OSNet-AIN, sem reconhecimento facial) entre conjuntos de recortes, e cada "
        "trecho incluído na rota foi confirmado pelo investigador ou, quando indicado, aceito automaticamente. "
        "Horários são o relógio real das exportações do VMS.")
    doc.add_paragraph(
        f"Calibração: {calibration.get('source')} — genuínos={calibration.get('n_genuinos')}, "
        f"impostores={calibration.get('n_impostores')}, rótulos humanos={calibration.get('n_rotulos_humanos')}. "
        "A probabilidade é indicativa; a decisão final é humana.")

    doc.add_heading("2. Rota cronológica do alvo", level=1)
    cols = ["Início", "Fim", "Câmera", "Rastro", "Prob.", "Score", "Margem", "Decisão"]
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Table Grid"
    for cell, h in zip(table.rows[0].cells, cols):
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
    for r in route:
        vals = [r["inicio"], r["fim"], r["camera"], f"#{r['local_id']}", r["prob"], r["score"], r["margem"],
                r["decidido_por"]]
        for cell, v in zip(table.add_row().cells, vals):
            cell.text = str(v)

    if evidence_image and Path(evidence_image).exists():
        doc.add_heading("3. Evidências visuais", level=1)
        doc.add_picture(str(evidence_image), width=Inches(6.5))

    doc.add_heading("4. Integridade", level=1)
    doc.add_paragraph("Registros extraídos do banco da sessão (sentinel.db), incluindo candidatos rejeitados e "
                      "não aceitos, para permitir revisão independente.")
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf
