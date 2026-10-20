"""
Módulo de Geração de Relatórios Periciais - Sentinel-Trace.
Compila timelines forenses e evidências de rastreio em documentos formais (.docx).
"""

from datetime import datetime, timezone
from io import BytesIO
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def generate_forensic_report(global_id: int, events_df: pd.DataFrame) -> BytesIO:
    """
    Gera um relatório forense formal contendo metadados do incidente,
    sumário executivo e a cronologia de rastreamento entre câmaras.
    """
    doc = Document()

    # Configuração de estilo de cabeçalho
    title = doc.add_heading(level=0)
    run_title = title.add_run("🛡️ SENTINEL-TRACE — LAUDO DE AUDITORIA FORENSE")
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    # Subtítulo com dados do documento
    now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y às %H:%M:%S UTC")
    p_meta = doc.add_paragraph()
    p_meta.add_run(f"Data de Emissão: {now_str}\n").italic = True
    p_meta.add_run(f"Identificador Auditado: ID Global #{global_id}\n").bold = True
    p_meta.add_run("Natureza: Rastreamento Cross-Camera e Prevenção de Perdas\n")

    doc.add_heading("1. Sumário do Incidente", level=1)
    p_summary = doc.add_paragraph(
        "Este documento consolida os registros extraídos automaticamente pelo pipeline do Sentinel-Trace. "
        "A identificação e a correlação entre câmaras foram estabelecidas via vetores biométricos corporais (ReID), "
        "garantindo conformidade com a privacidade individual e isolamento pericial dos dados."
    )

    doc.add_heading("2. Trajetória Cronológica Identificada", level=1)

    if events_df.empty:
        doc.add_paragraph("Nenhum registro encontrado para este identificador.")
    else:
        # Criação da tabela de evidências
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"

        hdr_cells = table.rows[0].cells
        headers = ["Câmara / Ponto", "Data e Hora (UTC)", "Track Local", "Coordenadas (Bounding Box)"]
        for i, header_text in enumerate(headers):
            hdr_cells[i].text = header_text
            hdr_cells[i].paragraphs[0].runs[0].font.bold = True

        for _, row in events_df.iterrows():
            row_cells = table.add_row().cells
            row_cells[0].text = str(row["camera_id"])
            row_cells[1].text = str(row["timestamp"])
            row_cells[2].text = str(row["local_track_id"])
            bbox_str = f"[{row['bbox_x1']}, {row['bbox_y1']}, {row['bbox_x2']}, {row['bbox_y2']}]"
            row_cells[3].text = bbox_str

    doc.add_heading("3. Declaração de Integridade", level=1)
    doc.add_paragraph(
        "Os registros acima refletem os dados persistidos no banco relacional e vetorial de auditoria. "
        "A integridade dos vetores matemáticos assegura a rastreabilidade entre diferentes feeds de CCTV."
    )

    # Salva o arquivo em buffer de memória para download direto
    file_stream = BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream