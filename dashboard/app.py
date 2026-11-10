"""
Interface Forense e Descoberta de Topologia - Sentinel-Trace.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import sqlite3
import pandas as pd
import streamlit as st
from config.settings import DB_DIR
from utils.report_generator import generate_forensic_report

st.set_page_config(
    page_title="Sentinel-Trace | LP Investigation & Auto-Map",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sentinel-Trace — Investigação Forense & Auto-Mapeamento de Site")
st.markdown("Varredura sob demanda, deteção de anomalias de permanência (*Dwell Time*) e descoberta de topologia de câmaras.")

db_path = DB_DIR / "sentinel.db"

if not db_path.exists():
    st.warning("Nenhuma base de dados encontrada. Execute `python main.py` para varrer as câmaras.")
else:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, global_id, camera_id, local_track_id, timestamp, 
               bbox_x1, bbox_y1, bbox_x2, bbox_y2 
        FROM events 
        ORDER BY timestamp ASC
    """, conn)
    conn.close()

    if df.empty:
        st.info("A base de dados está vazia. Rastreie um vídeo primeiro.")
    else:
        df["dt"] = pd.to_datetime(df["timestamp"])

        # Cálculo de resumo por Câmara e por Alvo (Dwell Time)
        summary_rows = []
        for (gid, cam), group in df.groupby(["global_id", "camera_id"]):
            t_in = group["dt"].min()
            t_out = group["dt"].max()
            dwell_sec = (t_out - t_in).total_seconds()
            summary_rows.append({
                "Alvo (ID Global)": gid,
                "Câmara": cam,
                "Entrada (UTC)": t_in.strftime("%H:%M:%S"),
                "Saída (UTC)": t_out.strftime("%H:%M:%S"),
                "Tempo na Câmara (s)": round(dwell_sec, 1),
                "Status LP": "🚨 ALERTA: Permanência Excessiva" if dwell_sec >= 8.0 else "✅ Passagem Normal"
            })
        summary_df = pd.DataFrame(summary_rows)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registos Forenses", len(df))
        col2.metric("Indivíduos Rastreados", df["global_id"].nunique())
        col3.metric("Câmaras Mapeadas", df["camera_id"].nunique())
        alerts_count = sum(1 for r in summary_rows if "ALERTA" in r["Status LP"])
        col4.metric("Alertas de Anomalia (LP)", alerts_count)

        st.divider()

        # Seletor de Investigação de Alvo
        st.subheader("🎯 Investigação de Associado / Alvo")
        selected_id = st.selectbox("Selecione o ID Global do Alvo Investigado:", sorted(df["global_id"].unique()))

        target_summary = summary_df[summary_df["Alvo (ID Global)"] == selected_id]
        st.write(f"### ⏱️ Análise de Permanência (*Dwell Time*) — Alvo #{selected_id}")
        st.dataframe(target_summary, use_container_width=True)

        # GERAÇÃO AUTOMÁTICA DO MAPA DO SITE (TOPOLOGIA DESCOBERTA)
        st.subheader("🗺️ Mapa Topológico Descoberto Automaticamente (Rota do Alvo)")
        st.caption("O Sentinel-Trace reconstrói a planta lógica conectando a sequência de câmaras onde o alvo foi avistado.")

        person_events = df[df["global_id"] == selected_id].sort_values("dt")
        cam_sequence = []
        for cam in person_events["camera_id"]:
            if not cam_sequence or cam_sequence[-1] != cam:
                cam_sequence.append(cam)

        if len(cam_sequence) >= 1:
            dot_graph = "digraph G {\n  rankdir=LR;\n  node [shape=box, style=\"filled,rounded\", fillcolor=\"#1E293B\", fontcolor=\"white\", fontname=\"Arial\"];\n  edge [color=\"#10B981\", penwidth=2.5, fontname=\"Arial\", fontcolor=\"#94A3B8\"];\n"
            for i in range(len(cam_sequence)):
                cam_name = cam_sequence[i]
                # Destaca em vermelho no mapa a câmara onde houve anomalia de tempo
                is_alert = any((r["Câmara"] == cam_name and "ALERTA" in r["Status LP"]) for r in summary_rows if r["Alvo (ID Global)"] == selected_id)
                fill = "#DC2626" if is_alert else "#0F172A"
                dot_graph += f'  "{cam_name}" [fillcolor="{fill}", label="{cam_name}\\n({"ALERTA LP" if is_alert else "OK"})"];\n'
                if i < len(cam_sequence) - 1:
                    next_cam = cam_sequence[i + 1]
                    dot_graph += f'  "{cam_name}" -> "{next_cam}" [label=" Deslocamento #{i+1} "];\n'
            dot_graph += "}"
            st.graphviz_chart(dot_graph)

        # Exportação do Laudo
        person_raw_df = df[df["global_id"] == selected_id].drop(columns=["dt"])
        docx_buffer = generate_forensic_report(selected_id, person_raw_df)
        st.download_button(
            label=f"📥 Baixar Dossiê Pericial Completo (Alvo #{selected_id}) em Word (.docx)",
            data=docx_buffer,
            file_name=f"Dossie_LP_Sentinel_Alvo_{selected_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )