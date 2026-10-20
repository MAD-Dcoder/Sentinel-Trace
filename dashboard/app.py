"""
Interface Forense e Operacional - Sentinel-Trace.
Construída com Streamlit para monitorização de incidentes e auditoria de trajetórias.
"""

import sys
from pathlib import Path

# Adiciona a raiz do repositório ao sys.path para permitir importações relativas
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import sqlite3
import pandas as pd
import streamlit as st
from config.settings import DB_DIR
from utils.report_generator import generate_forensic_report

st.set_page_config(
    page_title="Sentinel-Trace | CCTV Loss Prevention",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Sentinel-Trace — Painel de Auditoria e Prevenção de Perdas")
st.markdown("Monitorização contínua de indivíduos e rastreio cruzado entre câmaras.")

# Conexão à base de dados
db_path = DB_DIR / "sentinel.db"

if not db_path.exists():
    st.warning("Nenhuma base de dados encontrada. Execute uma simulação no pipeline para gerar registos.")
else:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT id, global_id, camera_id, local_track_id, timestamp, 
               bbox_x1, bbox_y1, bbox_x2, bbox_y2 
        FROM events 
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Deteções Gravadas", len(df))
    with col2:
        st.metric("Indivíduos Únicos Rastreados", df["global_id"].nunique() if not df.empty else 0)
    with col3:
        st.metric("Câmaras Ativas", df["camera_id"].nunique() if not df.empty else 0)

    st.divider()

    st.subheader("📋 Registo Cronológico de Eventos (Timeline Forense)")
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        st.subheader("🔍 Filtrar Trajetória por ID Global")
        selected_id = st.selectbox("Selecione o indivíduo:", sorted(df["global_id"].unique()))
        person_df = df[df["global_id"] == selected_id].sort_values("timestamp")
        
        st.write(f"Histórico de movimentação para o **ID Global {selected_id}**:")
        st.table(person_df[["camera_id", "timestamp", "local_track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]])

        # Gerador do laudo pericial para download
        docx_buffer = generate_forensic_report(selected_id, person_df)
        st.download_button(
            label=f"📥 Baixar Laudo Forense Pericial (ID #{selected_id}) em Word (.docx)",
            data=docx_buffer,
            file_name=f"Laudo_Sentinel_ID_{selected_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )