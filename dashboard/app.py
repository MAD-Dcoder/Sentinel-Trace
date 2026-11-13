"""
Sentinel-Trace — Interface de investigação.

    python main.py            (abre no navegador: http://localhost:8501)

Fluxo: Nova investigação -> Localizar pessoas na câmera de início -> Escolher o alvo
-> Caçar em rodadas (confirmando cada aparição) -> Laudo.
Nada é processado até o investigador clicar.
"""

import json
import re
import sys
from datetime import datetime, date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from config.settings import CASES_DIR, ANCHOR_WINDOW_S, HUNT_BACK_S, HUNT_FORWARD_S
from core.attributes import COLORS
from pipeline.hunter import Engine, Case, Hunt
from utils.inspector import export_search_report, route_rows
from utils.report_generator import generate_forensic_report

st.set_page_config(page_title="Sentinel-Trace", page_icon="🛡️", layout="wide")
S = st.session_state


@st.cache_resource(show_spinner="Carregando modelos (uma vez só)...")
def get_engine() -> Engine:
    return Engine()


engine = get_engine()
cams = engine.archive.cameras


def cam_label(cid: str) -> str:
    c = cams.get(cid)
    return f"{cid} — {c.label}" if c and c.label else cid


def hms(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def progress_bar(text: str):
    bar = st.progress(0.0, text=text)
    return lambda f, msg: bar.progress(min(1.0, f), text=f"{text} — {msg}")


def color_badges(attrs: dict, wanted: dict) -> str:
    out = []
    for k, nome in (("camisa", "camisa"), ("calca", "calça")):
        v = attrs.get(k) or "?"
        ok = wanted.get(k) and wanted[k] == v
        out.append(f"{'✅ ' if ok else ''}{nome} **{v}**")
    return " · ".join(out)


def go(stage: str):
    S.stage = stage
    st.rerun()


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("🛡️ Sentinel-Trace")
    st.caption("Busca forense sob demanda — só processa quando você pede.")
    if st.button("➕ Nova investigação", width="stretch"):
        for k in ("case", "hunt", "people", "cands", "wanted"):
            S.pop(k, None)
        go("novo")

    casos = sorted((p.name for p in CASES_DIR.glob("*") if (p / "sentinel.db").exists()), reverse=True) \
        if CASES_DIR.exists() else []
    if casos:
        escolha = st.selectbox("Abrir caso existente", ["—"] + casos)
        if escolha != "—" and st.button("Abrir", width="stretch"):
            case = Case(engine, escolha)
            sid = case.db.latest_search_id()
            S.case, S.wanted = case, case.db.info().get("cores", {})
            if sid:
                s, _ = case.db.search_rows(sid)
                S.hunt = Hunt(case, (s[2], s[3]), S.wanted, search_id=sid)
                S.cands = S.hunt.candidates()
                go("cacada")
            go("novo")

    st.divider()
    edges = engine.knowledge.edges()
    st.metric("Passagens aprendidas (mapa)", len(edges))
    st.metric("Decisões que calibram a IA", len(engine.knowledge.labels()))
    if edges:
        dot = ['digraph G { rankdir=LR; node [shape=box, style="filled,rounded", fillcolor="#1E293B", fontcolor="white"];']
        dot += [f'"{a}" -> "{b}" [label="{n}x ~{dt:.0f}s"];' for a, b, n, dt in edges]
        st.graphviz_chart("\n".join(dot + ["}"]))
    st.divider()
    with st.expander(f"📼 Acervo: {len(cams)} câmeras", expanded=bool(engine.archive.problems)):
        st.caption("Exporte do Smart Client para a pasta data/ (subpastas valem). Nome e horário são lidos sozinhos.")
        for c in sorted(cams.values(), key=lambda c: c.camera_id):
            icon = "🕒" if c.time_source.startswith("relógio") else "✍️"
            st.markdown(f"**{c.label or c.camera_id}**  \n{icon} {c.start_time:%d/%m %H:%M:%S} → "
                        f"{c.end_time:%H:%M:%S} · {c.width}×{c.height}  \n<small>{c.time_source}</small>",
                        unsafe_allow_html=True)
        for p in engine.archive.problems:
            st.warning(p)
        if st.button("🔄 Recarregar acervo", width="stretch"):
            from core.video import VideoArchive
            with st.spinner("Lendo vídeos novos..."):
                engine.archive = VideoArchive()
            st.rerun()
    st.caption(f"Detector: {engine.detector.backend}")

S.setdefault("stage", "novo")

# ---------------------------------------------------------------- 1. nova investigação
if S.stage == "novo":
    st.header("1 · Nova investigação")
    if not cams:
        st.error("Nenhum vídeo em data/. Coloque as exportações do XProtect lá e configure config/cameras.yaml.")
        st.stop()
    first = min(c.start_time for c in cams.values())
    last = max(c.end_time for c in cams.values())
    st.caption(f"Acervo disponível: {first:%d/%m/%Y %H:%M:%S} até {last:%H:%M:%S}")

    with st.form("novo"):
        c1, c2 = st.columns(2)
        nome = c1.text_input("Nome do caso", value=f"caso_{datetime.now():%Y%m%d_%H%M}")
        investigador = c2.text_input("Investigador")
        associado = st.text_input("Associado (nome / matrícula / crachá) — opcional")
        st.markdown("**Onde começar** (ex.: câmera da catraca onde ele bateu o crachá)")
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        cam = c1.selectbox("Câmera de início", list(cams), format_func=cam_label)
        dia = c2.date_input("Data", value=first.date(), format="DD/MM/YYYY")
        hora = c3.text_input("Horário (HH:MM:SS)", value=f"{first:%H:%M:%S}")
        janela = c4.number_input("Janela ± segundos", 5, 600, ANCHOR_WINDOW_S, step=5)
        st.markdown("**Como ele estava** (ajuda a ordenar os candidatos — não é obrigatório)")
        c1, c2, c3 = st.columns(3)
        camisa = c1.selectbox("Cor da camisa/colete", ["não sei"] + COLORS)
        calca = c2.selectbox("Cor da calça", ["não sei"] + COLORS)
        bone = c3.selectbox("Boné", ["não sei", "sim", "não"])
        obs = st.text_area("Observações (óculos, mochila, tatuagem, manga longa...)", height=70)
        ok = st.form_submit_button("🔎 Localizar pessoas nesse momento", type="primary")

    if ok:
        if not re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", hora.strip()):
            st.error("Horário inválido. Use HH:MM:SS.")
            st.stop()
        hh = hora.strip() + (":00" if hora.strip().count(":") == 1 else "")
        when = datetime.combine(dia, datetime.strptime(hh, "%H:%M:%S").time())
        wanted = {k: v for k, v in (("camisa", camisa), ("calca", calca)) if v != "não sei"}
        case = Case(engine, re.sub(r"[^\w\-]", "_", nome))
        case.db.set_info({"investigador": investigador, "associado": associado, "cores": wanted,
                          "descricao": ", ".join(f"{k} {v}" for k, v in wanted.items()) + (f", boné {bone}" if bone != "não sei" else ""),
                          "observacoes": obs, "inicio": {"camera": cam, "horario": when.isoformat(), "janela": janela}})
        S.case, S.wanted = case, wanted
        S.people = case.locate(cam, when, janela, wanted, progress_bar("Varrendo a câmera de início"))
        st.toast(f"Varredura: {case.stats['frames_lidos']} frames lidos, IA usada em {case.stats['frames_com_ia']}, "
                 f"{case.stats['segundos']:.0f}s")
        go("escolher")

# ---------------------------------------------------------------- 2. escolher o alvo
elif S.stage == "escolher":
    case: Case = S.case
    st.header("2 · Quem é o alvo?")
    people = S.get("people", [])
    if not people:
        st.warning("Ninguém encontrado nessa janela. Volte e aumente a janela ou confira o horário.")
    else:
        st.caption("Ordenado: primeiro quem bate com a descrição, depois quem estava mais perto do horário informado.")
    for p in people:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.image(str(case.dir / p["crops_path"]))
            c2.markdown(f"**{hms(p['t_start'])} – {hms(p['t_end'])}**  \n{p['camera_id']} · rastro #{p['local_id']}  \n"
                        + color_badges(p["attrs"], S.wanted))
            if c2.button("🎯 É este", key=f"alvo_{p['local_id']}", type="primary"):
                S.hunt = Hunt(case, (p["camera_id"], p["local_id"]), S.wanted)
                S.cands = []
                go("cacada")
    if st.button("⬅️ Voltar"):
        go("novo")

# ---------------------------------------------------------------- 3. caçada
elif S.stage == "cacada":
    case, hunt = S.case, S.hunt
    st.header("3 · Caçada")
    st.markdown("#### Rota confirmada até agora")
    for t in hunt.route:
        c1, c2 = st.columns([1, 3])
        c1.markdown(f"**{hms(t['t_start'])}–{hms(t['t_end'])}**  \n{t['camera_id']} #{t['local_id']}")
        c2.image(str(case.dir / t["crops_path"]), width=420)

    st.divider()
    st.markdown("#### Próxima rodada")
    c1, c2, c3 = st.columns([4, 1, 1])
    sel = c1.multiselect("Câmeras a varrer (sugeridas pelo mapa aprendido)", list(cams),
                         default=hunt.suggest_cameras(), format_func=cam_label)
    back = c2.number_input("Segundos antes", 0, 600, HUNT_BACK_S, step=5)
    fwd = c3.number_input("Segundos depois", 10, 3600, HUNT_FORWARD_S, step=10)
    last = hunt.route[-1]
    st.caption(f"Janela: {hms(last['t_start'] - back)} até {hms(last['t_end'] + fwd)} "
               f"(a partir do último avistamento). Trechos já varridos neste caso não são processados de novo.")
    b1, b2 = st.columns(2)
    if b1.button("🐾 Caçar", type="primary", disabled=not sel):
        S.cands = hunt.run_round(sel, back, fwd, progress_bar("Caçando"))
        st.rerun()
    if b2.button("🏁 Encerrar e gerar laudo"):
        hunt.close()
        go("laudo")

    cands = S.get("cands", [])
    if cands:
        st.markdown("#### Candidatos (do mais parecido para o menos)")
    for c in cands:
        t = c["tracklet"]
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            c1.image(str(case.dir / t["crops_path"]))
            m = "-" if c["margin"] is None else f"{c['margin']:+.2f}"
            c2.markdown(f"**{hms(t['t_start'])} – {hms(t['t_end'])}**  \n{t['camera_id']} · #{t['local_id']}  \n"
                        f"Semelhança **{c['prob']:.0%}** · margem {m}  \n" + color_badges(t["attrs"], S.wanted))
            y, n = c2.columns(2)
            if y.button("✅ É ele", key=f"y_{c['key']}"):
                hunt.decide(c, True)
                S.cands = hunt.candidates()
                st.rerun()
            if n.button("❌ Não", key=f"n_{c['key']}"):
                hunt.decide(c, False)
                S.cands = hunt.candidates()
                st.rerun()
    if not cands:
        st.info("Escolha as câmeras e clique em **Caçar**. Na primeira rodada, sem mapa aprendido, "
                "o Sentinel sugere todas as câmeras do acervo.")

# ---------------------------------------------------------------- 4. laudo
elif S.stage == "laudo":
    case, hunt = S.case, S.hunt
    st.header("4 · Laudo")
    text = export_search_report(case.dir, hunt.search_id)
    s, rows, info = route_rows(case.dir, hunt.search_id)
    route = sorted((r for r in rows if r["decisao"] in ("ancora", "aceito")), key=lambda r: r["t_start"])
    board = case.dir / f"cacada_{hunt.search_id}_evidencias.jpg"
    if board.exists():
        st.image(str(board))
    st.code(text, language=None)
    st.download_button("📥 Baixar laudo (.docx)",
                       generate_forensic_report(hunt.search_id, f"{s[2]} #{s[3]}", route, json.loads(s[5]), board),
                       file_name=f"Laudo_Sentinel_{case.name}.docx", type="primary")
    if st.button("⬅️ Voltar para a caçada"):
        go("cacada")
