<div align="center">

# 🛡️ Sentinel-Trace
### Autonomous Cross-Camera Tracking & Multi-Person Loss Prevention

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<p align="center">
  Sistema modular de visão computacional concebido para monitorização de circuitos fechados de TV (CCTV), prevenção de perdas logísticas e rastreio contínuo de indivíduos entre câmaras sem recurso a reconhecimento facial.
</p>

</div>

---

## 📌 Visão Geral

O **Sentinel-Trace** resolve a quebra de rastreio (*track loss*) comum em sistemas tradicionais quando indivíduos transitam entre pontos cegos ou trocam de câmara em complexos logísticos e industriais. 

Ao combinar **YOLOv8** para deteção em tempo real, **ByteTrack** para tracking intra-câmara e **OSNet (Deep ReID)** para extração de vetores biométricos corporais, a plataforma consolida uma timeline unificada de eventos com auditoria forense.

### Principais Funcionalidades
- **Deteção e Rastreio Robusto:** Deteção com YOLOv8 e manutenção de trajetória com ByteTrack.
- **Cross-Camera Re-Identification:** Extração de *embeddings* visuais de 512 dimensões invariantes a ângulo e iluminação.
- **Base de Dados Forense:** Armazenamento vetorial indexado com SQLite/PostgreSQL para buscas retroativas por afinidade de cosseno.
- **Auditoria e Exportação:** Geração automática de relatórios em formato DOCX com capturas das evidências e carimbos de data/hora.
- **Dashboard Operacional:** Interface interativa construída em Streamlit para análise de incidentes e visualização de trajetórias.

---

## 🏗️ Arquitetura do Pipeline

```text
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │ Câmaras CCTV    │──────>│ Deteção YOLOv8  │──────>│ ByteTrack       │
 │ Feeds de Vídeo  │       │ & Bounding Box  │       │ IDs Locais      │
 └─────────────────┘       └─────────────────┘       └────────┬────────┘
                                                              │
                                                              ▼
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │ Dashboard       │<──────│ Base Vetorial   │<──────│ OSNet (ReID)    │
 │ Streamlit / Rel │       │ Cosine Distance │       │ 512-d Vectors   │
 └─────────────────┘       └─────────────────┘       └─────────────────┘
## 🚀 Como usar

```bash
python main.py            # abre a interface em http://localhost:8501
python main.py --rede     # permite acesso de outros PCs da rede
```

1. **Nova investigação** — câmera de início (ex.: catraca), data/horário, janela ± segundos e, se souber, cor da camisa/calça.
2. **Quem é o alvo?** — o Sentinel varre só aquela câmera naquela janela e lista as pessoas (quem bate com a descrição primeiro).
3. **Caçada** — a cada rodada ele varre só as câmeras sugeridas, logo após o último avistamento; você confirma ✅ ou rejeita ❌.
4. **Laudo** — rota cronológica, prancha de evidências e .docx.

Nada roda sem você pedir. Trechos já varridos no caso não são reprocessados. Cada confirmação ensina o Sentinel:
calibra a % de semelhança e registra a passagem câmera A → B (o mapa do galpão se monta sozinho, em `db/conhecimento.db`).

### Instalação (uma vez)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# ReID (OSNet-AIN x1.0 / MSMT17, torchreid model zoo, MIT) — ~17 MB
curl -L -o models/osnet_ain_x1_0_msmt17.pth "https://drive.usercontent.google.com/download?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal&export=download&confirm=t"
# Detector otimizado para CPU Intel (OpenVINO)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='openvino', imgsz=(736,1280))"
move yolov8s_openvino_model models\yolov8s_1280_openvino_model
```

Coloque as exportações do XProtect em `data/` e o horário real do frame 0 de cada uma em `config/cameras.yaml`.
`casos/` guarda recortes de pessoas (**dado biométrico — nunca versionar nem compartilhar**).
