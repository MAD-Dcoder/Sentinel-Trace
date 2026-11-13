# AGENTS.md — Guia de Desenvolvimento e Operação do Sentinel-Trace

Este documento orienta agentes de IA e desenvolvedores sobre a arquitetura, estrutura de código, comandos de execução e regras operacionais do **Sentinel-Trace**.

---

## 1. Visão Geral da Arquitetura e Stack

O **Sentinel-Trace** é um motor forense sob demanda para prevenção de perdas (Loss Prevention) em centros de distribuição e galpões. Em vez de monitoramento contínuo, realiza busca guiada por humanos ("alvo primeiro") de pessoas entre múltiplas câmeras.

### Stack Tecnológica
- **Detecção de Pessoas:** [YOLOv8s](https://docs.ultralytics.com/) acelerado via **Intel OpenVINO** (`models/yolov8s_1280_openvino_model/`) com fallback em PyTorch (`yolov8s.pt`).
- **Rastreamento Intra-câmera:** **ByteTrack** via Ultralytics com perfil calibrado para corredores, carrinhos e oclusões (`config/warehouse_tracker.yaml`).
- **Re-Identificação (ReID Inter-câmeras):** **OSNet-AIN** (treinado em MSMT17), gerando embeddings de 512 dimensões por recorte corporal.
- **Interface e Dashboard:** **Streamlit** (`dashboard/app.py`), fornecendo fluxo interativo: seleção do caso → definição do alvo → caçada em rodadas → revisão humana → emissão de laudo.
- **Integração Milestone XProtect:** Processamento de vídeos exportados do Smart Client (MP4/AVI) colocados em `data/`. Sincronização temporal precisa realizada pela leitura automatizada do carimbo/relógio OSD gravado no próprio vídeo (`core/osd_clock.py`).
- **Base de Dados:** **SQLite** (um banco isolado por investigação em `casos/<caso>/sentinel.db` e banco compartilhado de conhecimento/topologia em `db/conhecimento.db`).
- **Calibração e Laudos:** `scikit-learn` (calibração de probabilidade de matching) e `python-docx` (geração de laudos periciais formatados).

---

## 2. Mapa dos Principais Arquivos e Pastas

```text
Sentinel-Trace/
├── main.py                     # Entry point para inicializar o Streamlit
├── requirements.txt            # Dependências Python (CPU-first)
├── config/
│   ├── settings.py             # Parâmetros globais (resoluções, thresholds, buffers, caminhos)
│   ├── warehouse_tracker.yaml  # Configurações do ByteTrack para cenários de galpão
│   ├── cameras.yaml            # Configuração e mapeamento opcional de câmeras/offsets
│   └── osd_font.npz            # Moldes de dígitos aprendidos para leitura do relógio OSD
├── core/
│   ├── detector.py             # YOLOv8s + integração ByteTrack por câmera
│   ├── tracklet.py             # Extração, filtragem qualitativa e seleção top-K de crops
│   ├── reid.py                 # Extrator de características OSNet-AIN
│   ├── matcher.py              # Cálculo de similaridade por cosseno e calibração de scores
│   ├── motion.py               # Detector de movimento leve (miniaturas anti-tremor)
│   ├── osd_clock.py            # OCR leve baseado em templates para ler timestamp do Milestone
│   ├── video.py                # Leitura sob demanda e gerenciamento do acervo de vídeo
│   ├── attributes.py           # Análise complementar de atributos visuais (ex: cor de roupas)
│   └── models/osnet_ain.py     # Definição da rede neural OSNet-AIN (torchreid)
├── pipeline/
│   └── hunter.py               # Orquestrador das rodadas de busca (Engine, Case e Hunt)
├── dashboard/
│   └── app.py                  # Interface web Streamlit
├── db/
│   ├── database.py             # Operações SQLite (caso específico e conhecimento global)
│   └── conhecimento.db         # Grafo aprendido de transições entre câmeras
├── utils/
│   ├── inspector.py            # Pranchas visuais e inspeção de evidências
│   ├── report_generator.py     # Exportador de relatório pericial em .docx
│   └── osd_learn.py            # Utilitário para aprender nova fonte de relógio OSD
├── data/                       # Diretório de vídeos exportados do VMS (não versionado)
├── casos/                      # Diretório de casos com crops e bancos locais (não versionado)
└── models/                     # Pesos dos modelos (.pth, .bin, .xml, .pt)
```

---

## 3. Comandos para Rodar o Projeto no Ambiente venv

O projeto utiliza um ambiente virtual Python 3.12 localizado em `venv/`. **Sempre utilize os executáveis do `venv` diretamente no Windows PowerShell:**

### Ativar o Ambiente (opcional se chamar executável direto)
```powershell
.\venv\Scripts\Activate.ps1
```

### Rodar a Aplicação
```powershell
# Execução local padrão (http://localhost:8501)
.\venv\Scripts\python.exe main.py

# Disponibilizar na rede local (porta 8501)
.\venv\Scripts\python.exe main.py --rede

# Alternativa direta via Streamlit
.\venv\Scripts\streamlit.exe run dashboard\app.py
```

### Instalar ou Atualizar Dependências
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Exportar / Recriar Modelo OpenVINO (CPU)
```powershell
.\venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='openvino', imgsz=(736,1280))"
```

---

## 4. Regras Obrigatórias para o Agente

1. **Sempre usar o Python do venv:**
   - Execute comandos de terminal apontando explicitamente para `.\venv\Scripts\python.exe`, `.\venv\Scripts\pip.exe` ou `.\venv\Scripts\streamlit.exe`.
   - Nunca use comandos globais `python` ou `pip` que possam referenciar outro ambiente fora do repositório.

2. **Nunca ler arquivos binários de mídia ou modelos:**
   - Não tente ler arquivos de vídeo (`.mp4`, `.avi`, `.mkv`), pesos de rede (`.pt`, `.pth`, `.bin`), arquivos comprimidos (`.npz`) ou `.json` gigantescos via ferramentas de leitura de arquivo textual (`read_file`).
   - Para inspecionar propriedades desses arquivos, utilize comandos como `Get-Item`, `Get-ChildItem` ou scripts Python dedicados via terminal.

3. **Manter respostas e código limpos e diretos:**
   - Evite redundâncias e explicações excessivamente prolixas.
   - Escreva código modular, bem tipado e seguindo os padrões já adotados no projeto.

4. **Pedir confirmação antes de alterar arquivos de configuração:**
   - **SEMPRE** consulte o usuário antes de modificar ou sobrescrever:
     - `config/warehouse_tracker.yaml` (parâmetros do ByteTrack)
     - `config/cameras.yaml` (mapeamentos de câmeras)
     - `config/settings.py` (thresholds globais de detecção e buffers)
     - `config/osd_font.npz` (templates de OCR)
   - Mudanças nesses arquivos alteram drasticamente a acurácia forense do pipeline.
