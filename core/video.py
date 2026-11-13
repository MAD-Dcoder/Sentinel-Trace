"""
Acervo de vídeo — Sentinel-Trace.

Abstrai "de onde vem o vídeo": hoje são exportações do Smart Client numa pasta (data/,
inclusive subpastas); na integração futura, o mesmo read_window() buscará o trecho
direto no XProtect.

Montagem automática do acervo — o investigador só exporta e joga na pasta:
  - nome da câmera: vem do nome do arquivo exportado (datas/horas do nome são removidas);
  - horário real do frame 0: lido do relógio gravado na imagem (core/osd_clock.py);
  - resultado fica em cache (data/.sentinel_acervo.json) — cada vídeo é lido uma vez só;
  - config/cameras.yaml continua opcional, para dar apelido à câmera ou corrigir o horário.

O ponto central de economia: o Sentinel NUNCA lê um vídeo inteiro. Ele lê só a
janela de tempo pedida, de uma câmera, na taxa de amostragem pedida.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

from config.settings import DATA_DIR, CAMERAS_CONFIG_PATH, ARCHIVE_INDEX

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

# Datas/horas que o Smart Client (ou o investigador) costuma pôr no nome do arquivo exportado
_NAME_NOISE = [
    r"\d{4}[-_.]\d{2}[-_.]\d{2}", r"\d{2}[-_.]\d{2}[-_.]\d{4}", r"\d{8}",          # datas
    r"\d{2}[-_.h]\d{2}[-_.m]\d{2}s?", r"\d{6}",                                       # horas
    r"\(\d+\)",                                                                        # (1), (2)...
]


def camera_name_from_file(path: Path) -> tuple[str, str]:
    """('cam_01_corredor_baixo', 'cam 01 corredor baixo') a partir do nome do arquivo."""
    stem = path.name
    while Path(stem).suffix.lower() in VIDEO_EXT:          # "x.mp4.MP4" -> "x"
        stem = Path(stem).stem
    label = stem
    for pat in _NAME_NOISE:
        label = re.sub(pat, " ", label)
    label = re.sub(r"[\s_\-.]+", " ", label).strip() or stem
    cid = re.sub(r"[^\w]+", "_", label.lower()).strip("_")
    return cid, label


@dataclass
class Camera:
    camera_id: str
    path: Path
    start_time: datetime
    fps: float
    n_frames: int
    width: int
    height: int
    label: str = ""
    time_source: str = ""          # de onde veio o horário (relógio, yaml, padrão)

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.n_frames / self.fps)


class VideoArchive:
    def __init__(self, data_dir: Path = DATA_DIR, config_path: Path = CAMERAS_CONFIG_PATH, progress=None):
        cfg = {}
        if config_path.exists():
            cfg = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("cameras") or {}
        by_file = {c["file"]: (cid, c) for cid, c in cfg.items() if c and c.get("file")}
        index = _load_index()
        clock = None
        files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT) if data_dir.exists() else []

        self.cameras: dict[str, Camera] = {}
        self.problems: list[str] = []
        for i, f in enumerate(files):
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                self.problems.append(f"{f.name}: não abriu")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            n, w, h = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH,
                                                  cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            cid, label = camera_name_from_file(f)
            yaml_id, c = by_file.get(f.name, (None, {}))
            cid, label = yaml_id or cid, c.get("label") or label

            # Horário: yaml (correção manual) > cache > relógio da imagem
            key = f"{f.relative_to(data_dir)}|{f.stat().st_size}|{int(f.stat().st_mtime)}"
            if c.get("start_time"):
                start, source = datetime.fromisoformat(c["start_time"]), "manual (cameras.yaml)"
            elif key in index:
                start = datetime.fromisoformat(index[key]["start"]) if index[key]["start"] else None
                source = index[key]["source"]
            else:
                if progress:
                    progress(i / max(1, len(files)), f"lendo relógio de {f.name}")
                if clock is None:
                    from core.osd_clock import OSDClock
                    clock = OSDClock()
                start, source = clock.start_time(f)
                index[key] = {"start": start.isoformat() if start else None, "source": source}
            if start is None:
                self.problems.append(f"{f.name}: {source} — informe start_time em config/cameras.yaml")
                continue
            if cid in self.cameras:                     # mesma câmera exportada em 2 arquivos
                cid = f"{cid}_{i}"
            self.cameras[cid] = Camera(cid, f, start, fps, n, w, h, label, source)
        _save_index(index)

    def read_window(self, camera_id: str, t0: datetime, t1: datetime, sample_fps: float):
        """Gera (frame_idx, horário_real, frame) só dentro de [t0, t1], amostrado em sample_fps."""
        cam = self.cameras[camera_id]
        t0, t1 = max(t0, cam.start_time), min(t1, cam.end_time)
        if t1 <= t0:
            return
        f0 = int((t0 - cam.start_time).total_seconds() * cam.fps)
        f1 = int((t1 - cam.start_time).total_seconds() * cam.fps)
        step = max(1.0, cam.fps / sample_fps)

        cap = cv2.VideoCapture(str(cam.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            idx, next_take = f0, float(f0)
            while idx <= f1:
                if not cap.grab():          # grab = avança sem montar a imagem (barato)
                    break
                if idx >= next_take:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    yield idx, cam.start_time + timedelta(seconds=idx / cam.fps), frame
                    next_take += step
                idx += 1
        finally:
            cap.release()


def _load_index() -> dict:
    try:
        return json.loads(ARCHIVE_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_index(index: dict):
    try:
        ARCHIVE_INDEX.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
