"""
Acervo de vídeo — Sentinel-Trace.

Abstrai "de onde vem o vídeo": hoje são exportações do XProtect na pasta data/
(com o horário real do frame 0 em config/cameras.yaml); na integração futura, o
mesmo método read_window() buscará o trecho direto no XProtect via SDK.

O ponto central de economia: o Sentinel NUNCA lê um vídeo inteiro. Ele lê só a
janela de tempo pedida, de uma câmera, na taxa de amostragem pedida.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import yaml

from config.settings import DATA_DIR, CAMERAS_CONFIG_PATH

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}


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

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.n_frames / self.fps)


class VideoArchive:
    def __init__(self, data_dir: Path = DATA_DIR, config_path: Path = CAMERAS_CONFIG_PATH):
        cfg = {}
        if config_path.exists():
            cfg = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("cameras") or {}
        by_file = {c["file"]: (cid, c) for cid, c in cfg.items() if c and "file" in c}
        default_start = next((datetime.fromisoformat(c["start_time"]) for c in cfg.values()
                              if c and c.get("start_time")), datetime(2000, 1, 1))

        self.cameras: dict[str, Camera] = {}
        for f in sorted(data_dir.iterdir()) if data_dir.exists() else []:
            if f.suffix.lower() not in VIDEO_EXT:
                continue
            cid, c = by_file.get(f.name, (f.name.split(".")[0], {}))
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                continue
            self.cameras[cid] = Camera(
                camera_id=cid, path=f,
                start_time=datetime.fromisoformat(c["start_time"]) if c.get("start_time") else default_start,
                fps=cap.get(cv2.CAP_PROP_FPS) or 15.0, n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                label=c.get("label", ""),
            )
            cap.release()

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

    def frame_at(self, camera_id: str, when: datetime):
        cam = self.cameras[camera_id]
        cap = cv2.VideoCapture(str(cam.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((when - cam.start_time).total_seconds() * cam.fps))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
