"""
Extrator de ReID — Sentinel-Trace.

OSNet-AIN x1.0 treinada no MSMT17 (pedestres, 4.101 identidades, 15 câmeras).
Diferente da ResNet18/ImageNet + histograma de cor anterior, ela foi treinada para
separar PESSOAS diferentes — e não objetos — mesmo com roupas parecidas (uniformes).

Pesos: torchreid model zoo (MIT). Ver config.settings.REID_WEIGHTS_URL.
"""

import cv2
import numpy as np
import torch

from config.settings import REID_WEIGHTS, REID_WEIGHTS_URL, REID_IMG_SIZE, REID_BATCH
from core.models.osnet_ain import osnet_ain_x1_0

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDExtractor:
    dim = 512

    def __init__(self, weights=REID_WEIGHTS, device: str | None = None):
        if not weights.exists():
            raise FileNotFoundError(
                f"Pesos de ReID não encontrados em {weights}.\n"
                f"Baixe com:\n  curl -L -o \"{weights}\" \"{REID_WEIGHTS_URL}\""
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.get("state_dict", state).items()}
        state = {k: v for k, v in state.items() if not k.startswith("classifier")}

        self.model = osnet_ain_x1_0(num_classes=1, pretrained=False, loss="softmax")
        missing = [k for k in self.model.load_state_dict(state, strict=False).missing_keys
                   if not k.startswith("classifier")]
        if missing:
            raise RuntimeError(f"Pesos de ReID incompatíveis, faltando: {missing[:5]}")
        self.model.to(self.device).eval()

    def _prep(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = REID_IMG_SIZE
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)

    @torch.no_grad()
    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Retorna (N, 512) L2-normalizado. Crops vazios viram vetor zero."""
        out = np.zeros((len(crops), self.dim), dtype=np.float32)
        valid = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
        for s in range(0, len(valid), REID_BATCH):
            idx = valid[s:s + REID_BATCH]
            batch = torch.from_numpy(np.stack([self._prep(crops[i]) for i in idx])).to(self.device)
            # Média com o flip horizontal: pose espelhada não deve mudar a identidade
            feats = self.model(batch) + self.model(torch.flip(batch, dims=[3]))
            feats = torch.nn.functional.normalize(feats, dim=1)
            out[idx] = feats.cpu().numpy()
        return out
