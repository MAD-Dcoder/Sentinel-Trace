"""
Cortador Automático de Clipes de Investigação LP - Sentinel-Trace.
Remove os tempos vazios dos vídeos originais e gera as câmaras de teste na pasta data/.
"""

import cv2
from pathlib import Path
from config.settings import DATA_DIR


def cut_clip(input_path: Path, output_name: str, start_sec: float, end_sec: float):
    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        return

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    output_path = DATA_DIR / output_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()
    print(f"✅ Gerado: {output_name} ({end_sec - start_sec:.1f}s)")


def main():
    v1 = DATA_DIR / "raw_1.mp4"  # Vídeo do corredor visto de cima
    v2 = DATA_DIR / "raw_2.mp4"  # Vídeo da garagem com a moto

    print("✂️ Cortando momentos exatos de aparição...")

    # Cria 4 câmaras simuladas apenas com os momentos de ação!
    if v1.exists():
        cut_clip(v1, "cam_01_corredor_roupao.mp4", 0.0, 11.5)
        cut_clip(v1, "cam_02_corredor_externo.mp4", 25.0, 37.0)

    if v2.exists():
        cut_clip(v2, "cam_03_garagem_passagem.mp4", 2.0, 11.5)
        cut_clip(v2, "cam_04_garagem_furto_camisa_branca.mp4", 16.0, 36.5)
        cut_clip(v2, "cam_05_garagem_furto_roupao.mp4", 48.0, 59.0)

    print("\n🎯 Pronto! Remova ou mova os arquivos 'raw_1.mp4' e 'raw_2.mp4' da pasta data/ e rode 'python main.py'!")


if __name__ == "__main__":
    main()