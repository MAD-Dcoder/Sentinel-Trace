"""
Sentinel-Trace — abre a interface de investigação no navegador.

    python main.py                 -> http://localhost:8501 (só neste computador)
    python main.py --rede          -> acessível por outros PCs da rede (http://<ip-deste-pc>:8501)

Nada é processado até o investigador iniciar uma investigação na interface.
"""

import subprocess
import sys
from pathlib import Path


def main():
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    addr = "0.0.0.0" if "--rede" in sys.argv else "localhost"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app),
           "--server.address", addr, "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    if addr == "localhost":
        print("🛡️  Sentinel-Trace: abra http://localhost:8501 no navegador (Ctrl+C para encerrar)")
    else:
        print("🛡️  Sentinel-Trace acessível na rede pela porta 8501 (Ctrl+C para encerrar)")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
