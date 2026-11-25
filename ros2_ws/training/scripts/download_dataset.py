from huggingface_hub import snapshot_download
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"

DATASET_DIR.mkdir(parents=True, exist_ok=True)

inicio = time.perf_counter()

print("Iniciando descarga del dataset...")
print("Destino:", DATASET_DIR)

path = snapshot_download(
    repo_id="JdeRobot/e2e-visual-control-combined-dataset",
    repo_type="dataset",
    local_dir=str(DATASET_DIR),
    allow_patterns=["*.png", "label.csv"],
    max_workers=8
)

fin = time.perf_counter()

print("\nDescarga completada")
print("Ubicacion:", path)
print("Tiempo:", round((fin - inicio) / 60, 2), "minutos")
