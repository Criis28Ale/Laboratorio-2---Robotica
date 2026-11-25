from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SPLITS_DIR = DATASET_DIR / "splits"

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 80
IMAGE_CHANNELS = 3

CROP_TOP = 200

V_MIN = 4.0
V_MAX = 7.0
W_SCALE = 2.5

SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 7

NUM_WORKERS = 4

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
