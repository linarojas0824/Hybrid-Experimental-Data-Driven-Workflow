from pathlib import Path
# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "databases"
DB_FILE = DATABASE_DIR / "Ternary_round1.db"
RAW_DIR = DATA_DIR / "raw"
# Results
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLING_DIR = RESULTS_DIR / "Sampling"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = RESULTS_DIR / "logs"
