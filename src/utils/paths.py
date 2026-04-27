from pathlib import Path
# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# main paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
DATABASE_DIR = DATA_DIR / "databases"
DESCRIPTORS_DIR = DATA_DIR / "descriptors_data"
RAW_DIR = DATA_DIR / "raw"

# Results paths
PLOTS_DIR = RESULTS_DIR/ "Plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SAMPLING_DIR = RESULTS_DIR/ "Sampling"
TABLES_DIR = RESULTS_DIR/ "Tables"
EXP_RESULT = RESULTS_DIR/"results"

 
# src paths
MODELS_DIR = SRC_DIR/ "models"
MODELS_SCR_DIR = SRC_DIR / "models_scripts"

##================== R paths ======================
R_PROJECT_ROOT = "/Users/linarojas/Desktop/Research/Papers/Combinatorial_Ternary/Experimental design"

