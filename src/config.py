from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "ASDS6304_Group10_Dataset.csv"
VENDOR_MAPPING_PATH = BASE_DIR / "data" / "vendor_mapping.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
MODELS_DIR = BASE_DIR / "outputs" / "models"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

TARGET_REGRESSION = "how_long_delayed"
TARGET_CLASSIFICATION = "delay_category"
RANDOM_STATE = 1025
TEST_SIZE = 0.20

DELAY_BINS = [-1, 10, 20, 40, 60, float("inf")]
DELAY_LABELS = [
    "On Time / Slight",
    "Minor Delay",
    "Moderate Delay",
    "Major Delay",
    "Extreme Delay",
]
