import os
from pathlib import Path

# ---------- High-level switches ----------
DEBUG   = False
SAVE_LOG = True

# ---------- Paths ----------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Supplement_Sales_Weekly.csv"
TARGET_COL = "Units Sold"

# ---------- LLM ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-...")
LLM_MODEL      = "gpt-4.1-mini-2025-04-14"

# ---------- Modelling ----------
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
BASELINE_REGRESSOR = "RandomForestRegressor"

# ---------- Global variable names ----------
RAW_DF_VAR   = "shared_df"
CLEAN_DF_VAR = "clean_df"
X_TRAIN_VAR  = "X_train"
X_TEST_VAR   = "X_test"
Y_TRAIN_VAR  = "y_train"
Y_TEST_VAR   = "y_test"

# ---------- Helper ----------
def get(key, default=None):
    return globals().get(key, default)