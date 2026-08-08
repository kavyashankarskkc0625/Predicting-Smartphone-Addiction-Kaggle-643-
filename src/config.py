"""
Project Configuration

Centralized paths and constants for the
Smartphone Addiction Kaggle competition.
"""

from pathlib import Path


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================================
# DATA DIRECTORIES
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"


# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
FEATURE_IMPORTANCE_DIR = OUTPUTS_DIR / "feature_importance"


# ==========================================================
# RAW DATA FILES
# ==========================================================

TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"


# ==========================================================
# PROJECT CONSTANTS
# ==========================================================

TARGET_COLUMN = "addicted_label"
ID_COLUMN = "id"

RANDOM_STATE = 42