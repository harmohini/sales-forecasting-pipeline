"""
Centralized Configuration Settings using Environment Variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "sales_forecasting")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Model Artifacts & Retraining Directories
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/"))
MODELS_DIR = BASE_DIR / MODEL_PATH
ARCHIVE_MODELS_DIR = MODELS_DIR / "archive"
RETRAINING_HISTORY_PATH = MODELS_DIR / "retraining_history.json"
MIN_IMPROVEMENT_PERCENT = float(os.getenv("MIN_IMPROVEMENT_PERCENT", 1.0))


# Data Directories
DATA_DIR = BASE_DIR / "data"
INCOMING_DATA_DIR = DATA_DIR / "incoming"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARCHIVE_DATA_DIR = DATA_DIR / "archive"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_DIR = BASE_DIR / "logs"
