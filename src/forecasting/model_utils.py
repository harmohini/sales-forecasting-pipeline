"""
Model Utilities Component.

Responsible for model serialization, deserialization, and metadata persistence.
"""

import json
from pathlib import Path
import joblib
from config.settings import BASE_DIR, MODEL_PATH

MODELS_DIR = BASE_DIR / MODEL_PATH

def save_model_artifact(artifact: dict, filepath: Path = None) -> Path:
    """
    Serialize and save complete model pipeline artifact to disk using joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if filepath is None:
        filepath = MODELS_DIR / "sales_forecasting_model.joblib"
    
    filepath = Path(filepath)
    joblib.dump(artifact, filepath)
    return filepath

def load_model_artifact(filepath: Path = None) -> dict:
    """
    Load and deserialize model pipeline artifact from disk using joblib.
    """
    if filepath is None:
        filepath = MODELS_DIR / "sales_forecasting_model.joblib"
        
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")
        
    return joblib.load(filepath)

def save_model_metadata(metadata: dict, filepath: Path = None) -> Path:
    """
    Save model training metadata as JSON file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if filepath is None:
        filepath = MODELS_DIR / "model_metadata.json"
        
    filepath = Path(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    return filepath

def load_model_metadata(filepath: Path = None) -> dict:
    """
    Load model training metadata from JSON file.
    """
    if filepath is None:
        filepath = MODELS_DIR / "model_metadata.json"
        
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model metadata not found at: {filepath}")
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
