"""
Prediction & Inference Component.

Responsible for generating sales forecasts using trained model artifacts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.forecasting.model_utils import load_model_artifact

def generate_forecast(input_df: pd.DataFrame, artifact: dict = None, model_filepath: Path = None) -> np.ndarray:
    """
    Generate non-negative sales predictions using a trained forecasting model artifact.
    Validates required features, transforms categorical inputs using the fitted encoder,
    and returns numpy array of predictions.
    """
    if artifact is None:
        artifact = load_model_artifact(model_filepath)

    model = artifact["model"]
    encoder = artifact["encoder"]
    feature_names = artifact["feature_names"]
    categorical_cols = artifact.get("categorical_cols", [])

    # Validate required columns
    missing_cols = set(feature_names) - set(input_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns for forecast: {sorted(list(missing_cols))}. "
            f"Expected features: {feature_names}"
        )

    # Prepare features copy
    X = input_df[feature_cols].copy() if 'feature_cols' in locals() else input_df[feature_names].copy()

    # Transform categorical features using pre-fitted encoder
    if categorical_cols:
        X[categorical_cols] = encoder.transform(X[categorical_cols])

    # Predict non-negative sales
    raw_preds = model.predict(X)
    predictions = np.maximum(0.0, raw_preds)
    return predictions

def predict_from_saved_model(input_data: pd.DataFrame, model_filepath: Path = None) -> np.ndarray:
    """
    Load serialized model artifact and predict sales forecasts.
    """
    return generate_forecast(input_df=input_data, model_filepath=model_filepath)
