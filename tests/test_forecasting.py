"""
Unit tests for Stage 3 Forecasting Model Training, Evaluation, and Prediction logic.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.forecasting.evaluate import calculate_metrics, evaluate_model
from src.forecasting.model_utils import save_model_artifact, load_model_artifact, save_model_metadata, load_model_metadata
from src.forecasting.predict import generate_forecast, predict_from_saved_model
from src.forecasting.train import train_forecasting_pipeline


@pytest.fixture
def sample_metrics_data():
    y_true = np.array([100.0, 200.0, 0.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 10.0, 420.0, 480.0])
    return y_true, y_pred


@pytest.fixture
def synthetic_feature_df():
    return pd.DataFrame({
        "store": [1, 1, 1, 2, 2],
        "date": pd.date_range("2015-07-01", periods=5, freq="D"),
        "sales": [500, 520, 540, 600, 610],
        "customers": [50, 52, 54, 60, 61],
        "day_of_week": [3, 4, 5, 3, 4],
        "open": [1, 1, 1, 1, 1],
        "promo": [1, 0, 1, 0, 1],
        "state_holiday": ["0", "0", "0", "0", "0"],
        "school_holiday": [0, 0, 0, 0, 0],
        "store_type": ["a", "a", "a", "c", "c"],
        "assortment": ["a", "a", "a", "c", "c"],
        "competition_distance": [1270.0, 1270.0, 1270.0, 500.0, 500.0],
        "competition_open_since_month": [9, 9, 9, 0, 0],
        "competition_open_since_year": [2008, 2008, 2008, 0, 0],
        "promo2": [0, 0, 0, 1, 1],
        "promo2_since_week": [0, 0, 0, 14, 14],
        "promo2_since_year": [0, 0, 0, 2011, 2011],
        "promo_interval": ["None", "None", "None", "Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct"],
        "has_competition_date": [1, 1, 1, 0, 0],
        "year": [2015, 2015, 2015, 2015, 2015],
        "month": [7, 7, 7, 7, 7],
        "day": [1, 2, 3, 1, 2],
        "day_of_month": [1, 2, 3, 1, 2],
        "week_of_year": [27, 27, 27, 27, 27],
        "quarter": [3, 3, 3, 3, 3],
        "is_weekend": [0, 0, 0, 0, 0],
        "competition_open_months": [82, 82, 82, 0, 0],
        "promo2_open_months": [0, 0, 0, 51, 51],
        "is_state_holiday": [0, 0, 0, 0, 0],
        "sales_lag_1": [480, 500, 520, 590, 600],
        "sales_lag_7": [450, 460, 470, 550, 560],
        "sales_lag_14": [440, 445, 450, 540, 545],
        "sales_lag_28": [400, 410, 420, 500, 510],
        "sales_rolling_mean_7": [470.0, 475.0, 480.0, 560.0, 565.0],
        "sales_rolling_std_7": [10.0, 10.0, 10.0, 12.0, 12.0],
        "sales_rolling_mean_14": [450.0, 455.0, 460.0, 540.0, 545.0],
        "sales_rolling_std_14": [15.0, 15.0, 15.0, 18.0, 18.0],
        "sales_rolling_mean_28": [430.0, 435.0, 440.0, 520.0, 525.0],
        "sales_rolling_std_28": [20.0, 20.0, 20.0, 22.0, 22.0]
    })


def test_metrics_calculation(sample_metrics_data):
    y_true, y_pred = sample_metrics_data
    metrics = calculate_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "rmspe" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert not np.isnan(metrics["rmspe"])
    assert not np.isinf(metrics["rmspe"])


def test_rmspe_zero_safe_handling():
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([10.0, 0.0, 110.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert not np.isnan(metrics["rmspe"])
    assert not np.isinf(metrics["rmspe"])


def test_chronological_split_ordering(synthetic_feature_df):
    synthetic_feature_df["date"] = pd.to_datetime(synthetic_feature_df["date"])
    split_date = synthetic_feature_df["date"].min() + pd.Timedelta(days=2)
    
    train_df = synthetic_feature_df[synthetic_feature_df["date"] <= split_date]
    val_df = synthetic_feature_df[synthetic_feature_df["date"] > split_date]
    
    assert train_df["date"].max() < val_df["date"].min()


def test_target_and_customers_exclusion():
    excluded_features = ["date", "sales", "customers"]
    cols = ["store", "date", "sales", "customers", "promo", "sales_lag_7"]
    feature_cols = [c for c in cols if c not in excluded_features]
    
    assert "sales" not in feature_cols
    assert "customers" not in feature_cols
    assert "date" not in feature_cols
    assert "promo" in feature_cols
    assert "sales_lag_7" in feature_cols


def test_model_artifact_save_load(tmp_path):
    artifact_file = tmp_path / "test_model.joblib"
    metadata_file = tmp_path / "test_metadata.json"
    
    sample_artifact = {
        "model": "dummy_model",
        "feature_names": ["promo", "sales_lag_7"],
        "model_version": "1.0.0"
    }
    sample_metadata = {
        "best_model": "HistGradientBoostingRegressor",
        "model_version": "1.0.0"
    }
    
    save_model_artifact(sample_artifact, artifact_file)
    save_model_metadata(sample_metadata, metadata_file)
    
    loaded_art = load_model_artifact(artifact_file)
    loaded_meta = load_model_metadata(metadata_file)
    
    assert loaded_art["model_version"] == "1.0.0"
    assert loaded_art["feature_names"] == ["promo", "sales_lag_7"]
    assert loaded_meta["best_model"] == "HistGradientBoostingRegressor"


def test_prediction_utility_validations(synthetic_feature_df):
    # Load actual saved model artifact
    saved_artifact = load_model_artifact()
    
    # Predict on synthetic DataFrame
    preds = generate_forecast(synthetic_feature_df, artifact=saved_artifact)
    
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(synthetic_feature_df)
    assert (preds >= 0).all()
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()


def test_prediction_missing_required_column_raises(synthetic_feature_df):
    saved_artifact = load_model_artifact()
    incomplete_df = synthetic_feature_df.drop(columns=["sales_lag_7"])
    
    with pytest.raises(ValueError, match="Missing required feature columns"):
        generate_forecast(incomplete_df, artifact=saved_artifact)
