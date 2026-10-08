"""
Integration and unit tests for FastAPI REST endpoints (api/main.py, api/routes.py).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.forecasting.predict import generate_forecast
from src.forecasting.model_utils import load_model_artifact
import pandas as pd
import numpy as np

client = TestClient(app)

@pytest.fixture
def valid_sample_payload():
    """Valid sample payload corresponding to Store 1 on 2015-07-31 validation row."""
    return {
        "store": 1,
        "day_of_week": 5,
        "open": 1,
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 1,
        "store_type": "c",
        "assortment": "a",
        "competition_distance": 1270.0,
        "competition_open_since_month": 9,
        "competition_open_since_year": 2008,
        "promo2": 0,
        "promo2_since_week": 0,
        "promo2_since_year": 0,
        "promo_interval": "None",
        "has_competition_date": 1,
        "year": 2015,
        "month": 7,
        "day": 31,
        "day_of_month": 31,
        "week_of_year": 31,
        "quarter": 3,
        "is_weekend": 0,
        "competition_open_months": 82.0,
        "promo2_open_months": 0.0,
        "is_state_holiday": 0,
        "sales_lag_1": 5020.0,
        "sales_lag_7": 3706.0,
        "sales_lag_14": 4852.0,
        "sales_lag_28": 4665.0,
        "sales_rolling_mean_7": 4140.71,
        "sales_rolling_std_7": 1965.23,
        "sales_rolling_mean_14": 3816.36,
        "sales_rolling_std_14": 1763.35,
        "sales_rolling_mean_28": 3757.93,
        "sales_rolling_std_28": 1676.19
    }


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Sales Forecasting REST API is online."


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "sales-forecasting-api"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "HistGradientBoostingRegressor"
    assert data["model_version"] == "1.0.0"
    assert "validation_rmspe" in data
    assert "validation_mae" in data
    assert "validation_rmse" in data
    assert data["features_count"] == 36


def test_predict_endpoint_valid_payload(valid_sample_payload):
    response = client.post("/predict", json=valid_sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sales" in data
    assert isinstance(data["predicted_sales"], (int, float))
    assert data["predicted_sales"] >= 0
    assert not np.isnan(data["predicted_sales"])
    assert not np.isinf(data["predicted_sales"])


def test_predict_matches_direct_model_inference(valid_sample_payload):
    """
    CRITICAL PARITY TEST:
    Verify that API response matches direct Python model inference within floating-point tolerance.
    """
    # 1. API Call
    response = client.post("/predict", json=valid_sample_payload)
    assert response.status_code == 200
    api_predicted_sales = response.json()["predicted_sales"]

    # 2. Direct Python Inference using model artifact
    artifact = load_model_artifact()
    df_sample = pd.DataFrame([valid_sample_payload])
    direct_preds = generate_forecast(df_sample, artifact=artifact)
    direct_predicted_sales = round(float(direct_preds[0]), 2)

    # 3. Parity Check
    assert abs(api_predicted_sales - direct_predicted_sales) < 0.1, (
        f"API prediction ({api_predicted_sales}) does not match direct inference ({direct_predicted_sales})"
    )


def test_predict_missing_required_feature(valid_sample_payload):
    invalid_payload = valid_sample_payload.copy()
    del invalid_payload["sales_lag_7"]  # Delete required feature
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_invalid_field_type(valid_sample_payload):
    invalid_payload = valid_sample_payload.copy()
    invalid_payload["store"] = "not_an_int"
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
