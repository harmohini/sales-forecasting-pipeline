"""
Pydantic Request & Response Data Schemas.

Defines schemas for:
- Sales Forecast Request (36 model features)
- Sales Forecast Response
- Health Status Response
- Model Information Response
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ForecastRequest(BaseModel):
    """
    Schema for sales forecast prediction request input parameters.
    Excludes date, sales, and customers (unavailable at inference time).
    """
    store: int = Field(..., description="Store ID integer (> 0)")
    day_of_week: int = Field(..., description="Day of week (1=Monday .. 7=Sunday)")
    open: int = Field(..., description="Store open status (1=Open, 0=Closed)")
    promo: int = Field(..., description="Promo active status (1=Active, 0=Inactive)")
    state_holiday: str = Field("0", description="State holiday category ('0', 'a', 'b', 'c')")
    school_holiday: int = Field(..., description="School holiday status (1=Yes, 0=No)")
    store_type: str = Field(..., description="Store type category ('a', 'b', 'c', 'd')")
    assortment: str = Field(..., description="Assortment level category ('a', 'b', 'c')")
    competition_distance: float = Field(..., description="Distance in meters to nearest competitor")
    competition_open_since_month: int = Field(0, description="Month competitor opened (0 if unknown)")
    competition_open_since_year: int = Field(0, description="Year competitor opened (0 if unknown)")
    promo2: int = Field(..., description="Promo2 participation status (1=Yes, 0=No)")
    promo2_since_week: int = Field(0, description="Promo2 start week (0 if not participating)")
    promo2_since_year: int = Field(0, description="Promo2 start year (0 if not participating)")
    promo_interval: str = Field("None", description="Promo2 recurring interval months")
    has_competition_date: int = Field(..., description="Indicator if competition open date is known (1/0)")
    year: int = Field(..., description="Calendar year")
    month: int = Field(..., description="Calendar month (1-12)")
    day: int = Field(..., description="Calendar day of month (1-31)")
    day_of_month: int = Field(..., description="Day of month (1-31)")
    week_of_year: int = Field(..., description="ISO week of year (1-53)")
    quarter: int = Field(..., description="Calendar quarter (1-4)")
    is_weekend: int = Field(..., description="Weekend status (1=Saturday/Sunday, 0=Weekday)")
    competition_open_months: float = Field(0.0, description="Months competitor has been open")
    promo2_open_months: float = Field(0.0, description="Months Promo2 has been active")
    is_state_holiday: int = Field(..., description="Binary state holiday flag (1=Holiday, 0=None)")
    sales_lag_1: float = Field(..., description="Sales value from 1 day prior")
    sales_lag_7: float = Field(..., description="Sales value from 7 days prior")
    sales_lag_14: float = Field(..., description="Sales value from 14 days prior")
    sales_lag_28: float = Field(..., description="Sales value from 28 days prior")
    sales_rolling_mean_7: float = Field(..., description="7-day historical rolling mean sales")
    sales_rolling_std_7: float = Field(..., description="7-day historical rolling std sales")
    sales_rolling_mean_14: float = Field(..., description="14-day historical rolling mean sales")
    sales_rolling_std_14: float = Field(..., description="14-day historical rolling std sales")
    sales_rolling_mean_28: float = Field(..., description="28-day historical rolling mean sales")
    sales_rolling_std_28: float = Field(..., description="28-day historical rolling std sales")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ForecastResponse(BaseModel):
    """
    Schema for sales forecast prediction response output.
    """
    predicted_sales: float = Field(..., description="Non-negative predicted sales volume")
    model_version: str = Field("1.0.0", description="Model version tag")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_sales": 5419.86,
                "model_version": "1.0.0"
            }
        }
    }


class HealthResponse(BaseModel):
    """
    Schema for API health status check.
    """
    status: str = Field("ok", description="Status code")
    service: str = Field("sales-forecasting-api", description="Service name")


class ModelInfoResponse(BaseModel):
    """
    Schema for model metadata and validation metrics.
    """
    model_type: str = Field(..., description="Estimator architecture name")
    model_version: str = Field(..., description="Model semantic version")
    validation_rmspe: float = Field(..., description="Validation RMSPE percentage")
    validation_mae: float = Field(..., description="Validation MAE")
    validation_rmse: float = Field(..., description="Validation RMSE")
    training_data_range: str = Field(..., description="Historical training date interval")
    validation_data_range: str = Field(..., description="Validation date interval")
    training_rows: int = Field(..., description="Total training rows")
    validation_rows: int = Field(..., description="Total validation rows")
    features_count: int = Field(..., description="Total model features")
