"""
Unit tests for Stage 2 Preprocessing & Feature Engineering modules:
- src/preprocessing/cleaning.py
- src/preprocessing/feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing.cleaning import (
    clean_sales_data, clean_store_data, merge_sales_store_data
)
from src.preprocessing.feature_engineering import (
    create_date_features, create_store_business_features, create_promotion_features,
    create_lag_features, create_rolling_features, prepare_modeling_data, time_series_split
)


@pytest.fixture
def synthetic_sales_df():
    """Small synthetic sales dataset covering 5 days for 2 stores."""
    dates = pd.date_range("2015-07-01", periods=5, freq="D")
    data = []
    for store_id in [1, 2]:
        for i, d in enumerate(dates):
            data.append({
                "store": store_id,
                "date": d,
                "day_of_week": d.dayofweek + 1,
                "sales": (store_id * 100) + (i + 1) * 10,
                "customers": 50,
                "open": 1,
                "promo": 1 if i % 2 == 0 else 0,
                "state_holiday": "0",
                "school_holiday": 0
            })
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_store_df():
    """Small synthetic store metadata dataset."""
    return pd.DataFrame({
        "store": [1, 2],
        "store_type": ["c", "a"],
        "assortment": ["a", "a"],
        "competition_distance": [1270.0, None],  # One missing distance
        "competition_open_since_month": [9.0, None],
        "competition_open_since_year": [2008.0, None],
        "promo2": [0, 1],
        "promo2_since_week": [None, 14.0],
        "promo2_since_year": [None, 2011.0],
        "promo_interval": [None, "Jan,Apr,Jul,Oct"]
    })


def test_clean_sales_data(synthetic_sales_df):
    # Add duplicate and negative rows
    dup_row = synthetic_sales_df.iloc[[0]].copy()
    synthetic_sales_df = pd.concat([synthetic_sales_df, dup_row], ignore_index=True)
    
    cleaned = clean_sales_data(synthetic_sales_df)
    assert len(cleaned) == 10  # Duplicate removed
    assert pd.api.types.is_datetime64_any_dtype(cleaned['date'])


def test_clean_store_data(synthetic_store_df):
    cleaned = clean_store_data(synthetic_store_df)
    assert cleaned['competition_distance'].isnull().sum() == 0
    assert cleaned.loc[cleaned['store'] == 2, 'competition_distance'].values[0] == 1270.0  # Median
    assert 'has_competition_date' in cleaned.columns
    assert cleaned.loc[cleaned['store'] == 1, 'has_competition_date'].values[0] == 1
    assert cleaned.loc[cleaned['store'] == 2, 'has_competition_date'].values[0] == 0


def test_merge_sales_store_data(synthetic_sales_df, synthetic_store_df):
    merged = merge_sales_store_data(synthetic_sales_df, synthetic_store_df)
    assert len(merged) == len(synthetic_sales_df)
    assert "store_type" in merged.columns


def test_date_features(synthetic_sales_df):
    df_feat = create_date_features(synthetic_sales_df)
    expected_cols = ["year", "month", "day", "day_of_week", "day_of_month", "week_of_year", "quarter", "is_weekend"]
    for col in expected_cols:
        assert col in df_feat.columns
    assert "date" in df_feat.columns  # Preserved original date column


def test_no_future_data_leakage_in_lags():
    """
    STRICT LEAKAGE TEST:
    Verify that lag_1 and lag_2 strictly retrieve historical sales from D-1 and D-2,
    and NEVER incorporate future sales from D+1 or D+2.
    """
    df = pd.DataFrame({
        "store": [1, 1, 1, 1],
        "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"]),
        "sales": [100, 200, 300, 400]
    })
    
    df_lag = create_lag_features(df, lags=[1, 2])
    
    # Date 1 (2015-01-01): Lags are NaN
    assert pd.isna(df_lag.loc[df_lag['date'] == "2015-01-01", "sales_lag_1"].values[0])
    
    # Date 2 (2015-01-02): sales_lag_1 MUST be 100 (Date 1)
    assert df_lag.loc[df_lag['date'] == "2015-01-02", "sales_lag_1"].values[0] == 100
    
    # Date 3 (2015-01-03): sales_lag_1 MUST be 200 (Date 2), sales_lag_2 MUST be 100 (Date 1)
    assert df_lag.loc[df_lag['date'] == "2015-01-03", "sales_lag_1"].values[0] == 200
    assert df_lag.loc[df_lag['date'] == "2015-01-03", "sales_lag_2"].values[0] == 100

    # Date 4 (2015-01-04): sales_lag_1 MUST be 300 (Date 3)
    assert df_lag.loc[df_lag['date'] == "2015-01-04", "sales_lag_1"].values[0] == 300


def test_no_future_data_leakage_in_rolling():
    """
    STRICT LEAKAGE TEST:
    Verify that rolling_mean_2 strictly uses sales from [D-2, D-1] (shifted by 1 day)
    and DOES NOT include current day's target sales at date D.
    """
    df = pd.DataFrame({
        "store": [1, 1, 1, 1],
        "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"]),
        "sales": [100, 200, 300, 400]
    })

    df_roll = create_rolling_features(df, windows=[2])

    # Date 1 & Date 2: NaN (insufficient 2-day historical window prior to current date)
    assert pd.isna(df_roll.loc[df_roll['date'] == "2015-01-01", "sales_rolling_mean_2"].values[0])
    assert pd.isna(df_roll.loc[df_roll['date'] == "2015-01-02", "sales_rolling_mean_2"].values[0])

    # Date 3 (2015-01-03): rolling mean of previous 2 sales (Date 1=100, Date 2=200) = 150.0
    # Current target (300) MUST NOT be included!
    val_date3 = df_roll.loc[df_roll['date'] == "2015-01-03", "sales_rolling_mean_2"].values[0]
    assert val_date3 == 150.0

    # Date 4 (2015-01-04): rolling mean of previous 2 sales (Date 2=200, Date 3=300) = 250.0
    val_date4 = df_roll.loc[df_roll['date'] == "2015-01-04", "sales_rolling_mean_2"].values[0]
    assert val_date4 == 250.0


def test_prepare_modeling_data(synthetic_sales_df):
    df_feat = create_lag_features(synthetic_sales_df, lags=[1])
    # Add a closed day record
    df_feat.loc[0, 'open'] = 0
    df_feat.loc[0, 'sales'] = 0

    df_prep = prepare_modeling_data(df_feat, open_only=True, drop_na_lags=True)
    assert (df_prep['open'] == 1).all()
    assert (df_prep['sales'] > 0).all()
    assert not df_prep['sales_lag_1'].isna().any()


def test_chronological_time_series_split():
    df = pd.DataFrame({
        "date": pd.date_range("2015-06-01", "2015-07-31", freq="D"),
        "sales": 100
    })
    train_df, val_df = time_series_split(df, val_days=10)
    assert val_df['date'].min() > train_df['date'].max()
    assert len(val_df) == 10
