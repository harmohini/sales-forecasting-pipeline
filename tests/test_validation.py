"""
Unit tests for data validation module (src/ingestion/validator.py).
"""

import pytest
import pandas as pd
from src.ingestion.validator import (
    validate_dataset,
    validate_columns,
    validate_dates,
    check_duplicates,
    SchemaValidationError,
    DataValidationError
)


@pytest.fixture
def valid_train_df():
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "DayOfWeek": [5, 5, 5],
        "Date": ["2015-07-31", "2015-07-31", "2015-07-31"],
        "Sales": [5263, 6064, 8314],
        "Customers": [555, 625, 821],
        "Open": [1, 1, 1],
        "Promo": [1, 1, 1],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [1, 1, 1]
    })


@pytest.fixture
def valid_store_df():
    return pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["c", "a"],
        "Assortment": ["a", "a"],
        "CompetitionDistance": [1270.0, 570.0],
        "CompetitionOpenSinceMonth": [9.0, 11.0],
        "CompetitionOpenSinceYear": [2008.0, 2007.0],
        "Promo2": [0, 1],
        "Promo2SinceWeek": [None, 13.0],
        "Promo2SinceYear": [None, 2010.0],
        "PromoInterval": [None, "Jan,Apr,Jul,Oct"]
    })


@pytest.fixture
def valid_test_df():
    return pd.DataFrame({
        "Id": [1, 2],
        "Store": [1, 3],
        "DayOfWeek": [4, 4],
        "Date": ["2015-09-17", "2015-09-17"],
        "Open": [1, 1],
        "Promo": [1, 1],
        "StateHoliday": ["0", "0"],
        "SchoolHoliday": [0, 0]
    })


def test_valid_train_dataset(valid_train_df):
    res = validate_dataset(valid_train_df, "train")
    assert res["valid"] is True
    assert res["total_rows"] == 3
    assert res["duplicate_count"] == 0


def test_valid_store_dataset(valid_store_df):
    res = validate_dataset(valid_store_df, "store")
    assert res["valid"] is True
    assert res["total_rows"] == 2


def test_valid_test_dataset(valid_test_df):
    res = validate_dataset(valid_test_df, "test")
    assert res["valid"] is True
    assert res["total_rows"] == 2


def test_missing_required_column(valid_train_df):
    df_missing = valid_train_df.drop(columns=["Sales"])
    with pytest.raises(SchemaValidationError, match="Missing required columns"):
        validate_dataset(df_missing, "train")


def test_invalid_date_format(valid_train_df):
    valid_train_df.loc[0, "Date"] = "31-07-2015"  # Invalid format (not YYYY-MM-DD)
    with pytest.raises(DataValidationError, match="invalid or unparseable date"):
        validate_dataset(valid_train_df, "train")


def test_duplicate_records(valid_train_df):
    # Duplicate row with same Store and Date
    dup_df = pd.concat([valid_train_df, valid_train_df.iloc[[0]]], ignore_index=True)
    res = validate_dataset(dup_df, "train")
    assert res["duplicate_count"] == 1
    assert any("duplicate" in issue for issue in res["issues"])


def test_invalid_numeric_values(valid_train_df):
    # Negative sales
    valid_train_df.loc[0, "Sales"] = -100
    with pytest.raises(DataValidationError, match="negative Sales"):
        validate_dataset(valid_train_df, "train")


def test_invalid_store_id(valid_train_df):
    valid_train_df.loc[0, "Store"] = 0
    with pytest.raises(DataValidationError, match="invalid Store IDs"):
        validate_dataset(valid_train_df, "train")


def test_empty_csv():
    empty_df = pd.DataFrame()
    with pytest.raises((SchemaValidationError, DataValidationError)):
        validate_dataset(empty_df, "train")

