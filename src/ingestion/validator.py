"""
Data Validation Component.

Provides reusable schema and quality validation functions for Rossmann sales datasets:
- train.csv
- store.csv
- test.csv
"""

import pandas as pd
import numpy as np

class SchemaValidationError(Exception):
    """Raised when CSV schema does not match required columns."""
    pass

class DataValidationError(Exception):
    """Raised when critical data quality constraints are violated."""
    pass

# Expected columns for each dataset type
REQUIRED_COLUMNS = {
    "train": ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"],
    "store": ["Store", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"],
    "test": ["Id", "Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
}

# Unique/natural key columns for duplicate checking
KEY_COLUMNS = {
    "train": ["Store", "Date"],
    "store": ["Store"],
    "test": ["Id"]
}

def validate_columns(df: pd.DataFrame, dataset_type: str) -> list:
    """
    Validate that required columns exist in the DataFrame.
    Returns list of unexpected columns if any.
    """
    if dataset_type not in REQUIRED_COLUMNS:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'. Must be one of {list(REQUIRED_COLUMNS.keys())}")
    
    expected = set(REQUIRED_COLUMNS[dataset_type])
    actual = set(df.columns)
    
    missing = expected - actual
    if missing:
        raise SchemaValidationError(f"Missing required columns for {dataset_type}: {sorted(list(missing))}")
    
    unexpected = actual - expected
    return sorted(list(unexpected))

def check_duplicates(df: pd.DataFrame, keys: list) -> int:
    """
    Count duplicate records based on specified key columns.
    """
    if not keys or not set(keys).issubset(df.columns):
        return 0
    return int(df.duplicated(subset=keys).sum())

def validate_dates(df: pd.DataFrame, date_col: str = "Date") -> bool:
    """
    Validate that Date column contains valid YYYY-MM-DD formatted dates.
    Raises DataValidationError if parsing fails.
    """
    if date_col not in df.columns:
        return True
    
    # Try parsing dates
    parsed = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    invalid_count = parsed.isna().sum()
    if invalid_count > 0:
        raise DataValidationError(f"Found {invalid_count} invalid or unparseable date values in '{date_col}' column.")
    return True

def validate_dataset(df: pd.DataFrame, dataset_type: str) -> dict:
    """
    Comprehensive validation for train, store, or test dataset.
    Returns a dictionary summarizing validation results.
    """
    if df.empty:
        raise DataValidationError(f"Dataset '{dataset_type}' is empty (0 rows).")

    issues = []
    
    # 1. Column structure validation
    unexpected_cols = validate_columns(df, dataset_type)
    if unexpected_cols:
        issues.append(f"Unexpected columns present: {unexpected_cols}")
        
    # 2. Check nulls on critical key columns
    key_cols = KEY_COLUMNS.get(dataset_type, [])
    for kcol in key_cols:
        null_count = int(df[kcol].isna().sum())
        if null_count > 0:
            raise DataValidationError(f"Critical key column '{kcol}' in {dataset_type} contains {null_count} null values.")
            
    # 3. Duplicate row check
    duplicate_count = check_duplicates(df, key_cols)
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate rows based on keys {key_cols}.")
        
    # 4. Date format validation
    if "Date" in df.columns:
        validate_dates(df, "Date")
        
    # 5. Dataset specific numeric domain checks
    if "Store" in df.columns:
        invalid_stores = (df["Store"] <= 0).sum()
        if invalid_stores > 0:
            raise DataValidationError(f"Found {invalid_stores} invalid Store IDs (<= 0).")
            
    if dataset_type == "train":
        negative_sales = (df["Sales"] < 0).sum()
        if negative_sales > 0:
            raise DataValidationError(f"Found {negative_sales} negative Sales values.")
        negative_customers = (df["Customers"] < 0).sum()
        if negative_customers > 0:
            raise DataValidationError(f"Found {negative_customers} negative Customers values.")
            
    # Missing value audit
    null_summary = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    
    return {
        "dataset": dataset_type,
        "valid": True,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_count": duplicate_count,
        "missing_values": null_summary,
        "issues": issues
    }
