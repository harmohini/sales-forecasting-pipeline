"""
Feature Engineering Component.

Responsible for generating:
- Date/calendar features
- Store & business domain features
- Promotion/holiday indicators
- Strictly historical lag features (zero future leakage)
- Shifted rolling window statistics (zero future leakage)
- Chronological time-series train/validation splitting
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("ingestion_logger")


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar and temporal features from the 'date' column.
    Preserves original 'date' column.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek + 1  # 1 = Monday, 7 = Sunday
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)

    return df


def create_store_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate business features from store metadata.
    """
    df = df.copy()

    # Calculate competition open duration in months up to the current record date
    if 'competition_open_since_year' in df.columns and 'competition_open_since_month' in df.columns:
        has_comp = (df['competition_open_since_year'] > 0) & (df['competition_open_since_month'] > 0)
        comp_months = 12 * (df['year'] - df['competition_open_since_year']) + (df['month'] - df['competition_open_since_month'])
        df['competition_open_months'] = np.where(has_comp & (comp_months > 0), comp_months, 0)

    # Promo2 active duration in months
    if 'promo2_since_year' in df.columns and 'promo2_since_week' in df.columns:
        has_p2 = (df['promo2'] == 1) & (df['promo2_since_year'] > 0) & (df['promo2_since_week'] > 0)
        p2_months = 12 * (df['year'] - df['promo2_since_year']) + ((df['week_of_year'] - df['promo2_since_week']) / 4.0)
        df['promo2_open_months'] = np.where(has_p2 & (p2_months > 0), p2_months, 0)

    return df


def create_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize promotion and holiday binary/categorical indicator features.
    """
    df = df.copy()

    # Binary state holiday indicator
    if 'state_holiday' in df.columns:
        df['is_state_holiday'] = np.where(df['state_holiday'].astype(str).isin(['a', 'b', 'c']), 1, 0)
    
    return df


def create_lag_features(df: pd.DataFrame, lags: list = [1, 7, 14, 28]) -> pd.DataFrame:
    """
    Construct historical sales lag features per store.
    
    STRICT LEAKAGE PREVENTION:
    - Data is explicitly sorted by ['store', 'date']
    - Shift operations shift historical target values downwards per store.
    - For date D, sales_lag_7 strictly retrieves sales from D-7 (never D or D+k).
    """
    df = df.sort_values(by=['store', 'date']).reset_index(drop=True)

    for lag in lags:
        col_name = f"sales_lag_{lag}"
        df[col_name] = df.groupby('store')['sales'].shift(lag)

    logger.info(f"Generated historical lag features for lags: {lags}")
    return df


def create_rolling_features(df: pd.DataFrame, windows: list = [7, 14, 28]) -> pd.DataFrame:
    """
    Construct store-wise rolling mean and std features.
    
    STRICT LEAKAGE PREVENTION:
    - Target 'sales' is first shifted by 1 day per store (shift(1)).
    - Rolling window statistics are computed on the shifted series.
    - For date D, sales_rolling_mean_7 strictly uses sales from [D-7, D-1] (never D).
    """
    df = df.sort_values(by=['store', 'date']).reset_index(drop=True)

    # Shift sales by 1 day per store to prevent including current day's target
    sales_shifted = df.groupby('store')['sales'].shift(1)

    for w in windows:
        mean_col = f"sales_rolling_mean_{w}"
        std_col = f"sales_rolling_std_{w}"
        
        # Calculate rolling statistics per store on shifted series
        df[mean_col] = sales_shifted.groupby(df['store']).transform(lambda x: x.rolling(window=w, min_periods=w).mean())
        df[std_col] = sales_shifted.groupby(df['store']).transform(lambda x: x.rolling(window=w, min_periods=w).std())

    logger.info(f"Generated historical rolling window features for windows: {windows}")
    return df


def prepare_modeling_data(df: pd.DataFrame, open_only: bool = True, drop_na_lags: bool = True) -> pd.DataFrame:
    """
    Prepare modeling dataset:
    - Optionally filter open stores (open == 1 and sales > 0)
    - Optionally drop initial boundary rows where lag/rolling features are NaN
    """
    df = df.copy()
    initial_count = len(df)

    if open_only and 'open' in df.columns:
        df = df[(df['open'] == 1) & (df['sales'] > 0)]
        logger.info(f"Filtered for open stores (open == 1 & sales > 0): {len(df)} rows retained.")

    if drop_na_lags:
        lag_cols = [c for c in df.columns if c.startswith('sales_lag_') or c.startswith('sales_rolling_')]
        if lag_cols:
            df = df.dropna(subset=lag_cols)
            dropped = initial_count - len(df)
            logger.info(f"Dropped {dropped} rows with undefined historical lag/rolling values.")

    return df.reset_index(drop=True)


def time_series_split(df: pd.DataFrame, val_days: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological time-series split based on the most recent val_days (default 42 days / 6 weeks).
    Returns (train_df, val_df).
    """
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train_df = df[df['date'] <= split_date].copy()
    val_df = df[df['date'] > split_date].copy()

    logger.info(
        f"Chronological split completed (val_days={val_days}):\n"
        f"  Train period: {train_df['date'].min().strftime('%Y-%m-%d')} -> {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)\n"
        f"  Validation period: {val_df['date'].min().strftime('%Y-%m-%d')} -> {val_df['date'].max().strftime('%Y-%m-%d')} ({len(val_df)} rows)"
    )

    return train_df, val_df
