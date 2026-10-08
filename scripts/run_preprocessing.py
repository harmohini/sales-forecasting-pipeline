"""
Stage 2 - Data Preprocessing & Feature Engineering CLI Script.

Executes end-to-end preprocessing pipeline:
1. Connects to PostgreSQL database
2. Loads raw sales and store data
3. Validates and cleans raw datasets
4. Merges sales history with store metadata
5. Generates date, store, promo, historical lag, and rolling window features
6. Performs chronological time-series split preparation
7. Persists processed dataset to data/processed/sales_features.parquet and PostgreSQL sales_features table
8. Displays comprehensive preprocessing summary
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import PROCESSED_DATA_DIR
from src.ingestion.logger import setup_logger
from src.database.connection import get_engine, test_connection
from src.database.queries import fetch_raw_sales_data, fetch_raw_store_data, save_processed_features
from src.preprocessing.cleaning import (
    clean_sales_data, clean_store_data, merge_sales_store_data
)
from src.preprocessing.feature_engineering import (
    create_date_features, create_store_business_features, create_promotion_features,
    create_lag_features, create_rolling_features, prepare_modeling_data, time_series_split
)

logger = setup_logger("preprocessing_main")


def run_stage_2_pipeline(save_to_db: bool = True) -> dict:
    """
    Execute full Stage 2 Preprocessing & Feature Engineering pipeline.
    """
    start_time = time.time()
    logger.info("========================================")
    logger.info("STARTING STAGE 2 - DATA PREPROCESSING")
    logger.info("========================================")

    # 1. Connect to PostgreSQL
    engine = get_engine()
    connected, conn_msg = test_connection(engine)
    if not connected:
        logger.error(f"Database connection failed: {conn_msg}")
        return {"status": "FAILED", "error": conn_msg}

    # 2 & 3. Load raw data
    raw_sales_df = fetch_raw_sales_data(engine)
    raw_store_df = fetch_raw_store_data(engine)
    raw_sales_count = len(raw_sales_df)
    raw_store_count = len(raw_store_df)

    # 4, 5, 6. Clean data
    clean_sales = clean_sales_data(raw_sales_df)
    clean_store = clean_store_data(raw_store_df)
    after_cleaning_count = len(clean_sales)

    # 7. Merge sales + store data
    merged_df = merge_sales_store_data(clean_sales, clean_store)
    merged_count = len(merged_df)

    # 8. Date features
    df_feat = create_date_features(merged_df)
    date_features_list = ['year', 'month', 'day', 'day_of_week', 'day_of_month', 'week_of_year', 'quarter', 'is_weekend']
    date_feature_count = len(date_features_list)

    # 9. Store & Business / Promo features
    df_feat = create_store_business_features(df_feat)
    df_feat = create_promotion_features(df_feat)

    # 10. Lag features
    lags = [1, 7, 14, 28]
    df_feat = create_lag_features(df_feat, lags=lags)
    lag_feature_count = len(lags)

    # 11. Rolling features
    windows = [7, 14, 28]
    df_feat = create_rolling_features(df_feat, windows=windows)
    rolling_feature_count = len(windows) * 2  # mean and std for each window

    # 12. Filter open stores & drop lag boundary NaNs
    before_prep = len(df_feat)
    final_df = prepare_modeling_data(df_feat, open_only=True, drop_na_lags=True)
    final_count = len(final_df)
    rows_removed_lag = before_prep - final_count

    # 13. Chronological Time-Series Split
    train_df, val_df = time_series_split(final_df, val_days=42)

    # 14. Save processed output
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "sales_features.parquet"
    final_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved processed Parquet dataset to: {parquet_path} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")

    if save_to_db:
        logger.info("Persisting processed features into PostgreSQL table 'sales_features'...")
        save_processed_features(final_df, engine, table_name="sales_features")
        logger.info("Successfully updated PostgreSQL table 'sales_features'.")

    duration = round(time.time() - start_time, 2)

    return {
        "status": "SUCCESS",
        "raw_sales_count": raw_sales_count,
        "raw_store_count": raw_store_count,
        "after_cleaning_count": after_cleaning_count,
        "merged_count": merged_count,
        "date_feature_count": date_feature_count,
        "lag_feature_count": lag_feature_count,
        "rolling_feature_count": rolling_feature_count,
        "rows_removed_lag": rows_removed_lag,
        "final_count": final_count,
        "train_min_date": train_df['date'].min().strftime("%Y-%m-%d"),
        "train_max_date": train_df['date'].max().strftime("%Y-%m-%d"),
        "val_min_date": val_df['date'].min().strftime("%Y-%m-%d"),
        "val_max_date": val_df['date'].max().strftime("%Y-%m-%d"),
        "parquet_path": str(parquet_path),
        "duration": duration
    }


def print_summary(res: dict):
    """Print formatted summary of Stage 2 preprocessing."""
    print("\n========================================")
    print("STAGE 2 - DATA PREPROCESSING")
    print("========================================\n")
    print(f"Raw sales rows: {res.get('raw_sales_count', 0):,}")
    print(f"Raw store rows: {res.get('raw_store_count', 0):,}\n")
    print(f"After cleaning: {res.get('after_cleaning_count', 0):,}\n")
    print(f"Merged rows: {res.get('merged_count', 0):,}\n")
    print(f"Date features created: {res.get('date_feature_count', 0)}")
    print(f"Lag features created: {res.get('lag_feature_count', 0)}")
    print(f"Rolling features created: {res.get('rolling_feature_count', 0)}\n")
    print(f"Rows removed (closed stores / lag boundary): {res.get('rows_removed_lag', 0):,}\n")
    print(f"Final modeling rows: {res.get('final_count', 0):,}\n")
    print("Training period:")
    print(f"{res.get('train_min_date')} → {res.get('train_max_date')}\n")
    print("Validation period:")
    print(f"{res.get('val_min_date')} → {res.get('val_max_date')}\n")
    print("Output:")
    print(f"{res.get('parquet_path')}")
    print("PostgreSQL table: sales_features\n")
    print(f"STATUS: {res.get('status', 'FAILED')}")
    print("========================================\n")


if __name__ == "__main__":
    res = run_stage_2_pipeline(save_to_db=True)
    print_summary(res)
    if res.get("status") == "FAILED":
        sys.exit(1)
