"""
Data Cleaning Component.

Responsible for:
- Loading raw sales and store data from database
- Data type conversion and datetime parsing
- Missing value imputation with documented domain rationale
- Duplicate and negative value validation
- Validation-safe inner merge between sales history and store metadata
"""

import logging
import pandas as pd
import numpy as np
from src.database.queries import fetch_raw_sales_data, fetch_raw_store_data

logger = logging.getLogger("ingestion_logger")

def load_sales_data(engine=None) -> pd.DataFrame:
    """Load raw sales data from PostgreSQL database."""
    logger.info("Loading raw sales data from PostgreSQL database...")
    df = fetch_raw_sales_data(engine)
    logger.info(f"Loaded {len(df)} sales records.")
    return df

def load_store_data(engine=None) -> pd.DataFrame:
    """Load raw store data from PostgreSQL database."""
    logger.info("Loading raw store data from PostgreSQL database...")
    df = fetch_raw_store_data(engine)
    logger.info(f"Loaded {len(df)} store records.")
    return df

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw sales training data:
    - Convert date to pandas Datetime
    - Normalize state_holiday to standard string categories ('0', 'a', 'b', 'c')
    - Validate numeric domain constraints (store > 0, sales >= 0, customers >= 0)
    - Detect and drop duplicate (store, date) records
    """
    df = df.copy()
    initial_rows = len(df)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Normalize state_holiday column to string ('0', 'a', 'b', 'c')
    df['state_holiday'] = df['state_holiday'].astype(str).replace({'0.0': '0', '0': '0'})

    # Duplicate check on (store, date)
    duplicates = df.duplicated(subset=['store', 'date']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate (store, date) rows in sales data. Retaining first occurrences.")
        df = df.drop_duplicates(subset=['store', 'date'], keep='first')

    # Detect negative values
    invalid_sales = (df['sales'] < 0).sum()
    invalid_customers = (df['customers'] < 0).sum()
    invalid_stores = (df['store'] <= 0).sum()

    if invalid_sales > 0 or invalid_customers > 0 or invalid_stores > 0:
        logger.warning(f"Found invalid records: {invalid_sales} negative sales, {invalid_customers} negative customers, {invalid_stores} invalid stores.")
        df = df[(df['sales'] >= 0) & (df['customers'] >= 0) & (df['store'] > 0)]

    rows_removed = initial_rows - len(df)
    logger.info(f"Sales cleaning completed: {len(df)} rows retained ({rows_removed} rows removed/corrected).")
    return df

def clean_store_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw store metadata:
    - Impute missing competition_distance with median distance (3 missing stores)
    - Impute competition_open_since_month/year with 0 and add has_competition_date flag (354 missing)
    - Impute promo2_since_week/year with 0 for stores with promo2 == 0 (544 missing)
    - Impute promo_interval with 'None' for stores with promo2 == 0 (544 missing)
    - Validate duplicate store IDs
    """
    df = df.copy()

    # Check store duplicate IDs
    duplicate_stores = df.duplicated(subset=['store']).sum()
    if duplicate_stores > 0:
        logger.warning(f"Found {duplicate_stores} duplicate store IDs. Keeping first occurrence.")
        df = df.drop_duplicates(subset=['store'], keep='first')

    # 1. competition_distance: fill missing (3 rows) with median competition distance
    # Rationale: 3 stores missing competition_distance represent stores with unknown/distant competition.
    # Imputing median distance prevents extreme outliers while allowing model numeric evaluation.
    if df['competition_distance'].isnull().sum() > 0:
        median_dist = df['competition_distance'].median()
        df['competition_distance'] = df['competition_distance'].fillna(median_dist)
        logger.info(f"Imputed {df['competition_distance'].isnull().sum()} missing competition_distance values with median ({median_dist}m).")

    # 2. competition_open_since_month & competition_open_since_year (354 missing)
    # Rationale: Missing competition opening dates mean the exact start date of competition is unknown.
    # We add a binary indicator `has_competition_date` to preserve this distinction, filling dates with 0.
    df['has_competition_date'] = np.where(
        df['competition_open_since_month'].notnull() & df['competition_open_since_year'].notnull(), 1, 0
    )
    df['competition_open_since_month'] = df['competition_open_since_month'].fillna(0).astype(int)
    df['competition_open_since_year'] = df['competition_open_since_year'].fillna(0).astype(int)

    # 3. promo2_since_week, promo2_since_year, promo_interval (544 missing where promo2 == 0)
    # Rationale: Stores where promo2 == 0 do not participate in Promo2, so start week/year and interval are naturally None/0.
    df['promo2_since_week'] = df['promo2_since_week'].fillna(0).astype(int)
    df['promo2_since_year'] = df['promo2_since_year'].fillna(0).astype(int)
    df['promo_interval'] = df['promo_interval'].fillna('None').astype(str)

    logger.info(f"Store metadata cleaning completed for {len(df)} stores.")
    return df

def merge_sales_store_data(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validation-safe merge of sales data with store metadata on 'store'.
    Ensures row count matches sales_df and no unexpected row multiplication occurs.
    """
    initial_sales_rows = len(sales_df)
    
    # Verify all sales stores exist in store metadata
    unmatched_stores = set(sales_df['store']) - set(store_df['store'])
    if unmatched_stores:
        logger.error(f"Found {len(unmatched_stores)} stores in sales data missing from store metadata: {unmatched_stores}")
        raise ValueError(f"Unmatched stores found in sales dataset: {unmatched_stores}")

    # Inner join on 'store'
    merged_df = pd.merge(sales_df, store_df, on='store', how='inner')

    # Verify no row multiplication or row loss
    if len(merged_df) != initial_sales_rows:
        raise ValueError(
            f"Merge produced row count mismatch! Initial sales rows: {initial_sales_rows}, Merged rows: {len(merged_df)}"
        )

    logger.info(f"Successfully merged sales data with store metadata. Total rows: {len(merged_df)}")
    return merged_df
