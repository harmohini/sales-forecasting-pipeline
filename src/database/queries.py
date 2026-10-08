"""
Database Operations and Query Helpers.

Responsible for querying raw PostgreSQL datasets and persisting processed feature tables.
"""

import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine

def fetch_raw_sales_data(engine=None) -> pd.DataFrame:
    """
    Fetch raw sales training data from sales_train_raw table.
    """
    if engine is None:
        engine = get_engine()
    query = "SELECT store, date, day_of_week, sales, customers, open, promo, state_holiday, school_holiday FROM sales_train_raw;"
    return pd.read_sql(query, engine)

def fetch_raw_store_data(engine=None) -> pd.DataFrame:
    """
    Fetch raw store metadata from stores_raw table.
    """
    if engine is None:
        engine = get_engine()
    query = "SELECT store, store_type, assortment, competition_distance, competition_open_since_month, competition_open_since_year, promo2, promo2_since_week, promo2_since_year, promo_interval FROM stores_raw;"
    return pd.read_sql(query, engine)

def fetch_raw_test_data(engine=None) -> pd.DataFrame:
    """
    Fetch raw sales test data from sales_test_raw table.
    """
    if engine is None:
        engine = get_engine()
    query = "SELECT id, store, day_of_week, date, open, promo, state_holiday, school_holiday FROM sales_test_raw;"
    return pd.read_sql(query, engine)

def save_processed_features(df: pd.DataFrame, engine=None, table_name: str = "sales_features", chunksize: int = 50_000) -> None:
    """
    Persist processed feature dataset into PostgreSQL table.
    """
    if engine is None:
        engine = get_engine()
    
    # Save to PostgreSQL table (replace existing sales_features if exists)
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False, chunksize=chunksize)
