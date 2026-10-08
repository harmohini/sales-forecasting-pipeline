"""
Unit and integration tests for data ingestion module (src/ingestion/ingest.py).
"""

import pytest
import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine, test_connection as check_db_connection
from src.database.models import create_tables, StoresRaw, SalesTrainRaw, SalesTestRaw
from src.ingestion.ingest import ingest_file


@pytest.fixture
def db_engine():
    engine = get_engine()
    create_tables(engine)
    yield engine
    # Cleanup only synthetic test store IDs after tests
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM stores_raw WHERE store >= 990000;"))
        conn.execute(text("DELETE FROM sales_train_raw WHERE store >= 990000;"))


@pytest.fixture
def sample_store_csv(tmp_path):
    csv_file = tmp_path / "store.csv"
    df = pd.DataFrame({
        "Store": [999901, 999902],
        "StoreType": ["a", "b"],
        "Assortment": ["a", "c"],
        "CompetitionDistance": [500.0, 1200.0],
        "CompetitionOpenSinceMonth": [3.0, None],
        "CompetitionOpenSinceYear": [2012.0, None],
        "Promo2": [1, 0],
        "Promo2SinceWeek": [14.0, None],
        "Promo2SinceYear": [2011.0, None],
        "PromoInterval": ["Jan,Apr,Jul,Oct", None]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_train_csv(tmp_path):
    csv_file = tmp_path / "train.csv"
    df = pd.DataFrame({
        "Store": [999901, 999901, 999902],
        "DayOfWeek": [5, 4, 5],
        "Date": ["2015-07-31", "2015-07-30", "2015-07-31"],
        "Sales": [5000, 4800, 6200],
        "Customers": [500, 490, 610],
        "Open": [1, 1, 1],
        "Promo": [1, 1, 1],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [1, 0, 1]
    })
    df.to_csv(csv_file, index=False)
    return csv_file


def test_database_connection(db_engine):
    success, msg = check_db_connection(db_engine)
    assert success is True
    assert "successful" in msg


def test_ingest_store_file(db_engine, sample_store_csv):
    res = ingest_file(sample_store_csv, engine=db_engine, archive_on_success=False)
    assert res["status"] == "SUCCESS"
    assert res["rows_processed"] == 2
    assert res["rows_inserted"] == 2
    assert res["duplicates_skipped"] == 0

    with db_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM stores_raw WHERE store IN (999901, 999902);")).scalar()
        assert count == 2


def test_ingest_idempotency(db_engine, sample_store_csv):
    # First run
    res1 = ingest_file(sample_store_csv, engine=db_engine, archive_on_success=False)
    assert res1["status"] == "SUCCESS"

    # Second run (exact same file)
    res2 = ingest_file(sample_store_csv, engine=db_engine, archive_on_success=False)
    assert res2["status"] == "SUCCESS"
    assert res2["rows_processed"] == 2
    assert res2["rows_inserted"] == 0
    assert res2["duplicates_skipped"] == 2


def test_ingest_train_file(db_engine, sample_train_csv):
    res = ingest_file(sample_train_csv, engine=db_engine, archive_on_success=False)
    assert res["status"] == "SUCCESS"
    assert res["rows_processed"] == 3
    assert res["rows_inserted"] >= 0
