"""
Data Ingestion Pipeline Component.

Responsible for:
- Discovering incoming CSV files in data/incoming/
- Validating raw CSV schemas and records
- Chunked reading and inserting into PostgreSQL raw tables with ON CONFLICT DO NOTHING idempotency
- Safe archiving of successfully ingested files
"""

import time
import shutil
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import INCOMING_DATA_DIR, ARCHIVE_DATA_DIR
from src.ingestion.logger import setup_logger
from src.ingestion.validator import validate_dataset, SchemaValidationError, DataValidationError
from src.database.connection import get_engine
from src.database.models import SalesTrainRaw, StoresRaw, SalesTestRaw

logger = setup_logger("ingestion_logger")

DATASET_CONFIG = {
    "train.csv": {
        "dataset_type": "train",
        "model": SalesTrainRaw,
        "conflict_keys": ["store", "date"],
        "table_name": "sales_train_raw",
        "column_map": {
            "Store": "store",
            "DayOfWeek": "day_of_week",
            "Date": "date",
            "Sales": "sales",
            "Customers": "customers",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday"
        }
    },
    "store.csv": {
        "dataset_type": "store",
        "model": StoresRaw,
        "conflict_keys": ["store"],
        "table_name": "stores_raw",
        "column_map": {
            "Store": "store",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
            "CompetitionOpenSinceMonth": "competition_open_since_month",
            "CompetitionOpenSinceYear": "competition_open_since_year",
            "Promo2": "promo2",
            "Promo2SinceWeek": "promo2_since_week",
            "Promo2SinceYear": "promo2_since_year",
            "PromoInterval": "promo_interval"
        }
    },
    "test.csv": {
        "dataset_type": "test",
        "model": SalesTestRaw,
        "conflict_keys": ["id"],
        "table_name": "sales_test_raw",
        "column_map": {
            "Id": "id",
            "Store": "store",
            "DayOfWeek": "day_of_week",
            "Date": "date",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday"
        }
    }
}


def discover_incoming_files(directory: Path = INCOMING_DATA_DIR) -> list:
    """
    Discover all CSV files in the incoming directory.
    """
    if not directory.exists():
        logger.warning(f"Incoming directory '{directory}' does not exist.")
        return []
    files = list(directory.glob("*.csv"))
    logger.info(f"Discovered {len(files)} CSV file(s) in {directory}")
    return files


def prepare_chunk_records(chunk: pd.DataFrame, column_map: dict) -> list:
    """
    Rename DataFrame columns, handle date formatting and nulls, and convert to records list.
    """
    chunk = chunk.rename(columns=column_map).copy()

    # Convert Date column to date objects if present
    if "date" in chunk.columns:
        chunk["date"] = pd.to_datetime(chunk["date"]).dt.date

    # Ensure StateHoliday is string
    if "state_holiday" in chunk.columns:
        chunk["state_holiday"] = chunk["state_holiday"].astype(str)

    # Cast nullable integer float columns to python int/None
    int_cols = [
        "competition_open_since_month", "competition_open_since_year",
        "promo2_since_week", "promo2_since_year", "open"
    ]
    for c in int_cols:
        if c in chunk.columns:
            chunk[c] = chunk[c].apply(lambda x: int(x) if pd.notnull(x) else None)

    # Convert NaNs to None for database NULLs
    cleaned_df = chunk.astype(object).where(pd.notnull(chunk), None)
    return cleaned_df.to_dict(orient="records")


def ingest_file(file_path: Path, engine=None, chunksize: int = 100_000, archive_on_success: bool = True) -> dict:
    """
    Validate and ingest a single CSV file into PostgreSQL table.
    """
    start_time = time.time()
    filename = file_path.name

    if filename not in DATASET_CONFIG:
        logger.warning(f"Skipping unrecognized file: {filename}")
        return {
            "filename": filename,
            "status": "SKIPPED",
            "reason": "Unrecognized dataset file",
            "rows_processed": 0,
            "rows_inserted": 0,
            "duplicates_skipped": 0,
            "error_count": 0,
            "duration": 0
        }

    config = DATASET_CONFIG[filename]
    dataset_type = config["dataset_type"]
    model = config["model"]
    conflict_keys = config["conflict_keys"]
    table_name = config["table_name"]
    column_map = config["column_map"]

    logger.info(f"--- Starting ingestion for {filename} ---")
    if engine is None:
        engine = get_engine()

    # Step 1: Validate file using sample/preview
    logger.info(f"Validating file schema and structure: {filename}")
    try:
        sample_df = pd.read_csv(file_path, nrows=5000, low_memory=False)
        val_result = validate_dataset(sample_df, dataset_type)
        logger.info(f"Validation PASSED for {filename}: {val_result['total_rows']} sample rows checked.")
    except (SchemaValidationError, DataValidationError) as e:
        logger.error(f"Validation FAILED for {filename}: {str(e)}")
        return {
            "filename": filename,
            "status": "VALIDATION_FAILED",
            "error": str(e),
            "rows_processed": 0,
            "rows_inserted": 0,
            "duplicates_skipped": 0,
            "error_count": 1,
            "duration": round(time.time() - start_time, 2)
        }

    # Step 2: Database Insertion (Chunked)
    logger.info(f"Database insertion started for {filename} into '{table_name}' table...")

    # Record initial table row count
    with engine.connect() as conn:
        initial_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};")).scalar() or 0

    total_rows_processed = 0
    chunk_count = 0

    try:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):

            chunk_count += 1
            total_rows_processed += len(chunk)
            records = prepare_chunk_records(chunk, column_map)

            stmt = pg_insert(model).values(records).on_conflict_do_nothing(index_elements=conflict_keys)

            with engine.begin() as conn:
                conn.execute(stmt)

            logger.info(f"Chunk {chunk_count} processed ({len(records)} rows) for {filename}.")

        # Query final table row count to calculate exact inserted vs duplicate count
        with engine.connect() as conn:
            final_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};")).scalar() or 0

        rows_inserted = final_count - initial_count
        duplicates_skipped = total_rows_processed - rows_inserted
        duration = round(time.time() - start_time, 2)

        logger.info(f"Database insertion completed for {filename}: "
                    f"{rows_inserted} inserted, {duplicates_skipped} duplicates skipped, in {duration}s.")

        # Step 3: Archive file if requested and fully successful
        if archive_on_success:
            archive_file(file_path)

        return {
            "filename": filename,
            "status": "SUCCESS",
            "rows_processed": total_rows_processed,
            "rows_inserted": rows_inserted,
            "duplicates_skipped": duplicates_skipped,
            "error_count": 0,
            "duration": duration
        }

    except Exception as e:
        logger.error(f"Error inserting {filename} into database: {str(e)}", exc_info=True)
        return {
            "filename": filename,
            "status": "FAILED",
            "error": str(e),
            "rows_processed": total_rows_processed,
            "rows_inserted": 0,
            "duplicates_skipped": 0,
            "error_count": 1,
            "duration": round(time.time() - start_time, 2)
        }


def archive_file(file_path: Path):
    """
    Move processed file to archive directory.
    """
    ARCHIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = ARCHIVE_DATA_DIR / file_path.name
    shutil.move(str(file_path), str(dest_path))
    logger.info(f"Archived {file_path.name} to {dest_path}")
