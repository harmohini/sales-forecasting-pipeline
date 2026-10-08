"""
Data Ingestion CLI Entry Point.

Executes end-to-end data ingestion pipeline:
1. Load configuration
2. Test database connection
3. Create PostgreSQL raw tables if missing
4. Discover incoming CSV files
5. Validate CSV schemas and data quality
6. Ingest datasets with ON CONFLICT DO NOTHING idempotency
7. Output structured summary report
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import INCOMING_DATA_DIR

from src.ingestion.logger import setup_logger
from src.database.connection import get_engine, test_connection
from src.database.models import create_tables
from src.ingestion.ingest import discover_incoming_files, ingest_file

logger = setup_logger("ingestion_main")

def run_ingestion_pipeline(archive_on_success: bool = False) -> dict:
    """
    Run full data ingestion pipeline and return summary metrics.
    archive_on_success: Set to True to move CSV files to data/archive/ after ingestion.
    """
    pipeline_start = time.time()
    logger.info("========================================")
    logger.info("STARTING SALES DATA INGESTION PIPELINE")
    logger.info("========================================")

    # 1 & 2. Test PostgreSQL Connection
    engine = get_engine()
    connected, conn_msg = test_connection(engine)
    if not connected:
        logger.error(f"PostgreSQL connection check failed: {conn_msg}")
        return {"status": "FAILED", "error": conn_msg}
    logger.info(f"Database connection status: {conn_msg}")

    # 3. Create tables if required
    logger.info("Ensuring PostgreSQL raw database tables exist...")
    create_tables(engine)

    # 4. Discover files
    incoming_files = discover_incoming_files(INCOMING_DATA_DIR)
    if not incoming_files:
        logger.warning(f"No CSV files found in {INCOMING_DATA_DIR}")
        return {"status": "NO_FILES", "file_summaries": []}

    # Expected order: store.csv, test.csv, train.csv (store metadata first)
    order = ["store.csv", "test.csv", "train.csv"]
    file_map = {f.name: f for f in incoming_files}
    files_to_process = [file_map[name] for name in order if name in file_map]
    # Add any remaining unrecognized CSV files
    for f in incoming_files:
        if f not in files_to_process:
            files_to_process.append(f)

    file_summaries = {}
    total_inserted = 0
    total_processed = 0
    has_errors = False

    # 5 & 6. Validate & Ingest
    for file_path in files_to_process:
        result = ingest_file(file_path, engine=engine, archive_on_success=archive_on_success)
        file_summaries[result["filename"]] = result
        total_inserted += result.get("rows_inserted", 0)
        total_processed += result.get("rows_processed", 0)
        if result.get("status") in ["FAILED", "VALIDATION_FAILED"]:
            has_errors = True

    total_time = round(time.time() - pipeline_start, 2)
    pipeline_status = "FAILED" if has_errors else "SUCCESS"

    return {
        "status": pipeline_status,
        "total_inserted": total_inserted,
        "total_processed": total_processed,
        "total_time": total_time,
        "file_summaries": file_summaries
    }


def print_summary(results: dict):
    """
    Format and print final ingestion summary to standard output.
    """
    print("\n========================================")
    print("SALES DATA INGESTION SUMMARY")
    print("========================================\n")

    summaries = results.get("file_summaries", {})
    expected_files = ["train.csv", "store.csv", "test.csv"]

    for fname in expected_files:
        print(f"{fname}")
        if fname in summaries:
            s = summaries[fname]
            print(f"Rows processed: {s.get('rows_processed', 0)}")
            print(f"Rows inserted: {s.get('rows_inserted', 0)}")
            print(f"Duplicates skipped: {s.get('duplicates_skipped', 0)}")
            print(f"Errors: {s.get('error_count', 0)}")
        else:
            print("Status: File not found in data/incoming/")
        print()

    print(f"Total rows inserted: {results.get('total_inserted', 0)}")
    print(f"Total time: {results.get('total_time', 0)} seconds")
    print(f"Status: {results.get('status', 'UNKNOWN')}")
    print("========================================\n")


if __name__ == "__main__":
    # Pass --archive flag if archiving desired
    archive_flag = "--archive" in sys.argv
    res = run_ingestion_pipeline(archive_on_success=archive_flag)
    print_summary(res)
    if res.get("status") == "FAILED":
        sys.exit(1)
