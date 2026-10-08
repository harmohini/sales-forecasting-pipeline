"""
Automated Model Retraining & Safe Promotion Pipeline.

Responsible for:
- Checking data availability and feature contract integrity
- Training and evaluating candidate forecasting models
- Comparing candidate performance against active production model metrics
- Safe atomic model promotion with archived version backups
- Comprehensive audit trail logging in models/retraining_history.json
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import (
    BASE_DIR, MODELS_DIR, ARCHIVE_MODELS_DIR, RETRAINING_HISTORY_PATH,
    MIN_IMPROVEMENT_PERCENT, PROCESSED_DATA_DIR
)
from src.ingestion.logger import setup_logger
from src.forecasting.train import train_forecasting_pipeline
from src.forecasting.model_utils import (
    load_model_metadata, load_model_artifact, save_model_artifact, save_model_metadata
)
from src.forecasting.predict import generate_forecast

logger = setup_logger("retraining_pipeline")


def load_retraining_history(filepath: Path = None) -> list:
    """
    Load retraining run history entries from JSON file.
    """
    if filepath is None:
        filepath = RETRAINING_HISTORY_PATH
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading retraining history: {str(e)}")
        return []


def save_retraining_history_entry(entry: dict, filepath: Path = None) -> None:
    """
    Append a new retraining run entry to models/retraining_history.json.
    """
    if filepath is None:
        filepath = RETRAINING_HISTORY_PATH
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    history = load_retraining_history(filepath)
    history.append(entry)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Appended retraining run '{entry.get('run_id')}' to {filepath}")


def load_current_production_metrics(metadata_path: Path = None) -> tuple[dict, str]:
    """
    Load current production model metrics and version from metadata.
    """
    if metadata_path is None:
        metadata_path = MODELS_DIR / "model_metadata.json"
    
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        # Fallback if no production model exists yet
        return {"mae": float("inf"), "rmse": float("inf"), "rmspe": float("inf")}, "0.0.0"

    try:
        meta = load_model_metadata(metadata_path)
        metrics = meta.get("model_metrics", {"mae": float("inf"), "rmse": float("inf"), "rmspe": float("inf")})
        version = meta.get("model_version", "1.0.0")
        return metrics, version
    except Exception as e:
        logger.error(f"Error loading current model metrics: {str(e)}")
        return {"mae": float("inf"), "rmse": float("inf"), "rmspe": float("inf")}, "0.0.0"


def compare_models(current_metrics: dict, candidate_metrics: dict, min_improvement_pct: float = MIN_IMPROVEMENT_PERCENT) -> tuple[bool, float, str]:
    """
    Compare candidate model metrics against active production model metrics.
    Returns (promoted: bool, improvement_pct: float, reason: str).
    """
    curr_rmspe = current_metrics.get("rmspe", float("inf"))
    cand_rmspe = candidate_metrics.get("rmspe", float("inf"))

    if np.isinf(curr_rmspe) or curr_rmspe <= 0:
        return True, 100.0, "No valid production model metrics found. Promoting candidate."

    # Calculate relative percentage improvement: ((curr - cand) / curr) * 100
    improvement_pct = ((curr_rmspe - cand_rmspe) / curr_rmspe) * 100.0
    improvement_pct = round(improvement_pct, 2)

    if improvement_pct >= min_improvement_pct:
        reason = (
            f"Candidate RMSPE ({cand_rmspe:.2f}%) improved by {improvement_pct:.2f}% "
            f"over current production model ({curr_rmspe:.2f}%), meeting threshold ({min_improvement_pct:.1f}%)."
        )
        return True, improvement_pct, reason
    else:
        reason = (
            f"Candidate RMSPE ({cand_rmspe:.2f}%) did not meet {min_improvement_pct:.1f}% "
            f"improvement threshold over production model ({curr_rmspe:.2f}%). Improvement was {improvement_pct:.2f}%."
        )
        return False, improvement_pct, reason


def promote_candidate_model(candidate_artifact: dict, candidate_metadata: dict, models_dir: Path = None) -> tuple[str, Path]:
    """
    Safely archive current production model and promote candidate model artifact to production.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir = Path(models_dir)
    archive_dir = models_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    prod_model_path = models_dir / "sales_forecasting_model.joblib"
    prod_meta_path = models_dir / "model_metadata.json"

    # Step 1: Archive current production model if it exists
    curr_version = "1.0.0"
    if prod_meta_path.exists():
        try:
            curr_meta = load_model_metadata(prod_meta_path)
            curr_version = curr_meta.get("model_version", "1.0.0")
        except Exception:
            pass

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prod_model_path.exists():
        archived_name = f"sales_forecasting_model_v{curr_version}_{timestamp_str}.joblib"
        archived_path = archive_dir / archived_name
        shutil.copy2(prod_model_path, archived_path)
        logger.info(f"Archived current production model v{curr_version} to: {archived_path}")

    # Step 2: Determine next semantic version number
    try:
        parts = [int(p) for p in curr_version.split(".")]
        parts[1] += 1  # Increment minor version
        new_version = f"{parts[0]}.{parts[1]}.{parts[2]}"
    except Exception:
        new_version = "1.1.0"

    # Step 3: Update version tag in candidate artifact & metadata
    candidate_artifact["model_version"] = new_version
    candidate_metadata["model_version"] = new_version

    # Step 4: Atomically save new candidate artifact & metadata as production
    save_model_artifact(candidate_artifact, prod_model_path)
    save_model_metadata(candidate_metadata, prod_meta_path)

    logger.info(f"Successfully promoted candidate model to production version v{new_version}")
    return new_version, prod_model_path


def run_automated_retraining(
    min_improvement_pct: float = MIN_IMPROVEMENT_PERCENT,
    force_promote: bool = False,
    parquet_path: Path = None,
    history_filepath: Path = None
) -> dict:
    """
    Execute end-to-end automated retraining workflow:
    1. Check data availability
    2. Load current model metrics
    3. Train candidate model
    4. Validate candidate predictions
    5. Compare candidate vs current model
    6. Execute promotion decision
    7. Save retraining metadata and history audit log
    """
    start_time_iso = datetime.now().isoformat()
    t_start = time.time()
    run_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("========================================")
    logger.info(f"AUTOMATED MODEL RETRAINING RUN: {run_id}")
    logger.info("========================================")

    # 1. Check data availability
    if parquet_path is None:
        parquet_path = PROCESSED_DATA_DIR / "sales_features.parquet"
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        logger.error(f"Retraining aborted: Processed dataset not found at {parquet_path}")
        return {"status": "FAILED", "error": f"Processed dataset missing: {parquet_path}"}

    # 2. Load current production model metrics & version
    current_metrics, current_version = load_current_production_metrics()
    logger.info(f"Current Production Model: v{current_version} (RMSPE: {current_metrics.get('rmspe', 'N/A')}%)")

    # 3. Train Candidate Model
    logger.info("Training candidate forecasting model...")
    try:
        # Execute training pipeline
        train_result = train_forecasting_pipeline(parquet_path=parquet_path)
        candidate_metadata = train_result["metadata"]
        candidate_metrics = train_result["model_metrics"]
    except Exception as e:
        logger.error(f"Candidate model training failed: {str(e)}", exc_info=True)
        # Log failure entry to history
        failure_entry = {
            "run_id": run_id,
            "timestamp": start_time_iso,
            "status": "FAILED",
            "error": str(e),
            "previous_model_version": current_version,
            "candidate_model_version": "N/A",
            "promoted": False,
            "promotion_reason": f"Training failed: {str(e)}"
        }
        save_retraining_history_entry(failure_entry, history_filepath)
        return {"status": "FAILED", "error": str(e), "current_version": current_version}

    # 4. Validate candidate model predictions (Safety Check)
    logger.info("Validating candidate model predictions...")
    try:
        df_sample = pd.read_parquet(parquet_path).head(100)
        # Verify candidate artifact can load and generate valid predictions
        cand_artifact = load_model_artifact(MODELS_DIR / "sales_forecasting_model.joblib")

        sample_preds = generate_forecast(df_sample, artifact=cand_artifact)

        if np.isnan(sample_preds).any() or np.isinf(sample_preds).any():
            raise ValueError("Candidate model generated NaN or Inf prediction values.")
    except Exception as e:
        logger.error(f"Candidate validation failed: {str(e)}", exc_info=True)
        failure_entry = {
            "run_id": run_id,
            "timestamp": start_time_iso,
            "status": "FAILED_VALIDATION",
            "error": str(e),
            "previous_model_version": current_version,
            "candidate_model_version": "N/A",
            "promoted": False,
            "promotion_reason": f"Candidate prediction validation failed: {str(e)}"
        }
        save_retraining_history_entry(failure_entry, history_filepath)
        return {"status": "FAILED_VALIDATION", "error": str(e), "current_version": current_version}

    # 5. Compare Candidate vs Production Model
    promoted, improvement_pct, decision_reason = compare_models(
        current_metrics, candidate_metrics, min_improvement_pct=min_improvement_pct
    )

    if force_promote and not promoted:
        promoted = True
        decision_reason += " (Overridden via force_promote flag)."

    # 6. Execute Promotion Decision
    if promoted:
        new_version, active_path = promote_candidate_model(cand_artifact, candidate_metadata)
        decision_status = "PROMOTED"
    else:
        new_version = current_version
        decision_status = "REJECTED"
        active_path = MODELS_DIR / "sales_forecasting_model.joblib"
        logger.info(f"Candidate rejected. Preserving current production model v{current_version}.")

    duration = round(time.time() - t_start, 2)

    # 7. Record Retraining History Entry
    history_entry = {
        "run_id": run_id,
        "timestamp": start_time_iso,
        "completed_at": datetime.now().isoformat(),
        "status": "SUCCESS",
        "decision": decision_status,
        "promoted": promoted,
        "promotion_reason": decision_reason,
        "previous_model_version": current_version,
        "active_model_version": new_version,
        "dataset_path": str(parquet_path),
        "training_rows": candidate_metadata.get("training_rows", 0),
        "validation_rows": candidate_metadata.get("validation_rows", 0),
        "training_start_date": candidate_metadata.get("training_data_start"),
        "training_end_date": candidate_metadata.get("training_data_end"),
        "validation_start_date": candidate_metadata.get("validation_data_start"),
        "validation_end_date": candidate_metadata.get("validation_data_end"),
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "improvement_percent": improvement_pct,
        "min_improvement_threshold_pct": min_improvement_pct,
        "duration_seconds": duration
    }
    save_retraining_history_entry(history_entry, history_filepath)

    return {
        "status": "SUCCESS",
        "run_id": run_id,
        "decision": decision_status,
        "promoted": promoted,
        "previous_version": current_version,
        "active_version": new_version,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "improvement_pct": improvement_pct,
        "decision_reason": decision_reason,
        "active_model_path": str(active_path),
        "duration": duration,
        "history_entry": history_entry
    }
