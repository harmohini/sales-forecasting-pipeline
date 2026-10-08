"""
Model Training Component.

Responsible for:
- Chronological train/validation splitting
- Feature selection & strict exclusion of future/unavailable columns ('customers')
- Preprocessing categorical features strictly on training data
- Seasonal Lag-7 baseline evaluation
- HistGradientBoostingRegressor model fitting & evaluation
- Serialization of model pipeline artifacts and metadata
"""

import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from config.settings import PROCESSED_DATA_DIR, MODEL_PATH, BASE_DIR
from src.ingestion.logger import setup_logger
from src.forecasting.evaluate import evaluate_model, print_evaluation_report
from src.forecasting.model_utils import save_model_artifact, save_model_metadata

logger = setup_logger("forecasting_trainer")

DEFAULT_PARQUET_PATH = PROCESSED_DATA_DIR / "sales_features.parquet"

def train_forecasting_pipeline(parquet_path: Path = None, val_days: int = 42, random_state: int = 42) -> dict:
    """
    Execute end-to-end forecasting model training and evaluation workflow.
    """
    start_time_iso = datetime.now().isoformat()
    t0 = time.time()

    if parquet_path is None:
        parquet_path = DEFAULT_PARQUET_PATH

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {parquet_path}. Run preprocessing first.")

    logger.info(f"Loading processed feature dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 1. Sort chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Chronological Train/Validation Split
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=val_days)

    train_df = df[df['date'] <= split_date].copy()
    val_df = df[df['date'] > split_date].copy()

    logger.info(
        f"Chronological split (val_days={val_days}):\n"
        f"  Train: {train_df['date'].min().strftime('%Y-%m-%d')} -> {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df):,} rows)\n"
        f"  Val:   {val_df['date'].min().strftime('%Y-%m-%d')} -> {val_df['date'].max().strftime('%Y-%m-%d')} ({len(val_df):,} rows)"
    )

    # 3. Target and Feature Selection
    target_col = "sales"
    # EXCLUDED: 'customers' is unavailable at production prediction time
    excluded_features = ["date", "sales", "customers"]
    feature_cols = [c for c in df.columns if c not in excluded_features]

    categorical_cols = ["state_holiday", "store_type", "assortment", "promo_interval"]
    # Ensure categorical columns exist in features
    categorical_cols = [c for c in categorical_cols if c in feature_cols]

    logger.info(f"Target: '{target_col}' | Excluded features: {excluded_features}")
    logger.info(f"Feature count: {len(feature_cols)} | Categorical features: {categorical_cols}")

    # Separate X and y
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].values

    # 4. Fit Categorical Encoder strictly on Training Data
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = encoder.transform(X_val[categorical_cols])

    # 5. Baseline Model Evaluation (Seasonal Lag-7)
    if "sales_lag_7" not in X_val.columns:
        raise ValueError("Required feature 'sales_lag_7' missing for baseline evaluation.")
        
    baseline_preds = X_val["sales_lag_7"].values
    baseline_metrics = evaluate_model(y_val, baseline_preds)

    # 6. Train Machine Learning Model (HistGradientBoostingRegressor)
    logger.info("Training HistGradientBoostingRegressor model...")
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.1,
        max_depth=12,
        min_samples_leaf=20,
        random_state=random_state
    )
    
    t_fit_start = time.time()
    model.fit(X_train, y_train)
    fit_duration = round(time.time() - t_fit_start, 2)
    logger.info(f"Model fitting completed in {fit_duration}s.")

    # 7. Evaluate ML Model on Validation Set
    raw_val_preds = model.predict(X_val)
    val_preds = np.maximum(0.0, raw_val_preds)  # Sales cannot be negative
    model_metrics = evaluate_model(y_val, val_preds)

    print_evaluation_report(baseline_metrics, model_metrics)

    # 8. Save Model Artifact & Metadata
    artifact = {
        "model": model,
        "encoder": encoder,
        "feature_names": feature_cols,
        "categorical_cols": categorical_cols,
        "excluded_features": excluded_features,
        "model_version": "1.0.0",
        "training_completed_at": datetime.now().isoformat()
    }
    artifact_path = save_model_artifact(artifact)

    end_time_iso = datetime.now().isoformat()
    metadata = {
        "model_type": "HistGradientBoostingRegressor",
        "model_version": "1.0.0",
        "training_started_at": start_time_iso,
        "training_completed_at": end_time_iso,
        "training_data_start": train_df['date'].min().strftime("%Y-%m-%d"),
        "training_data_end": train_df['date'].max().strftime("%Y-%m-%d"),
        "validation_data_start": val_df['date'].min().strftime("%Y-%m-%d"),
        "validation_data_end": val_df['date'].max().strftime("%Y-%m-%d"),
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "features_used": feature_cols,
        "excluded_features": excluded_features,
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "best_model": "HistGradientBoostingRegressor",
        "random_state": random_state,
        "fit_duration_seconds": fit_duration,
        "total_duration_seconds": round(time.time() - t0, 2)
    }
    metadata_path = save_model_metadata(metadata)

    logger.info(f"Saved trained model artifact to: {artifact_path}")
    logger.info(f"Saved model metadata to: {metadata_path}")

    return {
        "status": "SUCCESS",
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "metadata": metadata
    }
