"""
CLI Entry Point: Model Training Script.

Executes Stage 3 Forecasting Model Training, Evaluation, and Serialization:
Usage:
    python scripts/train_model.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.forecasting.train import train_forecasting_pipeline

def main():
    print("========================================")
    print("STAGE 3 - MODEL TRAINING & EVALUATION")
    print("========================================\n")

    res = train_forecasting_pipeline()

    if res.get("status") == "SUCCESS":
        meta = res["metadata"]
        b_metrics = res["baseline_metrics"]
        m_metrics = res["model_metrics"]

        print("========================================")
        print("STAGE 3 TRAINING SUMMARY REPORT")
        print("========================================\n")
        print(f"Model Selected: {meta['best_model']} (Version {meta['model_version']})")
        print(f"Training Data Range:   {meta['training_data_start']} → {meta['training_data_end']} ({meta['training_rows']:,} rows)")
        print(f"Validation Data Range: {meta['validation_data_start']} → {meta['validation_data_end']} ({meta['validation_rows']:,} rows)\n")
        print(f"Features Used ({len(meta['features_used'])}): {meta['features_used']}")
        print(f"Excluded Features: {meta['excluded_features']}\n")
        print("Baseline Metrics (Seasonal Lag-7):")
        print(f"  MAE:   {b_metrics['mae']:.2f}")
        print(f"  RMSE:  {b_metrics['rmse']:.2f}")
        print(f"  RMSPE: {b_metrics['rmspe']:.2f}%\n")
        print(f"ML Model Metrics ({meta['best_model']}):")
        print(f"  MAE:   {m_metrics['mae']:.2f}")
        print(f"  RMSE:  {m_metrics['rmse']:.2f}")
        print(f"  RMSPE: {m_metrics['rmspe']:.2f}%\n")
        print(f"Model Artifact Saved To: {res['artifact_path']}")
        print(f"Metadata Saved To:       {res['metadata_path']}")
        print("\nSTATUS: SUCCESS")
        print("========================================\n")
    else:
        print("STAGE 3 TRAINING FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
