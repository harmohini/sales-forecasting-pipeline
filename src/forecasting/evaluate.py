"""
Model Evaluation Component.

Responsible for computing forecasting metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Root Mean Squared Percentage Error (RMSPE)
"""

import numpy as np

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute performance evaluation metrics for sales predictions.
    RMSPE is handled safely by clipping zero/near-zero denominators to minimum 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE & RMSE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Zero-safe RMSPE calculation
    # Denominator clipped at 1 to prevent division by zero or extreme distortion on zero/near-zero actuals
    safe_denominator = np.maximum(y_true, 1.0)
    pct_errors = (y_true - y_pred) / safe_denominator
    rmspe = float(np.sqrt(np.mean(np.square(pct_errors))) * 100.0)

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "rmspe": round(rmspe, 2)
    }

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate predicted sales against true sales.
    """
    return calculate_metrics(y_true, y_pred)

def print_evaluation_report(baseline_metrics: dict, model_metrics: dict) -> None:
    """
    Print formatted evaluation summary table comparing baseline vs ML model.
    """
    print("\n========================================")
    print("FORECAST MODEL EVALUATION")
    print("========================================\n")
    print("Baseline - Seasonal Lag 7")
    print(f"MAE:   {baseline_metrics['mae']:.2f}")
    print(f"RMSE:  {baseline_metrics['rmse']:.2f}")
    print(f"RMSPE: {baseline_metrics['rmspe']:.2f}%\n")
    print("ML Forecasting Model (HistGBR)")
    print(f"MAE:   {model_metrics['mae']:.2f}")
    print(f"RMSE:  {model_metrics['rmse']:.2f}")
    print(f"RMSPE: {model_metrics['rmspe']:.2f}%\n")
    
    if model_metrics['rmspe'] < baseline_metrics['rmspe']:
        improvement = baseline_metrics['rmspe'] - model_metrics['rmspe']
        print(f"Result: ML Model OUTPERFORMED baseline by {improvement:.2f}% RMSPE reduction.")
    else:
        print("Result: ML Model did not outperform baseline.")
    print("========================================\n")
