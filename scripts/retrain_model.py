"""
CLI Entry Point: Automated Model Retraining Script.

Executes Stage 5 Automated Model Retraining & Safe Promotion Pipeline:
Usage:
    python scripts/retrain_model.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.retraining import run_automated_retraining
from config.settings import MIN_IMPROVEMENT_PERCENT

def main():
    start_ts = time.strftime('%Y%m%d_%H%M%S')
    run_id = f"retrain_{start_ts}"

    print("\n========================================")
    print("AUTOMATED MODEL RETRAINING")
    print("========================================\n")
    print(f"Run ID: {run_id}\n")

    print("[1/7] Checking data availability...")
    print("[2/7] Preparing feature dataset...")
    print("[3/7] Training candidate model...")
    print("[4/7] Evaluating candidate model...")
    print("[5/7] Comparing candidate vs production model...")
    print("[6/7] Executing promotion decision...")
    print("[7/7] Saving retraining metadata & history audit log...\n")

    res = run_automated_retraining(min_improvement_pct=MIN_IMPROVEMENT_PERCENT)

    if res.get("status") == "SUCCESS":
        curr_m = res.get("current_metrics", {})
        cand_m = res.get("candidate_metrics", {})
        
        curr_rmspe = f"{curr_m.get('rmspe', 0.0):.2f}%" if curr_m.get('rmspe') != float('inf') else "N/A"
        cand_rmspe = f"{cand_m.get('rmspe', 0.0):.2f}%"

        print("========================================")
        print("RETRAINING SUMMARY REPORT")
        print("========================================\n")
        print("Current Model:")
        print(f"  Version: {res.get('previous_version')}")
        print(f"  RMSPE:   {curr_rmspe}\n")

        print("Candidate Model:")
        print(f"  RMSPE:   {cand_rmspe}\n")

        print(f"Improvement: {res.get('improvement_pct', 0.0):.2f}% (Threshold: {MIN_IMPROVEMENT_PERCENT:.2f}%)\n")

        decision = res.get('decision', 'REJECTED')
        print(f"Decision: {decision}")
        print(f"Reason:   {res.get('decision_reason')}\n")
        print(f"Active Production Model Version: {res.get('active_version')}")
        print(f"Active Production Model Path:    {res.get('active_model_path')}")
        print("\n========================================")
        print("RETRAINING COMPLETE")
        print("========================================\n")
    else:
        print("\n========================================")
        print("RETRAINING FAILED")
        print(f"Error: {res.get('error')}")
        print("========================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
