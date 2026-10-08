"""
Unit and Integration Tests for Stage 5 Automated Model Retraining Pipeline.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import MODELS_DIR
from src.pipeline.retraining import (
    compare_models,
    load_current_production_metrics,
    load_retraining_history,
    promote_candidate_model,
    run_automated_retraining,
    save_retraining_history_entry,
)
from src.forecasting.model_utils import load_model_metadata, load_model_artifact


class TestRetrainingPipeline:
    """Test suite for automated retraining, comparison logic, promotion safety, and audit logs."""

    @pytest.fixture(autouse=True)
    def setup_temp_dirs(self):
        """Setup temporary directory for retraining tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_models_dir = self.temp_dir / "models"
        self.temp_models_dir.mkdir(parents=True, exist_ok=True)
        self.temp_history_file = self.temp_models_dir / "retraining_history.json"
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compare_models_promoted_when_improvement_exceeds_threshold(self):
        """Case A: Candidate model RMSPE improves beyond threshold -> Promoted."""
        current_metrics = {"mae": 583.64, "rmse": 846.96, "rmspe": 12.29}
        candidate_metrics = {"mae": 500.00, "rmse": 750.00, "rmspe": 10.00}

        promoted, improvement_pct, reason = compare_models(
            current_metrics, candidate_metrics, min_improvement_pct=1.0
        )

        assert promoted is True
        assert improvement_pct > 1.0
        assert "improved by" in reason
        assert "meeting threshold" in reason

    def test_compare_models_rejected_when_improvement_below_threshold(self):
        """Case B: Candidate model RMSPE does not meet threshold -> Rejected, production preserved."""
        current_metrics = {"mae": 583.64, "rmse": 846.96, "rmspe": 12.29}
        candidate_metrics = {"mae": 580.00, "rmse": 845.00, "rmspe": 12.25}

        promoted, improvement_pct, reason = compare_models(
            current_metrics, candidate_metrics, min_improvement_pct=1.0
        )

        assert promoted is False
        assert improvement_pct < 1.0
        assert "did not meet" in reason

    def test_compare_models_no_existing_production_model(self):
        """Case C: No prior production metrics exist -> Promoted automatically."""
        current_metrics = {"mae": float("inf"), "rmse": float("inf"), "rmspe": float("inf")}
        candidate_metrics = {"mae": 583.64, "rmse": 846.96, "rmspe": 12.29}

        promoted, improvement_pct, reason = compare_models(
            current_metrics, candidate_metrics, min_improvement_pct=1.0
        )

        assert promoted is True
        assert improvement_pct == 100.0
        assert "No valid production model metrics found" in reason

    def test_retraining_history_logging(self):
        """Case D: Test loading and appending retraining history entries."""
        assert load_retraining_history(self.temp_history_file) == []

        entry1 = {
            "run_id": "retrain_test_1",
            "status": "SUCCESS",
            "decision": "REJECTED",
            "promoted": False,
        }
        save_retraining_history_entry(entry1, self.temp_history_file)

        history = load_retraining_history(self.temp_history_file)
        assert len(history) == 1
        assert history[0]["run_id"] == "retrain_test_1"
        assert history[0]["decision"] == "REJECTED"

        entry2 = {
            "run_id": "retrain_test_2",
            "status": "SUCCESS",
            "decision": "PROMOTED",
            "promoted": True,
        }
        save_retraining_history_entry(entry2, self.temp_history_file)

        history2 = load_retraining_history(self.temp_history_file)
        assert len(history2) == 2
        assert history2[1]["run_id"] == "retrain_test_2"

    def test_promote_candidate_model(self):
        """Case E: Test archiving previous model and promoting candidate with minor version increment."""
        # Create dummy existing production model files
        prod_model_path = self.temp_models_dir / "sales_forecasting_model.joblib"
        prod_meta_path = self.temp_models_dir / "model_metadata.json"

        existing_meta = {
            "model_version": "1.0.0",
            "model_metrics": {"rmspe": 12.29},
        }
        dummy_prod_artifact = {"pipeline": "dummy_prod", "model_version": "1.0.0"}

        # Write dummy initial production model
        from src.forecasting.model_utils import save_model_artifact, save_model_metadata
        save_model_artifact(dummy_prod_artifact, prod_model_path)
        save_model_metadata(existing_meta, prod_meta_path)

        candidate_artifact = {"pipeline": "dummy_candidate"}
        candidate_metadata = {"model_metrics": {"rmspe": 10.50}}

        new_version, active_path = promote_candidate_model(
            candidate_artifact, candidate_metadata, models_dir=self.temp_models_dir
        )

        assert new_version == "1.1.0"
        assert active_path == prod_model_path
        assert prod_model_path.exists()
        assert prod_meta_path.exists()

        # Check archive directory created and archived artifact exists
        archive_dir = self.temp_models_dir / "archive"
        assert archive_dir.exists()
        archived_files = list(archive_dir.glob("*.joblib"))
        assert len(archived_files) == 1
        assert "v1.0.0" in archived_files[0].name

        # Verify new production metadata updated to v1.1.0
        updated_meta = load_model_metadata(prod_meta_path)
        assert updated_meta["model_version"] == "1.1.0"

    def test_feature_contract_exclusion(self):
        """Case F: Verify feature contract excludes 'customers', 'sales', and 'date'."""
        prod_meta_path = MODELS_DIR / "model_metadata.json"
        if prod_meta_path.exists():
            metadata = load_model_metadata(prod_meta_path)
            excluded = metadata.get("excluded_features", [])
            features_used = metadata.get("features_used", [])

            assert "customers" in excluded
            assert "sales" in excluded
            assert "date" in excluded

            assert "customers" not in features_used
            assert "sales" not in features_used
            assert "date" not in features_used

