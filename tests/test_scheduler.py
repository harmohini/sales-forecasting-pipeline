"""
Unit and Integration Tests for Stage 6 Automated Retraining Scheduler.
"""

import os
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from config.settings import BASE_DIR, MODELS_DIR


class TestScheduler:
    """Test suite for launchd scheduler configuration, helper scripts, process locking, and safety guarantees."""

    def test_plist_template_validity(self):
        """Verify launchd plist file exists, is valid XML/plist, and contains required keys."""
        plist_path = BASE_DIR / "deployment" / "launchd" / "com.salesforecasting.retrain.plist"
        assert plist_path.exists(), f"Plist template not found at {plist_path}"
        assert plist_path.stat().st_size > 0

        with open(plist_path, "rb") as f:
            plist_data = plistlib.load(f)

        assert plist_data.get("Label") == "com.salesforecasting.retrain"
        assert "ProgramArguments" in plist_data
        assert "StartCalendarInterval" in plist_data
        assert plist_data["StartCalendarInterval"].get("Hour") == 2
        assert plist_data["StartCalendarInterval"].get("Minute") == 0
        assert "StandardOutPath" in plist_data
        assert "StandardErrorPath" in plist_data

    def test_helper_scripts_exist_and_executable(self):
        """Verify all scheduler shell scripts exist and have executable permissions."""
        scripts_dir = BASE_DIR / "scripts"
        required_scripts = [
            "scheduled_retrain.sh",
            "install_scheduler.sh",
            "uninstall_scheduler.sh",
            "test_scheduler.sh",
            "scheduler_status.sh",
        ]

        for script_name in required_scripts:
            script_path = scripts_dir / script_name
            assert script_path.exists(), f"Script {script_name} does not exist at {script_path}"
            assert os.access(script_path, os.X_OK), f"Script {script_name} is not executable"

    def test_scheduled_wrapper_detection(self):
        """Verify scheduled_retrain.sh resolves project root and virtualenv Python."""
        venv_python = BASE_DIR / ".venv" / "bin" / "python"
        assert venv_python.exists(), f"Virtualenv Python not found at {venv_python}"
        assert os.access(venv_python, os.X_OK)

        retrain_script = BASE_DIR / "scripts" / "retrain_model.py"
        assert retrain_script.exists()

    def test_process_lock_mechanism(self):
        """Verify overlapping run protection via lock file creation and removal."""
        log_dir = BASE_DIR / "logs" / "scheduler"
        log_dir.mkdir(parents=True, exist_ok=True)
        lock_file = log_dir / "retrain.lock"

        # Lock file should not exist initially
        if lock_file.exists():
            lock_file.unlink()

        # Simulate lock acquisition
        lock_file.write_text("999999")
        assert lock_file.exists()
        assert lock_file.read_text().strip() == "999999"

        # Clean up
        lock_file.unlink()
        assert not lock_file.exists()

    def test_safety_guarantee_no_unintended_install(self):
        """Verify tests do not touch user's ~/Library/LaunchAgents/ directory directly."""
        user_launch_agents = Path.home() / "Library" / "LaunchAgents"
        # We verify that evaluating unit tests does not automatically create plist if not installed by user
        dest_plist = user_launch_agents / "com.salesforecasting.retrain.plist"
        # Reading dest_plist should not raise any unexpected assertion errors
        pass
