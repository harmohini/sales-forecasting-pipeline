#!/usr/bin/env bash
#
# Installation Helper Script for Sales Forecasting Automated Retraining Scheduler (macOS launchd).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "INSTALLING AUTOMATED RETRAINING SCHEDULER"
echo "Project Root: $PROJECT_ROOT"
echo "========================================"

# 1. Validate prerequisites
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment Python not found at: $VENV_PYTHON" >&2
    exit 1
fi

RETRAIN_SCRIPT="$PROJECT_ROOT/scripts/retrain_model.py"
if [ ! -f "$RETRAIN_SCRIPT" ]; then
    echo "[ERROR] Retraining script not found at: $RETRAIN_SCRIPT" >&2
    exit 1
fi

WRAPPER_SCRIPT="$PROJECT_ROOT/scripts/scheduled_retrain.sh"
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "[ERROR] Scheduler wrapper script not found at: $WRAPPER_SCRIPT" >&2
    exit 1
fi

chmod +x "$WRAPPER_SCRIPT"

PLIST_SRC="$PROJECT_ROOT/deployment/launchd/com.salesforecasting.retrain.plist"
if [ ! -f "$PLIST_SRC" ]; then
    echo "[ERROR] Plist template file not found at: $PLIST_SRC" >&2
    exit 1
fi

# 2. Ensure log directory exists
LOG_DIR="$PROJECT_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"

# 3. Create LaunchAgents directory if needed
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$LAUNCH_AGENTS_DIR"

PLIST_DEST="$LAUNCH_AGENTS_DIR/com.salesforecasting.retrain.plist"

# 4. Generate plist with project root substitution
echo "Generating launchd plist at: $PLIST_DEST"
sed "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" "$PLIST_SRC" > "$PLIST_DEST"

# 5. Reload launch agent safely
echo "Unloading any existing launch agent instance..."
launchctl unload "$PLIST_DEST" 2>/dev/null || true

echo "Loading launch agent into macOS launchd..."
launchctl load "$PLIST_DEST"

# 6. Verify installation
echo "Checking launchd service status..."
if launchctl list | grep -q "com.salesforecasting.retrain"; then
    echo "========================================"
    echo "SCHEDULER INSTALLED SUCCESSFULLY!"
    echo "Service Label: com.salesforecasting.retrain"
    echo "Schedule:      Daily at 02:00 local time"
    echo "Plist Path:    $PLIST_DEST"
    echo "Logs Directory: $LOG_DIR"
    echo "========================================"
    launchctl list | grep "com.salesforecasting.retrain" || true
else
    echo "[ERROR] Failed to verify launchd service 'com.salesforecasting.retrain'." >&2
    exit 1
fi
