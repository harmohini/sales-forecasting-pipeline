#!/usr/bin/env bash
#
# Uninstallation Helper Script for Sales Forecasting Automated Retraining Scheduler (macOS launchd).
#

set -euo pipefail

LABEL="com.salesforecasting.retrain"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "========================================"
echo "UNINSTALLING AUTOMATED RETRAINING SCHEDULER"
echo "Service Label: $LABEL"
echo "========================================"

# 1. Unload launch agent if loaded
if launchctl list | grep -q "$LABEL"; then
    echo "Unloading launch agent from macOS launchd..."
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
else
    echo "Launch agent is not currently loaded."
fi

# 2. Remove plist file from LaunchAgents
if [ -f "$PLIST_DEST" ]; then
    echo "Removing plist configuration: $PLIST_DEST"
    rm -f "$PLIST_DEST"
else
    echo "Plist file does not exist at: $PLIST_DEST"
fi

echo "========================================"
echo "SCHEDULER UNINSTALLED SUCCESSFULLY"
echo "Project files, model artifacts, and retraining history remain untouched."
echo "========================================"
