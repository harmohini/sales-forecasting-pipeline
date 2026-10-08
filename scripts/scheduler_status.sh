#!/usr/bin/env bash
#
# Scheduler Status Reporting Script.
# Displays launchd status, scheduler log output, and latest retraining run details.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LABEL="com.salesforecasting.retrain"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$PROJECT_ROOT/logs/scheduler"
STDOUT_LOG="$LOG_DIR/retraining_stdout.log"
STDERR_LOG="$LOG_DIR/retraining_stderr.log"
HISTORY_FILE="$PROJECT_ROOT/models/retraining_history.json"

echo "========================================"
echo "AUTOMATED RETRAINING SCHEDULER STATUS"
echo "========================================"
echo "Label:      $LABEL"
echo "Plist Path: $PLIST_DEST"
echo ""

# 1. Check LaunchAgent installation status
if [ -f "$PLIST_DEST" ]; then
    echo "[✓] LaunchAgent plist file exists."
else
    echo "[✗] LaunchAgent plist file NOT found at: $PLIST_DEST"
fi

# 2. Check launchctl service state
if launchctl list | grep -q "$LABEL"; then
    echo "[✓] LaunchAgent service is LOADED in launchd."
    echo ""
    echo "launchctl info:"
    launchctl list | grep "$LABEL" || true
else
    echo "[✗] LaunchAgent service is NOT loaded in launchd."
fi

echo ""
echo "========================================"
echo "RECENT SCHEDULER LOG OUTPUT"
echo "========================================"

if [ -f "$STDOUT_LOG" ]; then
    echo "--- Stdout Log ($STDOUT_LOG) ---"
    tail -n 15 "$STDOUT_LOG"
else
    echo "Stdout log not found at: $STDOUT_LOG"
fi

echo ""
if [ -f "$STDERR_LOG" ]; then
    echo "--- Stderr Log ($STDERR_LOG) ---"
    tail -n 15 "$STDERR_LOG"
else
    echo "Stderr log not found at: $STDERR_LOG"
fi

echo ""
echo "========================================"
echo "LATEST RETRAINING HISTORY ENTRY"
echo "========================================"

if [ -f "$HISTORY_FILE" ]; then
    "$PROJECT_ROOT/.venv/bin/python" -c "
import json, sys
try:
    with open('$HISTORY_FILE') as f:
        data = json.load(f)
        if data:
            last = data[-1]
            print(f'Run ID:       {last.get(\"run_id\")}')
            print(f'Timestamp:    {last.get(\"timestamp\")}')
            print(f'Decision:     {last.get(\"decision\")}')
            print(f'Active Ver:   {last.get(\"active_model_version\")}')
            print(f'Reason:       {last.get(\"promotion_reason\")}')
        else:
            print('History file is empty.')
except Exception as e:
    print(f'Error reading history: {e}')
"
else
    echo "Retraining history file not found at: $HISTORY_FILE"
fi

echo "========================================"
