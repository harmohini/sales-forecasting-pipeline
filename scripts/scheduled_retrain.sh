#!/usr/bin/env bash
#
# Scheduled Retraining Safe Wrapper Script.
# Executed by macOS launchd or triggered manually via scripts/test_scheduler.sh.
#

set -euo pipefail

# 1. Resolve project root directory dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 2. Ensure log directory exists
LOG_DIR="$PROJECT_ROOT/logs/scheduler"
mkdir -p "$LOG_DIR"

LOCK_FILE="$LOG_DIR/retrain.lock"
STDOUT_LOG="$LOG_DIR/retraining_stdout.log"
STDERR_LOG="$LOG_DIR/retraining_stderr.log"

# 3. Overlapping run protection via process locking
if [ -f "$LOCK_FILE" ]; then
    EXISTING_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "[$(date -Iseconds)] [WARNING] Retraining run already in progress (PID: $EXISTING_PID). Skipping scheduled run."
        exit 0
    else
        # Stale lock file cleanup
        rm -f "$LOCK_FILE"
    fi
fi

# Acquire lock
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

echo "========================================"
echo "[$(date -Iseconds)] STARTING SCHEDULED RETRAINING JOB"
echo "Project Root: $PROJECT_ROOT"
echo "PID: $$"
echo "========================================"

# 4. Verify Python virtual environment executable
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    echo "[$(date -Iseconds)] [ERROR] Virtual environment Python not found or not executable at: $VENV_PYTHON" >&2
    exit 1
fi

# 5. Verify retraining script
RETRAIN_SCRIPT="$PROJECT_ROOT/scripts/retrain_model.py"
if [ ! -f "$RETRAIN_SCRIPT" ]; then
    echo "[$(date -Iseconds)] [ERROR] Retraining script not found at: $RETRAIN_SCRIPT" >&2
    exit 1
fi

# 6. Execute retraining script cleanly
EXIT_CODE=0
"$VENV_PYTHON" "$RETRAIN_SCRIPT" || EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date -Iseconds)] SCHEDULED RETRAINING COMPLETED SUCCESSFULLY"
else
    echo "[$(date -Iseconds)] SCHEDULED RETRAINING FAILED (Exit Code: $EXIT_CODE)" >&2
fi
echo "========================================"

exit $EXIT_CODE
