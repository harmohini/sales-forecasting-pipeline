#!/usr/bin/env bash
#
# Manual Scheduler Testing Script.
# Immediately triggers the scheduled retraining wrapper script from project root without waiting for 02:00.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "MANUAL SCHEDULER TEST TRIGGER"
echo "Project Root: $PROJECT_ROOT"
echo "========================================"

WRAPPER_SCRIPT="$PROJECT_ROOT/scripts/scheduled_retrain.sh"

if [ ! -x "$WRAPPER_SCRIPT" ]; then
    chmod +x "$WRAPPER_SCRIPT"
fi

echo "Executing wrapper script: $WRAPPER_SCRIPT"
echo ""

EXIT_CODE=0
"$WRAPPER_SCRIPT" || EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "MANUAL SCHEDULER TEST PASSED"
else
    echo "MANUAL SCHEDULER TEST FAILED (Exit Code: $EXIT_CODE)" >&2
fi
echo "========================================"

exit $EXIT_CODE
