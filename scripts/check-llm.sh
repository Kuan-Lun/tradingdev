#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || ( "$1" != codex && "$1" != local ) ]]; then
    echo 'Usage: scripts/check-llm.sh codex|local [pytest options]' >&2
    echo 'Codex example: scripts/check-llm.sh codex --llm-model gpt-5.6-luna' >&2
    echo 'Local example (replace MODEL_NAME with your installed model name):' >&2
    echo '  scripts/check-llm.sh local --llm-model MODEL_NAME --llm-timeout 900' >&2
    exit 2
fi
provider=$1
shift
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -u -m pytest \
    tests/e2e/test_llm_workflows.py --llm-provider "$provider" -m live_llm \
    -x -vv --capture=tee-sys --tb=short --durations=3 "$@"
