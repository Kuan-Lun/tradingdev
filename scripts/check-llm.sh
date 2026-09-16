#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 || ( "$1" != codex && "$1" != local ) ]]; then
    echo 'Usage: scripts/check-llm.sh codex|local [pytest options]' >&2
    echo 'Local example: scripts/check-llm.sh local --llm-model qwen3.8:27b' >&2
    exit 2
fi
provider=$1
shift
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m pytest \
    tests/e2e/test_llm_workflows.py --llm-provider "$provider" -m live_llm "$@"
