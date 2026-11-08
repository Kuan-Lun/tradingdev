#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m tooling.environment tooling
scripts/check-fast.sh
exec "${TRADINGDEV_CHECK_PYTHON:-.venv/bin/python}" -m pytest tests/tooling \
    -n auto --dist worksteal --max-worker-restart=0
