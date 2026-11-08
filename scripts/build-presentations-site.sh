#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Only presentation dependencies are installed, never the TradingDev runtime.
npm ci --prefix tooling/marp --include=optional --ignore-scripts --no-audit --no-fund
bash scripts/stage-presentations.sh
