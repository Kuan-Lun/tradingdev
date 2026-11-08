#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Only presentation dependencies are installed, never the TradingDev runtime.
npm ci --prefix tooling/marp --ignore-scripts --no-audit --no-fund
node tooling/marp/install-browser.mjs
bash scripts/stage-presentations.sh
