#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
existing="$(git config --get core.hooksPath || true)"
case "$existing" in
    ''|scripts/hooks|.githooks) ;;
    *) echo "Refusing to replace existing hooksPath: $existing" >&2; exit 1 ;;
esac
if [[ -z "$existing" ]]; then
    shopt -s nullglob
    for hook in "$(git rev-parse --git-path hooks)"/*; do
        if [[ -f "$hook" && -x "$hook" && "$hook" != *.sample ]]; then
            echo "Refusing to disable existing hook: $hook" >&2
            exit 1
        fi
    done
fi
primary="$(scripts/detect-primary-branch.sh)"
git config --local core.hooksPath .githooks
git config --local "branch.$primary.mergeOptions" --no-ff
git config --local "branch.$primary.rebase" false
git config --local pull.rebase false
git config --local pull.ff only
printf 'Installed TradingDev hooks for %s.\n' "$primary"
