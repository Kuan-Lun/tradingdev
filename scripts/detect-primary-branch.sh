#!/usr/bin/env bash
set -euo pipefail

primary="$(git config --get tradingdev.primaryBranch || true)"
primary="${primary#refs/heads/}"
if [[ -z "$primary" ]]; then
    remote_head="$(git symbolic-ref --quiet refs/remotes/origin/HEAD || true)"
    primary="${remote_head#refs/remotes/origin/}"
fi
if [[ -z "$primary" ]]; then
    for candidate in main master; do
        if git show-ref --verify --quiet "refs/heads/$candidate" ||
            git show-ref --verify --quiet "refs/remotes/origin/$candidate"; then
            [[ -z "$primary" ]] || {
                echo 'Set tradingdev.primaryBranch: both main and master exist.' >&2
                exit 1
            }
            primary=$candidate
        fi
    done
fi
[[ -n "$primary" ]] && git check-ref-format "refs/heads/$primary" || {
    echo 'Cannot find primary; set git config tradingdev.primaryBranch <branch>.' >&2
    exit 1
}
printf '%s\n' "$primary"
