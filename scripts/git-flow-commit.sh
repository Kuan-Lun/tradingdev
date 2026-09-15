#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
[[ $# -ge 1 ]] || {
    echo 'Usage: scripts/git-flow-commit.sh "type: message" [files...]' >&2
    exit 1
}
primary="$(scripts/detect-primary-branch.sh)"
current="$(git branch --show-current)"
[[ -n "$current" && "$current" != "$primary" ]] || {
    echo 'Create a task branch before committing.' >&2
    exit 1
}
message=$1
shift
if [[ $# -gt 0 ]]; then
    git diff --cached --quiet || {
        echo 'The index already contains changes; review and commit it separately.' >&2
        exit 1
    }
    git add -- "$@"
fi
if git diff --cached --quiet; then
    echo 'Nothing staged to commit.' >&2
    exit 1
fi
git diff --cached --check
git commit -m "$message"
