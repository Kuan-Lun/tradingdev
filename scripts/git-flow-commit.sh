#!/usr/bin/env bash
# Atomic git flow: create feature branch → commit → merge --no-ff into main → delete branch.
#
# Usage:
#   scripts/git-flow-commit.sh <branch-suffix> <commit-msg> [files...]
#
# If files are provided they are staged before committing.
# If no files are provided, currently staged changes are used.
#
# Example:
#   scripts/git-flow-commit.sh "fix-typo" "$(cat <<'EOF'
#   fix: correct typo in README
#
#   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
#   EOF
#   )" README.md

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <branch-suffix> <commit-msg> [files...]" >&2
    exit 1
fi

BRANCH="feature/$1"
MSG="$2"
shift 2

# Ensure we start from main
CURRENT=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ "$CURRENT" != "main" ]; then
    echo "ERROR: must be run from main (currently on '$CURRENT')." >&2
    exit 1
fi

# Ensure working tree is clean except for staged/specified files
if [ $# -gt 0 ]; then
    git add "$@"
fi

if ! git diff --cached --quiet; then
    : # staged changes exist — proceed
else
    echo "ERROR: nothing staged to commit." >&2
    exit 1
fi

git checkout -b "$BRANCH"
git commit -m "$MSG"
git checkout main
git merge --no-ff "$BRANCH" -m "Merge branch '$BRANCH'"
git branch -d "$BRANCH"

echo ""
echo "Done: '$BRANCH' merged into main and deleted."
