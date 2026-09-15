#!/usr/bin/env bash
set -euo pipefail

fail() { echo "git-flow-merge: $1" >&2; exit 1; }
assert_clean() {
    [[ -z "$(git -C "$1" status --porcelain --untracked-files=all)" ]] \
        || fail "worktree is not clean: $1"
    for operation in MERGE_HEAD REBASE_HEAD CHERRY_PICK_HEAD REVERT_HEAD; do
        [[ ! -e "$(git -C "$1" rev-parse --path-format=absolute --git-path "$operation")" ]] \
            || fail "Git operation is active: $operation"
    done
}

task_worktree="$(git rev-parse --show-toplevel)"
cd "$task_worktree"
primary="$(scripts/detect-primary-branch.sh)"
task="$(git branch --show-current)"
[[ -n "$task" && "$task" != "$primary" ]] || fail 'run from a task branch'
assert_clean "$task_worktree"
git merge-base "$primary" "$task" >/dev/null || fail 'branches have no common ancestor'
[[ "$(git rev-list --count "$primary..$task")" -gt 0 ]] || fail 'no commits to merge'
[[ "$(git config --get core.hooksPath)" == .githooks ]] || fail 'install Git hooks first'

merge_worktree=''
candidate=''
while IFS= read -r line; do
    case "$line" in
        'worktree '*) candidate="${line#worktree }" ;;
        "branch refs/heads/$primary") merge_worktree=$candidate ;;
    esac
done < <(git worktree list --porcelain)
if [[ -n "$merge_worktree" ]]; then
    assert_clean "$merge_worktree"
else
    git switch "$primary"
    merge_worktree=$task_worktree
fi

if ! git -C "$merge_worktree" merge --no-ff --no-edit "$task"; then
    if [[ -f "$(git -C "$merge_worktree" rev-parse --path-format=absolute --git-path MERGE_HEAD)" ]]; then
        git -C "$merge_worktree" merge --abort || fail 'merge abort failed; inspect Git state'
    fi
    if [[ "$merge_worktree" == "$task_worktree" ]]; then
        git switch "$task"
    fi
    fail 'merge or checks failed; task branch retained'
fi

cd "$merge_worktree"
if [[ "$task_worktree" != "$merge_worktree" ]]; then
    assert_clean "$task_worktree"
    git worktree remove "$task_worktree"
fi
git branch -d "$task"
printf 'Merged %s into %s at %s.\n' "$task" "$primary" "$(git rev-parse --short HEAD)"
