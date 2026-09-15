#!/usr/bin/env bash
# Recreate this repository's development environment from the committed lockfile.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || {
    printf 'rebuild-env: uv is required\n' >&2
    exit 1
}
[[ -f uv.lock ]] || {
    printf 'rebuild-env: restore the committed uv.lock before rebuilding\n' >&2
    exit 1
}

# Reject a stale lock before removing the existing environment. Rebuilding must
# not upgrade dependencies or target another project's environment/cache.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv lock --check --python 3.13
uv venv --clear --python 3.13 .venv
uv sync --locked --all-extras --python 3.13
