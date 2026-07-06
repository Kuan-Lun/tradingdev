#!/usr/bin/env bash
# Stop-hook implementation shared by Claude and Codex.
# Mirrors VS Code's on-save pipeline:
#   1. Black                 — pre-format pass; also catches syntax errors.
#   2. Ruff `--fix`          — auto-fixes safe lints (imports, pyupgrade, …).
#   3. Black                 — second pass: re-format whatever ruff rewrote
#                              so the file is guaranteed black-stable, even
#                              when ruff's UP rules produce code that black
#                              would reformat (single-pipeline convergence).
#   4. Mypy                  — type check on the final formatted code.
#
# Why a Stop hook (not PostToolUse): incremental edits routinely produce
# transient broken states (e.g. add import, then add usage in next edit).
# Running checkers between every edit would block legitimate workflows.
# At Stop time the codebase is supposed to be coherent, so a full check is
# the right gate.
#
# Error handling: any tool failing causes the script to exit 2, which makes
# the caller surface the captured stderr to the model on the next turn.
# stderr from a successful tool is intentionally ignored, so the noisy
# "All done! ✨ 🍰 ✨" lines from black do not pollute the transcript.

set -eu
trap 'exit 2' ERR

PATHS=(src tests scripts)

# --all-extras: a bare `uv run` re-syncs to the base lock and silently drops
# optional extras (e.g. dashboard's streamlit/plotly). Since src/tests/scripts
# includes dashboard code, dropping the extra makes mypy.ini's
# ignore_missing_imports treat those modules as Any, masking real type errors
# in decorators and return types that rely on their real stubs.
uv run --all-extras black "${PATHS[@]}" >&2
uv run --all-extras ruff check --fix "${PATHS[@]}" >&2
uv run --all-extras black "${PATHS[@]}" >&2
uv run --all-extras mypy "${PATHS[@]}" >&2
