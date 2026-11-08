#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Compile the tracked sources, then stage only the declared delivery entries.
source_dir=docs/talks/tradingdev-introduction
target_dir=site-dist/tradingdev
for entry in slides.md theme.css assets evidence; do
    [[ -e "$source_dir/$entry" && ! -L "$source_dir/$entry" ]] || {
        echo "Missing or linked presentation entry: $entry" >&2
        exit 1
    }
done
[[ -z "$(find "$source_dir/assets" "$source_dir/evidence" -type l -print -quit)" ]] || {
    echo 'Presentation delivery cannot contain symlinks.' >&2
    exit 1
}
[[ ! -L site-dist && ! -L "$target_dir" ]] || {
    echo 'Presentation output cannot be a symlink.' >&2
    exit 1
}
mkdir -p site-dist
stage_dir="$(mktemp -d site-dist/.tradingdev.XXXXXX)"
trap 'rm -rf -- "$stage_dir"' EXIT
node tooling/marp/build.mjs "$source_dir/slides.md" "$source_dir/theme.css" "$stage_dir/index.html"
cp -R "$source_dir/assets" "$source_dir/evidence" "$stage_dir/"
"${TRADINGDEV_CHECK_PYTHON:-python3}" - "$stage_dir" <<'PY'
import sys
from pathlib import Path
from tooling.presentation_bundle import check_bundle
from tooling.presentation_html import check_html

root = Path(sys.argv[1])
check_html((root / "index.html").read_text(encoding="utf-8"), root=root)
check_bundle(root)
PY
# This fixed, generated directory is replaced only after every copy succeeded.
rm -rf -- "$target_dir"
mv "$stage_dir" "$target_dir"
echo "Presentation delivery staged at $target_dir"
