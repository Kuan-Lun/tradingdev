#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Stage only the declared delivery entries; no Python or Marp is needed in CI.
source_dir=docs/talks/tradingdev-introduction
target_dir=site-dist/tradingdev
for entry in index.html assets evidence; do
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
cp "$source_dir/index.html" "$stage_dir/index.html"
cp -R "$source_dir/assets" "$source_dir/evidence" "$stage_dir/"
# This fixed, generated directory is replaced only after every copy succeeded.
rm -rf -- "$target_dir"
mv "$stage_dir" "$target_dir"
echo "Presentation delivery staged at $target_dir"
