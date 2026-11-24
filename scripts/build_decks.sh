#!/usr/bin/env bash
#
# Export both Marimo notebooks as self-contained WebAssembly slide decks.
#
#   ./scripts/build_decks.sh
#   python3 -m http.server --directory build 8000
#
# The script uses the project environment created by `uv sync`. Run that once
# before the first build.

set -euo pipefail

cd "$(dirname "$0")/.."

UV="${UV:-uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        UV="$HOME/.local/bin/uv"
    else
        echo "error: uv not found. Install it from https://docs.astral.sh/uv/" >&2
        exit 1
    fi
fi
OUT="build"

rm -rf "$OUT"
mkdir -p "$OUT/00" "$OUT/01"

echo "→ Lecture 1: Introduction"
"$UV" run marimo export html-wasm notebooks/00/intro.py \
    -o "$OUT/00/intro" --mode run --no-show-code
# marimo exports its own assets but not the notebook's media directory, so the
# figures have to sit next to the generated index.html.
cp -R notebooks/00/media "$OUT/00/intro/"

echo "→ Lecture 2: Neural networks"
"$UV" run marimo export html-wasm notebooks/01/neural_networks.py \
    -o "$OUT/01/neural_networks" --mode run --no-show-code

cp index.html "$OUT/index.html"

echo
echo "Decks written to $OUT/."
echo "Serve them with:  python3 -m http.server --directory $OUT 8000"
