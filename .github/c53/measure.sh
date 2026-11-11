#!/usr/bin/env bash
# Label placement's evidence: this revision drawn with placement on and off, and the base revision,
# on the public corpus (the scene corpus on the core and ir_cairo, every graphical unit on the core),
# from one path; then identity (off against the base everywhere, on against the base wherever no
# label is drawn) and, for every case drawing a camera-facing label, each side against CE (the
# scene's ir_cairo frames, a graphical unit's Cairo control frames).
set -euo pipefail
export PYTHONHASHSEED=0
base_sha=0f8def6eabc11cded7a4632b358b1b536f7a80fe
here=$(pwd)
probe="$here/.github/c53"
work=$(mktemp -d)
echo "work: $work"

head_sha=$(git rev-parse HEAD)
git fetch --quiet --depth 1 origin "$base_sha"
tree="$work/tree"
checkout() {
  git worktree add --quiet --detach "$tree" "$1"
  (cd "$tree" && uv sync --quiet --all-extras --locked)
}

run() {
  local side=$1 placement=$2
  mkdir -p "$work/$side"
  cp "$probe/test_c53_corpus.py" "$tree/tests/frame_ir/test_c53_corpus.py"
  (
    cd "$tree"
    export C53_OUT="$work/$side" PYTHONPATH="$probe" PYTHONWARNINGS=ignore C53_FRAMES=1
    export C53_PLACEMENT="$placement"
    echo "--- $side: scene corpus"
    uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto --dist=loadfile \
      -p no:cacheprovider tests/frame_ir/test_c53_corpus.py | tail -3 || true
    echo "--- $side: graphical units on the core"
    C53_FORCE_CORE=1 uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto \
      --dist=loadfile -p no:cacheprovider tests/test_graphical_units | tail -3 || true
  )
  echo "$side: $(cat "$work/$side"/cases-*.jsonl | wc -l) cases"
}

checkout "$head_sha"
run on on
run off off
git worktree remove --force "$tree"
checkout "$base_sha"
run base on

echo "--- census: label cases, records filled with a nearly clear gradient"
cat "$work/on"/census-*.jsonl
echo "--- blend census over every case (placement on)"
uv run --no-sync python "$probe/blend_summary.py" "$work/on" | tee "$work/blend_summary.txt"
# The census's rows, kept for upload (ci-kit.yml's c53-census gate).
mkdir -p "$here/c53-out"
cat "$work/on"/census-*.jsonl >"$here/c53-out/census.jsonl"
cat "$work/on"/blends-*.jsonl >"$here/c53-out/blends.jsonl"
cp "$work/blend_summary.txt" "$here/c53-out/blend_summary.txt"
echo "--- identity and before/after against CE"
uv run --no-sync python "$probe/compare_m4.py" "$work/on" "$work/off" "$work/base" "$work/summary.json"
