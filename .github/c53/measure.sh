#!/usr/bin/env bash
# Measures this revision against the base revision on the public corpus: every frame each draws
# (the scene corpus on the core and ir_cairo, every graphical unit on the core), the core's
# streams without SOURCE against the base's byte for byte, the base's own core reading this
# revision's XF 1.5 streams, and SOURCE's share of this revision's streams.
set -euo pipefail
# str and bytes hashes are salted per process; set iteration over them orders
# some scenes' drawing (TransformMatchingParts' keys, QuickHull's facets).
export PYTHONHASHSEED=0
base_sha=0f8def6eabc11cded7a4632b358b1b536f7a80fe
here=$(pwd)
probe="$here/.github/c53"
work=$(mktemp -d)
echo "work: $work"

head_sha=$(git rev-parse HEAD)
git fetch --quiet --depth 1 origin "$base_sha"
# Both revisions run from one path, in turn: a mobject's stable id and source line carry its
# scene file's path, so two checkouts at different paths would differ in every delta's objects.
tree="$work/tree"
checkout() {
  git worktree add --quiet --detach "$tree" "$1"
  (cd "$tree" && uv sync --quiet --all-extras --locked)
}

run() {
  local side=$1 tree=$2
  mkdir -p "$work/$side"
  cp "$probe/test_c53_corpus.py" "$tree/tests/frame_ir/test_c53_corpus.py"
  (
    cd "$tree"
    export C53_OUT="$work/$side" PYTHONPATH="$probe" PYTHONWARNINGS=ignore
    echo "--- $side: scene corpus"
    uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto --dist=loadfile \
      -p no:cacheprovider tests/frame_ir/test_c53_corpus.py | tail -3 || true
    echo "--- $side: graphical units on the core"
    C53_FORCE_CORE=1 uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto \
      --dist=loadfile -p no:cacheprovider tests/test_graphical_units | tail -3 || true
  )
  echo "$side: $(cat "$work/$side"/cases-*.jsonl | wc -l) cases, $(ls "$work/$side/xf" | wc -l) streams"
}

checkout "$head_sha"
run head "$tree"
git worktree remove --force "$tree"
checkout "$base_sha"
run base "$tree"

echo "--- control: the base against itself under another hash seed"
mkdir -p "$work/base-seed1"
(
  cd "$tree"
  export C53_OUT="$work/base-seed1" PYTHONPATH="$probe" PYTHONWARNINGS=ignore PYTHONHASHSEED=1
  C53_FORCE_CORE=1 uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -p no:cacheprovider \
    tests/test_graphical_units/test_polyhedra.py tests/test_graphical_units/test_transform_matching_parts.py \
    | tail -1 || true
)
python3 - "$work/base" "$work/base-seed1" <<'PY'
import json, sys
from pathlib import Path
def load(root):
    found = {}
    for path in Path(root).glob("cases-*.jsonl"):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            found[entry["case"]] = entry["frames"]
    return found
a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
differ = [c for c in common if a[c] != b[c]]
print(f"base seed 0 vs seed 1: {len(common)} cases, {len(differ)} draw differently")
for c in differ:
    print("  ", c)
PY
echo "--- the base revision's core reads this revision's streams"
(cd "$tree" && uv run --no-sync python "$probe/old_reader.py" "$work/head/xf" "$work/base/xf" \
  "$work/old_reader.json")
echo "--- where streams without SOURCE differ"
(uv run --no-sync python "$probe/diag.py" "$work/head" "$work/base" "$work/old_reader.json" \
  test_polyhedra test_transform_matching) || true
echo "--- comparison and SOURCE size"
python3 "$probe/compare.py" "$work/head" "$work/base" "$work/old_reader.json" "$work/summary.json"
