#!/usr/bin/env bash
# Scratch (C5.3, never merged): writes the core's XF streams, SOURCE included, of the public
# corpus at this commit's parent: the head side of .github/c53/measure.sh (the scene corpus, then
# every graphical unit on the core), run from this checkout. Leaves in <out> only the streams
# (xf/), their SHA256SUMS and manifest.json.
set -euo pipefail
# str and bytes hashes are salted per process; set iteration over them orders
# some scenes' drawing (TransformMatchingParts' keys, QuickHull's facets).
export PYTHONHASHSEED=0
expected=551
here=$(pwd)
probe="$here/.github/c53streams"
out=$(realpath -m "$1")
work=$(mktemp -d)
mkdir -p "$out"

# A mobject's stable id and source line carry its scene file's path: the streams hold this
# checkout's path, recorded in the manifest.
cp "$probe/test_c53_corpus.py" "$here/tests/frame_ir/test_c53_corpus.py"
(
  export C53_OUT="$work" PYTHONPATH="$probe" PYTHONWARNINGS=ignore
  echo "--- scene corpus"
  uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto --dist=loadfile \
    -p no:cacheprovider tests/frame_ir/test_c53_corpus.py | tail -3 || true
  echo "--- graphical units on the core"
  C53_FORCE_CORE=1 uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto \
    --dist=loadfile -p no:cacheprovider tests/test_graphical_units | tail -3 || true
)
rm "$here/tests/frame_ir/test_c53_corpus.py"

cases=$(cat "$work"/cases-*.jsonl | wc -l)
errors=$(cat "$work"/errors-*.jsonl 2>/dev/null | wc -l || true)
mv "$work/xf" "$out/xf"
streams=$(find "$out/xf" -maxdepth 1 -type f -name '*.xf' | wc -l)
bytes=$(find "$out/xf" -maxdepth 1 -type f -name '*.xf' -printf '%s\n' | awk '{s += $1} END {print s + 0}')
echo "head: $cases cases ($errors scene errors), $streams streams, $bytes bytes"
(cd "$out/xf" && sha256sum -- * > ../SHA256SUMS)

export M_HEAD=$(git rev-parse HEAD^) M_HEAD_TREE=$(git rev-parse 'HEAD^^{tree}')
export M_CHECKOUT=$(git rev-parse HEAD) M_CHECKOUT_TREE=$(git rev-parse 'HEAD^{tree}')
export M_STREAMS=$streams M_BYTES=$bytes M_CASES=$cases M_PATH=$here M_RUN=${GITHUB_RUN_ID:-}
export M_PYTHON=$(uv run --no-sync python -c 'import platform; print(platform.python_version())')
python3 - "$out/manifest.json" <<'PY'
import json, os, sys
e = os.environ
manifest = {
    "head": e["M_HEAD"],
    "head_tree": e["M_HEAD_TREE"],
    "checkout_commit": e["M_CHECKOUT"],
    "checkout_tree": e["M_CHECKOUT_TREE"],
    "checkout_path": e["M_PATH"],
    "streams": int(e["M_STREAMS"]),
    "total_bytes": int(e["M_BYTES"]),
    "cases": int(e["M_CASES"]),
    "run_id": e["M_RUN"],
    "python": e["M_PYTHON"],
    "pythonhashseed": "0",
}
with open(sys.argv[1], "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY
cat "$out/manifest.json"

if [ "$streams" -ne "$expected" ]; then
  echo "::error::$streams streams, expected $expected"
  exit 1
fi
