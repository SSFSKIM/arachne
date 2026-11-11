#!/usr/bin/env bash
# Records the public corpus's core streams at this revision (the scene corpus on the core and
# ir_cairo, every graphical unit on the core) and sweeps them: every record facing the camera
# carries a facing group, with a planted control (facing_sweep.py).
set -euo pipefail
# str and bytes hashes are salted per process; set iteration over them orders
# some scenes' drawing (TransformMatchingParts' keys, QuickHull's facets).
export PYTHONHASHSEED=0
here=$(pwd)
probe="$here/.github/c53"
out=$(mktemp -d)
echo "out: $out at $(git rev-parse HEAD)"

cp "$probe/test_c53_corpus.py" tests/frame_ir/test_c53_corpus.py
(
  export C53_OUT="$out" PYTHONPATH="$probe" PYTHONWARNINGS=ignore
  echo "--- scene corpus"
  uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto --dist=loadfile \
    -p no:cacheprovider tests/frame_ir/test_c53_corpus.py | tail -3 || true
  echo "--- graphical units on the core"
  C53_FORCE_CORE=1 uv run --no-sync python -m pytest -p c53_probe -o addopts="" -q -n auto \
    --dist=loadfile -p no:cacheprovider tests/test_graphical_units | tail -3 || true
)
echo "recorded: $(cat "$out"/cases-*.jsonl | wc -l) cases, $(ls "$out/xf" | wc -l) streams"
echo "--- facing sweep"
uv run --no-sync python "$probe/facing_sweep.py" "$out"
