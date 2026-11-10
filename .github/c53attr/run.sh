#!/usr/bin/env bash
# Scratch: tests and renders at this head; where .github/c53attr/full exists, also the RED runs of
# the I1 fix and the frames without the strokes under 0.8 px.
set -uo pipefail
out=$1
mkdir -p "$out"
P=(uv run python -m pytest --no-cov -p no:cacheprovider -o addopts= -rA)
cp .github/c53attr/test_zz_attr_dump.py tests/frame_ir/

run_renders() {  # $1: output dir
  local dir=$1
  mkdir -p "$dir"
  ATTR_DUMP="$PWD/$dir/face_ends.npz" "${P[@]}" -s tests/frame_ir/test_zz_attr_dump.py > "$dir/face_ends.log" 2>&1
  "${P[@]}" -n auto tests/test_graphical_units -k core --set_test > "$dir/gu-set.log" 2>&1
  mkdir -p "$dir/gu"
  for d in tests/test_graphical_units/control_data/*/core; do
    m=$(basename "$(dirname "$d")")
    mkdir -p "$dir/gu/$m"
    cp "$d"/*.npz "$dir/gu/$m/"
  done
  "${P[@]}" -n auto --dist=load tests/test_docs_examples/test_docs_examples.py -k on_core --set_test > "$dir/docs-set.log" 2>&1
  mkdir -p "$dir/docs"
  cp tests/test_docs_examples/control_data/core/*.npz "$dir/docs/"
  git checkout -- tests/test_graphical_units/control_data tests/test_docs_examples/control_data
  git clean -fdq tests/test_graphical_units/control_data tests/test_docs_examples/control_data
}

# The failing tests, as they stand at this head.
"${P[@]}" -n auto tests/frame_ir/test_captured_face_groups.py tests/frame_ir/test_face_groups.py \
  tests/frame_ir/test_face_ends.py tests/test_graphical_units/test_threed.py \
  tests/frame_ir/test_units.py > "$out/tests.log" 2>&1
echo "exit $?" >> "$out/tests.log"

run_renders "$out/head"

if [ -f .github/c53attr/full ]; then
  # RED runs: the I1 fix's draw.py hunks reverted, one at a time and both.
  for which in d1ae8d23 f1b41d6d both; do
    case $which in
      both) git apply -R .github/c53attr/d1ae8d23.patch && git apply -R .github/c53attr/f1b41d6d.patch ;;
      *) git apply -R ".github/c53attr/$which.patch" ;;
    esac
    git diff --stat > "$out/red-$which.log"
    git diff arachne >> "$out/red-$which.log"
    "${P[@]}" -n auto tests/frame_ir/test_captured_face_groups.py tests/frame_ir/test_face_groups.py >> "$out/red-$which.log" 2>&1
    echo "exit $?" >> "$out/red-$which.log"
    git checkout -- arachne/timeline/draw.py
  done
  # The frames without the strokes under 0.8 px.
  python3 .github/c53attr/drop_thin.py
  git diff core > "$out/drop-build.log"
  uv sync --all-extras --locked --reinstall-package arachne-core >> "$out/drop-build.log" 2>&1
  ATTR_DROP_THIN=1 run_renders "$out/drop"
fi
ls -R "$out" > "$out/listing.txt"
exit 0
