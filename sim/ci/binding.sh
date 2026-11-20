#!/usr/bin/env bash
# Builds the simulation crate's wheel (one abi3 wheel for CPython 3.12+), checks its extension,
# the shipped artifact: it calls no transcendental function of the platform's maths library and
# no fused multiply-add (ci/imports.py), and links no GPU library (the core's check). Then runs
# the binding's tests under each Python version given (default 3.14).
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
command -v uv >/dev/null || pipx install uv==0.12.3

work=$(mktemp -d)
uv build --wheel --out-dir "$work/wheels" .
wheel=$(ls "$work"/wheels/arachne_sim-*.whl)
echo "wheel: $(basename "$wheel")"
python3 -m zipfile -e "$wheel" "$work/unpacked"
extension=$(ls "$work"/unpacked/arachne_sim/_sim*.so)
python3 ci/imports.py "$extension"
python3 ../core/ci/tree_check.py links "$extension"

for version in "${@:-3.14}"; do
  echo "--- Python $version"
  uv venv --quiet --python "$version" "$work/venv-$version"
  VIRTUAL_ENV="$work/venv-$version" uv pip install --quiet "$wheel" pytest
  "$work/venv-$version/bin/python" -m pytest -q
done
