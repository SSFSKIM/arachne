#!/usr/bin/env bash
# Builds the core's wheel (one abi3 wheel for CPython 3.12+), checks the extension links no GPU
# library, and runs the binding's tests under each Python version given (default 3.14).
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
command -v uv >/dev/null || pipx install uv==0.12.3

work=$(mktemp -d)
uv build --wheel --out-dir "$work/wheels" .
wheel=$(ls "$work"/wheels/arachne_core-*.whl)
echo "wheel: $(basename "$wheel")"
python3 -m zipfile -e "$wheel" "$work/unpacked"
python3 ci/tree_check.py links "$work"/unpacked/arachne_core/_core*

for version in "${@:-3.14}"; do
  echo "--- Python $version"
  uv venv --quiet --python "$version" "$work/venv-$version"
  VIRTUAL_ENV="$work/venv-$version" uv pip install --quiet "$wheel" pytest
  "$work/venv-$version/bin/python" -m pytest -q
done
