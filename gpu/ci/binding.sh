#!/usr/bin/env bash
# Builds the GPU build's wheel, installs it into the environment the shared setup made (.venv), and
# runs each given test directory with that interpreter, in software mode on lavapipe. Fails if a
# directory collects no test or skips one.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
bash gpu/ci/software_vulkan.sh
(source gpu/ci/toolchain.sh)
command -v uv >/dev/null || pipx install uv==0.12.3

work=$(mktemp -d)
uv build --wheel --out-dir "$work/wheels" gpu
wheel=$(ls "$work"/wheels/arachne_gpu-*.whl)
echo "wheel: $(basename "$wheel")"
python3 -m zipfile -e "$wheel" "$work/unpacked"
extension=$(ls "$work"/unpacked/arachne_gpu/_gpu*.so)
linked=$(readelf -d "$extension" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')
echo "links: $linked"
case "$linked" in *vulkan*) echo "FAIL: the extension links the Vulkan loader at build time"; exit 1 ;; esac

uv pip install --python .venv/bin/python "$wheel"
export ARACHNE_GPU_TEST_MODE=software
n=0
for tests in "$@"; do
  n=$((n + 1))
  echo "--- $tests"
  .venv/bin/python -m pytest -p no:cacheprovider -p no:randomly --junitxml="$work/junit-$n.xml" "$tests"
  python3 gpu/ci/junit_check.py "$work/junit-$n.xml"
done
