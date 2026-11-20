#!/usr/bin/env bash
# At one revision, a scene that uses nothing new is drawn to the same bytes with the GPU build
# installed and without it. Two environments are made from this commit, the engine alone
# and the engine with the GPU build's wheel; the first draws before the software Vulkan adapter is
# installed, the second after. Each arm (cpu_bytes_arm.sh) has fresh roots for media, partial
# movies and the session store, and an import tripwire; the runs must leave no record in either,
# and one deliberate import after them must leave one.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
command -v uv >/dev/null || pipx install uv==0.12.3
(source gpu/ci/toolchain.sh)

work=$(mktemp -d)
uv build --wheel --out-dir "$work/wheels" gpu
wheel=$(ls "$work"/wheels/arachne_gpu-*.whl)
echo "wheel: $(basename "$wheel")"

for arm in without with; do
  UV_PROJECT_ENVIRONMENT="$work/$arm" uv sync --all-extras --locked --quiet
done
uv pip install --python "$work/with/bin/python" "$wheel"

echo "--- without the GPU build"
bash gpu/ci/cpu_bytes_arm.sh "$work/without/bin/python" "$work/arm-without" absent
bash gpu/ci/software_vulkan.sh
echo "--- with the GPU build and lavapipe"
bash gpu/ci/cpu_bytes_arm.sh "$work/with/bin/python" "$work/arm-with" present
python3 gpu/ci/cpu_bytes_compare.py "$work/arm-without" "$work/arm-with"
