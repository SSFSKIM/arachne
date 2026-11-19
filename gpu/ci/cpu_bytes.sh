#!/usr/bin/env bash
# At one revision, a scene that uses nothing new is drawn to the same bytes with the GPU build
# installed and without it. Two environments are made from this commit, the engine alone
# and the engine with the GPU build's wheel; the first draws before the software Vulkan adapter is
# installed, the second after. Each has fresh roots for media, partial movies and the session store,
# and an import tripwire; the runs must leave no record in either, and one deliberate import after
# them must leave one.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd "$root"
command -v uv >/dev/null || pipx install uv==0.12.3
(source gpu/ci/toolchain.sh)

work=$(mktemp -d)
uv build --wheel --out-dir "$work/wheels" gpu
wheel=$(ls "$work"/wheels/arachne_gpu-*.whl)
echo "wheel: $(basename "$wheel")"

marker="cpu-bytes: the deliberate import"
for arm in without with; do
  UV_PROJECT_ENVIRONMENT="$work/$arm" uv sync --all-extras --locked --quiet
done
uv pip install --python "$work/with/bin/python" "$wheel"
if "$work/without/bin/python" -c 'import arachne_gpu' 2>/dev/null; then
  echo "FAIL: the GPU build is importable in the environment without it"
  exit 1
fi
for arm in without with; do
  site=$("$work/$arm/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  python3 gpu/ci/tripwire.py install "$site" "$work/$arm.tripwire"
done

echo "--- without the GPU build"
"$work/without/bin/python" gpu/ci/cpu_bytes_draw.py "$work/without-roots" "$work/without.json"
bash gpu/ci/software_vulkan.sh
echo "--- with the GPU build and lavapipe"
"$work/with/bin/python" gpu/ci/cpu_bytes_draw.py "$work/with-roots" "$work/with.json"

# The deliberate import, one in each environment: the tripwire must see it.
"$work/without/bin/python" -c "
try:
    import arachne_gpu  # $marker
except ImportError:
    pass"
"$work/with/bin/python" -c "import arachne_gpu  # $marker"
python3 gpu/ci/cpu_bytes_compare.py "$work/without.json" "$work/with.json" \
  "$work/without.tripwire" "$work/with.tripwire" "$marker"
