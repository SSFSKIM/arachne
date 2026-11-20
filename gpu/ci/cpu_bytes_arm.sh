#!/usr/bin/env bash
# One arm of the bytes comparison: the scenes of cpu_bytes_scenes.py drawn with one environment's
# interpreter into fresh roots, under the import tripwire, then the one deliberate import the
# tripwire must record. It installs and syncs nothing, so a host that already has its environments
# runs it as it is, an arm at a time; cpu_bytes_compare.py then compares the arms.
#
#     bash gpu/ci/cpu_bytes_arm.sh <python> <arm directory> absent|present
#
# absent or present is what the arm says of its environment's GPU build, and the arm fails if the
# environment says otherwise. The arm directory must be new or empty; it receives roots/ (media,
# partial movies, the session store), digests.json, tripwire.log, and arm.json: the build's
# presence and, where present, the adapters it sees, which cpu_bytes_compare.py can be asked to
# hold to (--no-hardware, --hardware). The tripwire is armed in the environment only while the
# arm draws and makes its deliberate import.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
if [ $# -ne 3 ] || { [ "$3" != absent ] && [ "$3" != present ]; }; then
  echo "usage: cpu_bytes_arm.sh <python> <arm directory> absent|present" >&2
  exit 2
fi
python=$1 arm=$2 build=$3
if [ -e "$arm" ] && [ -n "$(ls -A "$arm")" ]; then
  echo "FAIL: $arm is not empty: an arm draws into fresh roots"
  exit 1
fi
mkdir -p "$arm"
arm=$(cd "$arm" && pwd)
cd "$root"

# The tripwire lives in the environment's site directory, so it is scoped to this arm: any hook
# an earlier arm left there is taken out before anything runs, and this arm's is taken out when
# it ends, however it ends. Nothing of an arm is ever written to another arm's log.
site=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$python" "$here/tripwire.py" uninstall "$site"
trap '"$python" "$here/tripwire.py" uninstall "$site"' EXIT

# What the environment is, read before the tripwire is armed: whether the GPU build is there to
# be found (not whether an import of it happens to fail), and the adapters it sees.
found=$("$python" -c 'import importlib.util; print(importlib.util.find_spec("arachne_gpu") is not None)')
if [ "$build" = absent ]; then
  if [ "$found" != False ]; then
    echo "FAIL: the GPU build is installed in the environment without it ($python)"
    exit 1
  fi
  adapters=null
else
  if [ "$found" != True ]; then
    echo "FAIL: the GPU build is not installed in the environment with it ($python)"
    exit 1
  fi
  adapters=$("$python" -c 'import json, arachne_gpu; print(json.dumps(arachne_gpu.adapters()))')
fi
"$python" -c '
import json, os, sys
python, build, adapters, out = sys.argv[1:]
adapters = json.loads(adapters)
with open(out, "w") as f:
    json.dump({"python": python, "gpu_build": build, "adapters": adapters}, f, indent=1)
seen = ", ".join("%s (%s)" % (a["name"], a["device_type"]) for a in adapters or []) or "none"
name = os.path.basename(os.path.dirname(out))
print("arm %s: the GPU build %s; adapters: %s" % (name, build, seen))
' "$python" "$build" "$adapters" "$arm/arm.json"

"$python" "$here/tripwire.py" install "$site" "$arm/tripwire.log"
"$python" "$here/cpu_bytes_draw.py" "$arm/roots" "$arm/digests.json"

# The deliberate import: the tripwire must see it.
marker=$("$python" "$here/tripwire.py" marker)
if [ "$build" = absent ]; then
  "$python" -c "
try:
    import arachne_gpu  # $marker
except ImportError:
    pass"
else
  "$python" -c "import arachne_gpu  # $marker"
fi
