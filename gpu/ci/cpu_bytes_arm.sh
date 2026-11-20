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
# presence and, where present, the adapters it sees.
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

# What the environment is, read before the tripwire is armed.
if [ "$build" = absent ]; then
  if "$python" -c 'import arachne_gpu' 2>/dev/null; then
    echo "FAIL: the GPU build is importable in the environment without it ($python)"
    exit 1
  fi
  adapters=null
else
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

site=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
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
