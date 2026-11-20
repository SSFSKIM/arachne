"""An import tripwire for the GPU build, installed in one environment.

``python3 tripwire.py install <site-packages> <log>`` writes a start-up hook into the environment's
site directory: every Python process of that environment loads it (the command line, a session
server, a forked worker), and it appends one JSON line (the process id, its command line) to
``<log>`` whenever the process tries to import ``arachne_gpu``, whether or not the import succeeds.
Installed again, the hook writes to the new log; ``python3 tripwire.py uninstall <site-packages>``
takes it out.

``python3 tripwire.py marker`` prints the comment a deliberate import carries in its command line,
by which the log tells it from any other.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The comment a deliberate import carries in its command line.
MARKER = "cpu-bytes: the deliberate import"

HOOK = '''\
"""cpu-bytes' import tripwire: records every attempt to import arachne_gpu."""
import json as _json
import os as _os
import sys as _sys

_LOG = {log!r}


class _Tripwire:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "arachne_gpu":
            try:
                with open(f"/proc/{{_os.getpid()}}/cmdline", "rb") as f:
                    command = f.read().replace(b"\\0", b" ").decode(errors="replace").strip()
            except OSError:
                command = " ".join(_sys.argv)
            with open(_LOG, "a") as log:
                log.write(_json.dumps({{"pid": _os.getpid(), "command": command}}) + "\\n")
        return None

    @staticmethod
    def invalidate_caches():
        pass


_sys.meta_path.insert(0, _Tripwire)
'''


HOOK_FILES = ("_arachne_gpu_tripwire.pth", "_arachne_gpu_tripwire.py")


def uninstall(site: Path) -> None:
    for name in HOOK_FILES:
        (site / name).unlink(missing_ok=True)


def install(site: Path, log: Path) -> None:
    (site / "_arachne_gpu_tripwire.py").write_text(HOOK.format(log=str(log)))
    (site / "_arachne_gpu_tripwire.pth").write_text("import _arachne_gpu_tripwire\n")
    log.write_text("")


if __name__ == "__main__":
    if sys.argv[1:] == ["marker"]:
        print(MARKER)
    elif len(sys.argv) == 4 and sys.argv[1] == "install":
        install(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == "uninstall":
        uninstall(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
