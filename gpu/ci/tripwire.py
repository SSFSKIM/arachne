"""An import tripwire for the GPU build, installed in one environment.

``python3 tripwire.py install <site-packages> <log>`` writes a start-up hook into the environment's
site directory: every Python process of that environment loads it (the command line, a session
server, a forked worker), and it appends one JSON line (the process id, its command line) to
``<log>`` whenever the process tries to import ``arachne_gpu``, whether or not the import succeeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def install(site: Path, log: Path) -> None:
    (site / "_arachne_gpu_tripwire.py").write_text(HOOK.format(log=str(log)))
    (site / "_arachne_gpu_tripwire.pth").write_text("import _arachne_gpu_tripwire\n")
    log.write_text("")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "install":
        sys.exit(__doc__)
    install(Path(sys.argv[2]), Path(sys.argv[3]))
