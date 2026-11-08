"""The memory a server's entry reports is the server's own: on Linux its own
peak, never the peak of the process that started it, which ``ru_maxrss``
carries across fork and exec; elsewhere ``ru_maxrss``, which on macOS is the
process's own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from arachne.server import registry

from .conftest import WAIT_S, write

MIB = 1 << 20

STILL = """\
    from arachne import Scene, Square


    class Small(Scene):
        def construct(self):
            self.add(Square(side_length=1))
            self.wait(0.5)


    class Large(Small):
        pass
"""

#: A client holding ``argv[2]`` MiB, every page of it written, that starts a
#: server of the registry ``argv[1]``: its own child, run as ``python -m
#: arachne.server --socket``, without ``argv[3]``; with it, the session of
#: the scene ``argv[4]`` of the file ``argv[3]`` by
#: :func:`registry.connect_or_start`, whose server a launcher's fork detaches.
#: It prints the server's pid, once it serves, and the client's own peak in
#: MiB. The first way, ``ru_maxrss`` carries the client's peak into the
#: server; the second, the fork that detaches it starts its peak afresh.
CLIENT = """\
import json, os, subprocess, sys
from pathlib import Path

from arachne.server import registry

held = bytearray(b"x") * (int(sys.argv[2]) << 20)
if len(sys.argv) > 3:
    connection = registry.connect_or_start(
        Path(sys.argv[3]),
        sys.argv[4],
        {"pixel_width": 96, "pixel_height": 54, "frame_rate": 4},
        directory=Path(sys.argv[1]),
    )
    connection.close()
    pid = connection.pid
else:
    server = subprocess.Popen(
        [sys.executable, "-m", "arachne.server", "--socket"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "ARACHNE_SESSION_DIR": sys.argv[1]},
    )
    pid = json.loads(server.stdout.readline())["params"]["pid"]
print(json.dumps({"pid": pid, "own_mb": registry._rss_mb(), "held": len(held)}))
"""

#: A process that holds about 300 MiB, frees it, and resets its peak
#: (``5`` to ``/proc/self/clear_refs``): its peak before and after the
#: reset, in MiB, or ``null`` where it may not write the file.
RESET = """\
import json

from arachne.server import registry

held = bytearray(b"x") * (300 << 20)
del held
before = registry._rss_mb()
try:
    with open("/proc/self/clear_refs", "w") as refs:
        refs.write("5")
except OSError as error:
    print(json.dumps({"refused": str(error)}))
else:
    print(json.dumps({"before": before, "after": registry._rss_mb()}))
"""

ON_LINUX = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux: VmHWM, and the starting process's peak in ru_maxrss",
)


def _run(*argv: str) -> dict:
    """What a Python process running ``argv`` printed, as JSON."""
    done = subprocess.run(
        [sys.executable, "-c", *argv],
        capture_output=True,
        text=True,
        timeout=WAIT_S,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _started(host, held_mb: int, *session: str) -> tuple[float, float]:
    """The entry's memory of a server started by a client holding
    ``held_mb``, and that client's own peak (:data:`CLIENT`).
    """
    started = _run(CLIENT, os.fspath(host.directory), str(held_mb), *session)
    host.pids.add(started["pid"])
    entries = {entry.pid: entry for entry in registry.sessions(host.directory)}
    return entries[started["pid"]].rss_mb, started["own_mb"]


def _ru_maxrss(monkeypatch: pytest.MonkeyPatch, value: int) -> None:
    monkeypatch.setattr(
        registry.resource,
        "getrusage",
        lambda who: SimpleNamespace(ru_maxrss=value),
    )


def test_on_linux_the_peak_is_its_own_high_water_mark(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython3\n"
        "VmPeak:\t 9000000 kB\n"
        "VmSize:\t 8000000 kB\n"
        "VmHWM:\t   51200 kB\n"
        "VmRSS:\t   40960 kB\n"
    )
    monkeypatch.setattr(registry, "_STATUS", status)
    monkeypatch.setattr(registry.sys, "platform", "linux")
    # What the process that started it held at most: never this one's.
    _ru_maxrss(monkeypatch, 9_000 * 1024)

    assert registry._rss_mb() == 50.0


def test_on_linux_without_proc_the_peak_is_ru_maxrss_in_kilobytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(registry, "_STATUS", tmp_path / "no-proc" / "status")
    monkeypatch.setattr(registry.sys, "platform", "linux")
    _ru_maxrss(monkeypatch, 300 * 1024)

    assert registry._rss_mb() == 300.0


def test_on_macos_the_peak_is_ru_maxrss_in_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_STATUS", tmp_path / "no-proc" / "status")
    monkeypatch.setattr(registry.sys, "platform", "darwin")
    _ru_maxrss(monkeypatch, 300 * MIB)

    assert registry._rss_mb() == 300.0


@ON_LINUX
def test_a_server_started_by_a_large_client_reports_its_own_peak(host):
    small, _ = _started(host, 0)
    large, client_mb = _started(host, 300)
    # The client did hold what the server must not report.
    assert client_mb >= 300
    # The engine's own footprint sets what a server holds, whoever started
    # it: a server started by the large client reports what one started by
    # a small client does, give or take, never the 300 MiB more it was
    # spawned from.
    assert small > 0
    assert large < small + 150, (small, large)


@ON_LINUX
def test_a_server_the_registry_starts_for_a_large_client_reports_its_own_peak(
    host, tmp_path
):
    path = os.fspath(write(tmp_path, "still.py", STILL))
    small, _ = _started(host, 0, path, "Small")
    large, client_mb = _started(host, 300, path, "Large")
    assert client_mb >= 300
    # Detached by its launcher's fork, the server starts its peak afresh:
    # nothing of the client's reaches its entry.
    assert small > 0
    assert large < small + 150, (small, large)


@ON_LINUX
def test_a_reset_peak_is_the_peak_since():
    peaks = _run(RESET)
    if "refused" in peaks:
        pytest.skip(f"/proc/self/clear_refs is not writable here: {peaks['refused']}")
    # The 300 MiB held and freed counts before the reset, and not after it.
    assert peaks["before"] - peaks["after"] > 250, peaks
