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

from .conftest import WAIT_S

MIB = 1 << 20

#: A client holding ``argv[2]`` MiB, every page of it written, that starts a
#: server of the registry ``argv[1]`` itself, as its child: the server's pid,
#: once it serves, and the client's own peak in MiB. (A server
#: :func:`registry.connect_or_start` starts is detached by a fork that starts
#: its peak afresh; one a client runs itself carries the client's in
#: ``ru_maxrss``.)
CLIENT = """\
import json, os, subprocess, sys

from arachne.server import registry

held = bytearray(b"x") * (int(sys.argv[2]) << 20)
server = subprocess.Popen(
    [sys.executable, "-m", "arachne.server", "--socket"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env={**os.environ, "ARACHNE_SESSION_DIR": sys.argv[1]},
)
ready = json.loads(server.stdout.readline())
print(json.dumps({"pid": ready["params"]["pid"], "own_mb": registry._rss_mb(), "held": len(held)}))
"""


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


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux: where ru_maxrss carries the starting process's peak",
)
def test_a_server_started_by_a_large_client_reports_its_own_peak(host):
    def start(held_mb: int) -> tuple[float, float]:
        """The entry's memory of a server started by a client holding
        ``held_mb``, and that client's own peak.
        """
        client = subprocess.run(
            [sys.executable, "-c", CLIENT, os.fspath(host.directory), str(held_mb)],
            capture_output=True,
            text=True,
            timeout=WAIT_S,
            check=False,
        )
        assert client.returncode == 0, client.stderr
        started = json.loads(client.stdout)
        host.pids.add(started["pid"])
        entries = {entry.pid: entry for entry in registry.sessions(host.directory)}
        return entries[started["pid"]].rss_mb, started["own_mb"]

    small, _ = start(0)
    large, client_mb = start(300)
    # The client did hold what the server must not report.
    assert client_mb >= 300
    # The engine's own footprint sets what a server holds, whoever started
    # it: a server started by the large client reports what one started by
    # a small client does, give or take, never the 300 MiB more it was
    # spawned from.
    assert small > 0
    assert large < small + 150, (small, large)
