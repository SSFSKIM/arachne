"""What the tests of resident sessions share: a host of their own, whose
servers never outlive the test.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import tempfile
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from arachne.server import registry

#: How long a test waits for a server to start, answer or end.
WAIT_S = float(os.environ.get("ARACHNE_TEST_SERVER_WAIT_S", "60"))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gone(pid: int, wait: float = WAIT_S) -> bool:
    """Whether ``pid`` has ended within ``wait`` seconds."""
    until = time.monotonic() + wait
    while alive(pid):
        if time.monotonic() > until:
            return False
        time.sleep(0.05)
    return True


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


class Host:
    """Sessions started in a registry directory of the test's own, each
    server ended after the test, whatever the test left.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.pids: set[int] = set()
        self._connections: list[registry.Connection] = []

    def connect(
        self,
        file: Path,
        scene: str,
        config: dict[str, Any] | None = None,
        **options: Any,
    ) -> registry.Connection:
        connection = registry.connect_or_start(
            file, scene, config, directory=self.directory, **options
        )
        self.pids.add(connection.pid)
        self._connections.append(connection)
        return connection

    def end(self) -> None:
        for connection in self._connections:
            connection.close()
        with contextlib.suppress(Exception):
            registry.stop_all(self.directory)
        for found in self.directory.glob("*.json"):
            with contextlib.suppress(ValueError):
                self.pids.add(int(found.stem))
        for pid in self.pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        for pid in self.pids:
            assert gone(pid), f"server {pid} outlived its test"


@pytest.fixture
def host() -> Iterator[Host]:
    # Under /tmp: a socket's whole path must fit a Unix socket's address.
    directory = Path(tempfile.mkdtemp(prefix="ah-", dir="/tmp"))
    found = Host(directory)
    try:
        yield found
    finally:
        found.end()
        shutil.rmtree(directory, ignore_errors=True)
