"""The compose tests run in two environments: Arachne's own test run, where
the GPU build is not installed, and the gate that installs it on a machine
whose one adapter is a software one. A test that needs a device is left out
by collection where the build is missing, never skipped; a test about a
host with or without a GPU makes that host itself, through the device's
door (:data:`arachne.compose.device.door`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arachne import tempconfig
from arachne.compose import device
from arachne.session import Session
from arachne.timeline import SourceLine

SCENES = Path(__file__).parent / "compose_scenes.py"

#: A small frame at a low rate, so drawing and encoding cost little.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

if importlib.util.find_spec("arachne_gpu") is None:
    # These draw on a device: the software adapter, through the GPU build's
    # testing door.
    collect_ignore = ["test_gpu_session.py"]

#: A pin no device here has.
PIN = {
    "adapter": "A Card",
    "backend": "Vulkan",
    "driver": "1.2.3",
    "wgpu": "30.0.2",
}


class _Door:
    """A host as a test makes it: with the device ``pin`` names, or none."""

    def __init__(self, pin: dict[str, str] | None) -> None:
        self._pin = pin

    def pin(self) -> dict[str, str] | None:
        return self._pin

    def open(self, width: int, height: int):
        if self._pin is None:
            return None
        raise AssertionError("this host's device draws nothing")


@pytest.fixture
def no_gpu(monkeypatch):
    """This process as one on a host with no GPU."""
    monkeypatch.setattr(device, "door", _Door(None))


@pytest.fixture
def a_gpu(monkeypatch):
    """This process as one on a host with a GPU, which draws nothing."""
    monkeypatch.setattr(device, "door", _Door(dict(PIN)))


def snapshot(root: Path) -> dict[str, str | None]:
    """Every file and directory under ``root``, by its path there: a file's
    sha256, ``None`` for a directory.
    """
    found: dict[str, str | None] = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        found[name] = (
            None if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
        )
    return found


def render(scene: type, media: Path, **options) -> None:
    """Render ``scene`` into ``media`` at the small size, as the command
    line renders it.
    """
    with tempconfig({**SMALL, "media_dir": str(media), **options}):
        scene().render()


def line_of(tag: str) -> SourceLine:
    """The line of the scenes file that carries the comment ``# @tag``."""
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return SourceLine(SCENES.name, number)
    raise LookupError(tag)


@pytest.fixture
def session_of():
    """Open a scene of the scenes file in a session at the small size."""
    opened: list[Session] = []

    def open_session(name: str, **settings) -> Session:
        session = Session.open(SCENES, name, config={**SMALL, **settings})
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()


def run_child(code: str, env: dict[str, str] | None = None) -> str:
    """Run ``code`` in a fresh interpreter of this environment, from the
    repository's root, and return what it printed.
    """
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout
