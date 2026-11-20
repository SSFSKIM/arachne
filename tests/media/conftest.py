"""The media tests run in two environments: Arachne's own test run, where the
GPU build is not installed, and the gate that installs it on a machine whose
one adapter is a software one. A test that draws a GPU-route frame is in a
file left out by collection where the build is missing, never skipped; it
takes its device from the launch mode (:mod:`tests.helpers.gpu_door`). A test
about a host with no GPU makes that host itself.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arachne.assets.reader import FIXTURES
from arachne.compose import device
from arachne.session import Session
from arachne.timeline import SourceLine
from tests.helpers import gpu_door

SCENES = Path(__file__).parent / "media_scenes.py"
#: The planted media assets the frame checks must refuse.
BROKEN = Path(__file__).parent / "broken"

#: A small frame, the test clip's own size, at a low rate.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

if importlib.util.find_spec("arachne_gpu") is None:
    # These draw on a device: the launch mode's.
    collect_ignore = ["test_drawn.py", "test_clock_drawn.py"]


def pytest_configure(config):
    gpu_door.report_adapters(config)


@pytest.fixture
def door(monkeypatch) -> gpu_door.LaunchDoor:
    """This run's device, the one every frame of the test is drawn on."""
    return gpu_door.install(monkeypatch)


class _NoGpu:
    """A host with no GPU."""

    def pin(self) -> None:
        return None

    def open(self, width: int, height: int) -> None:
        return None


@pytest.fixture
def no_gpu(monkeypatch):
    """This process as one on a host with no GPU."""
    monkeypatch.setattr(device, "door", _NoGpu())


def line_of(tag: str, scenes: Path = SCENES) -> SourceLine:
    """The line of ``scenes`` that carries the comment ``# @tag``."""
    for number, line in enumerate(scenes.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return SourceLine(scenes.name, number)
    raise LookupError(tag)


@pytest.fixture
def session_of():
    """Open a scene of the scenes file in a session at the small size."""
    opened: list[Session] = []

    def open_session(name: str, scenes: Path = SCENES, **settings: Any) -> Session:
        session = Session.open(scenes, name, config={**SMALL, **settings})
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()


def copied(tmp_path: Path, name: str, as_name: str | None = None) -> Path:
    """The fixture media asset ``name``, copied to ``tmp_path / as_name``."""
    target = tmp_path / (as_name or name)
    shutil.copytree(FIXTURES / name, target)
    return target


def edited(directory: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    """``directory``'s manifest rewritten by ``edit``, which changes the
    parsed document in place.
    """
    path = directory / "manifest.json"
    document = json.loads(path.read_text())
    edit(document)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return directory


def png(
    rows: list[bytes],
    width: int,
    height: int,
    depth: int,
    colour: int,
    chunks: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    """A PNG file of ``height`` ``rows`` (each a row's samples, unfiltered)
    ``width`` pixels wide, of ``depth`` bits per sample and the PNG colour
    type ``colour``, with ``chunks`` (each its kind and data) before its image
    data: written here, so that any form can be made.
    """

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 0)
    data = zlib.compress(b"".join(b"\0" + row for row in rows))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + b"".join(chunk(kind, body) for kind, body in chunks)
        + chunk(b"IDAT", data)
        + chunk(b"IEND", b"")
    )
