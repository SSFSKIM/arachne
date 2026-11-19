"""The routing tests run in two environments: Arachne's own test run, where
the GPU build is not installed, and the gate that installs it on a machine
with no GPU. A test about the build's absence makes that absence itself; a
test that needs the build is left out by collection where it is missing,
never skipped.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from arachne.session import Session
from arachne.timeline import SourceLine

SCENES = Path(__file__).parent / "routing_scenes.py"

#: A small frame, so drawing costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54}

if importlib.util.find_spec("arachne_gpu") is None:
    collect_ignore = ["test_with_build.py"]


def line_of(tag: str) -> SourceLine:
    """The line of the scenes file that carries the comment ``# @tag``."""
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return SourceLine(SCENES.name, number)
    raise LookupError(tag)


@pytest.fixture
def session_of():
    """Open a scene of the scenes file in a session at the small frame size."""
    opened: list[Session] = []

    def open_session(name: str) -> Session:
        session = Session.open(SCENES, name, config=dict(SMALL))
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()


def run_child(code: str, env: dict[str, str] | None = None) -> str:
    """Run ``code`` in a fresh interpreter of this environment, from the
    repository's root, and return what it printed; its failure fails the
    test with what it wrote.
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
