from __future__ import annotations

from pathlib import Path

import pytest

from arachne.session import Session
from arachne.timeline import admission

SCENES = Path(__file__).parent / "lint_scenes.py"

#: A small frame, so drawing costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54}


def line_of(tag: str, path: Path = SCENES) -> int:
    """The line of ``path`` that carries the comment ``# @tag``."""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


@pytest.fixture
def session_of():
    """Open a scene of ``lint_scenes.py`` (or of ``path``) in a session at
    the small frame size; each test starts and ends with no demotion.
    """
    opened: list[Session] = []
    admission.DEMOTED.clear()

    def open_session(
        name: str, path: Path = SCENES, *, grid="render", **settings
    ) -> Session:
        session = Session.open(path, name, grid=grid, config={**SMALL, **settings})
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()
    admission.DEMOTED.clear()
