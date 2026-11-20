"""The timing tests' fixtures: sessions of the timing scenes, closed after
each test; and the scene drivers' finders, as they were before it.
"""

from __future__ import annotations

import pytest

from arachne.assets import timing
from arachne.session import Session
from arachne.timeline import admission

from .reading import SCENES, SMALL


@pytest.fixture
def opened():
    sessions: list[Session] = []
    admission.DEMOTED.clear()

    def open_scene(name: str, fps: int) -> Session:
        session = Session.open(SCENES, name, config={**SMALL, "frame_rate": fps})
        sessions.append(session)
        return session

    yield open_scene
    for session in sessions:
        session.close()
    admission.DEMOTED.clear()


@pytest.fixture(autouse=True)
def finders():
    """Let go of the finders a test's scenes registered: each execution of a
    scene file registers its own.
    """
    before = list(timing._FINDERS)
    yield
    timing._FINDERS[:] = before
