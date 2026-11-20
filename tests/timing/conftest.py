"""The timing tests' fixture: sessions of the timing scenes, closed after
each test.
"""

from __future__ import annotations

import pytest

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
