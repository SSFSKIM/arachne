"""The effects' tests run in Arachne's own test run, where the GPU build is
not installed, and in the gate that installs it on a machine whose one
adapter is a software one. A test file that draws is left out by collection
where the build is missing, never skipped; it draws on this run's device
(:mod:`tests.helpers.gpu_door`).
"""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne.compose import device
from arachne.session import Session
from tests.helpers import gpu_door

SCENES = Path(__file__).parent / "effects_scenes.py"

#: A small frame at a low rate, so drawing costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54, "frame_rate": 15}

if importlib.util.find_spec("arachne_gpu") is None:
    collect_ignore = ["test_drawn.py", "test_window_drawn.py"]


def pytest_configure(config):
    gpu_door.report_adapters(config)


class _NoGpu:
    """A host with no GPU."""

    def pin(self):
        return None

    def open(self, width: int, height: int):
        return None


@pytest.fixture
def no_gpu(monkeypatch):
    """This process as one on a host with no GPU."""
    monkeypatch.setattr(device, "door", _NoGpu())


@pytest.fixture
def door(monkeypatch) -> gpu_door.LaunchDoor:
    """This run's device, the one every frame of the test is drawn on."""
    return gpu_door.install(monkeypatch)


@pytest.fixture
def session_of():
    """Open a scene of the scenes file in a session, at the small size
    unless the settings say otherwise.
    """
    opened: list[Session] = []

    def open_session(name: str, **settings) -> Session:
        session = Session.open(SCENES, name, config={**SMALL, **settings})
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()


def pixels(session: Session, t: Fraction | float) -> np.ndarray:
    """The frame at ``t`` as integers, height x width x RGBA."""
    return np.asarray(session.frame(t)).astype(int)
