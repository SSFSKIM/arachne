from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne import tempconfig
from arachne.session import Session

SCENES = Path(__file__).parent / "scenes.py"

#: A small frame, so drawing costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54}


def line_of(tag: str) -> int:
    """The line of ``scenes.py`` that carries the comment ``# @tag``."""
    for number, line in enumerate(SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def render(scene_class: type, fps: int, **settings) -> tuple[object, list[np.ndarray]]:
    """Render ``scene_class`` at ``fps`` with nothing written, and return the
    scene and every frame its renderer drew, repeats expanded.
    """
    frames: list[np.ndarray] = []
    with tempconfig(
        {
            **SMALL,
            "frame_rate": fps,
            "dry_run": True,
            "disable_caching": True,
            **settings,
        }
    ):
        scene = scene_class()
        writer = scene.renderer.file_writer

        def collect(pixels, *, repeat=1):
            frames.extend([pixels] * repeat)

        writer.write_frame = collect
        scene.render()
    return scene, frames


@pytest.fixture
def session_of():
    """Open a scene of ``scenes.py`` (or of ``path``) in a session at the
    small frame size.
    """
    opened: list[Session] = []

    def open_session(name: str, path: Path = SCENES, **settings) -> Session:
        session = Session.open(path, name, config={**SMALL, **settings})
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()
