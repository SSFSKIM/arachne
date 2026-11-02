from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arachne import tempconfig
from arachne.manager import Manager
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
    scene and every frame its renderer drew, repeats expanded: those of the
    execution that ended the render, when the scene was executed again.
    """
    executions: list[list[np.ndarray]] = []
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
        manager = Manager(scene)

        def collect() -> None:
            # On the writer instance: a class the engine defines, patched,
            # would take every play off its route.
            frames: list[np.ndarray] = []
            executions.append(frames)

            def write_frame(pixels, *, repeat=1):
                frames.extend([pixels] * repeat)

            scene.renderer.file_writer.write_frame = write_frame

        renew = manager._renew_scene

        def renewed() -> None:
            renew()
            collect()

        manager._renew_scene = renewed
        collect()
        scene.render()
    return scene, executions[-1]


@pytest.fixture
def session_of():
    """Open a scene of ``scenes.py`` (or of ``path``) in a session at the
    small frame size, on its render's grid unless ``grid`` says otherwise.
    """
    opened: list[Session] = []

    def open_session(
        name: str,
        path: Path = SCENES,
        *,
        grid="render",
        stepping=True,
        on_step=None,
        **settings,
    ) -> Session:
        session = Session.open(
            path,
            name,
            grid=grid,
            config={**SMALL, **settings},
            stepping=stepping,
            on_step=on_step,
        )
        opened.append(session)
        return session

    yield open_session
    for session in opened:
        session.close()
