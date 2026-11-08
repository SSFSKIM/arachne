"""Draw reuse changes no frame: every witness scene of the timeline tests, drawn
with reuse and without, writes the same bytes, on each draw path.

A reused frame stands for a frame drawn from the same recorded state, so it is
right only when its key covers everything a drawing is given of a capture. The
scenes are found, not listed, so that a scene added for a new kind of
recorded input (the cameras' and the reuse tests') is held to the same
check without anyone remembering to add it here.
"""

from __future__ import annotations

import inspect

import pytest

from arachne import Scene

from . import camera_scenes, reuse_scenes
from .test_draw_reuse import assert_same_writes, render_writes
from .test_video_output import CASES


def _scenes_of(module) -> list[type]:
    return [
        value
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, Scene)
        and value.__module__ == module.__name__
        and not name.startswith("_")
    ]


WITNESSES = list(
    dict.fromkeys(
        [
            *_scenes_of(camera_scenes),
            *_scenes_of(reuse_scenes),
            *(case.values[0] for case in CASES),
        ]
    )
)


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
@pytest.mark.parametrize(
    "scene_class", WITNESSES, ids=lambda scene_class: scene_class.__name__
)
def test_every_write_is_the_same_with_and_without_reuse(scene_class, draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    _, on = render_writes(scene_class, draw_path=draw_path)
    _, off = render_writes(scene_class, draw_path=draw_path, draw_reuse=False)
    assert on
    assert_same_writes(on, off)
