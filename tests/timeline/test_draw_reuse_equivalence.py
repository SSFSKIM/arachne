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
from arachne.frame_ir.extract import Unable
from arachne.frame_ir.painter import deliveries

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


def _outcome(scene_class, **settings):
    """The writes of a render, or why the draw path could not draw its frames
    (``render.unable``, which the path names and never draws wrong).
    """
    try:
        _, writes = render_writes(scene_class, **settings)
    except Unable as error:
        return str(error)
    assert writes
    return writes


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
@pytest.mark.parametrize(
    "scene_class", WITNESSES, ids=lambda scene_class: scene_class.__name__
)
def test_every_write_is_the_same_with_and_without_reuse(scene_class, draw_path):
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    on = _outcome(scene_class, draw_path=draw_path)
    off = _outcome(scene_class, draw_path=draw_path, draw_reuse=False)
    if isinstance(on, str) or isinstance(off, str):
        assert on == off
    else:
        assert_same_writes(on, off)


#: Scenes whose frames are reused (``reused``), and scenes whose camera the
#: scene's code changed (a display function or a camera method replaced),
#: which reuse nothing.
DELIVERED = [
    (reuse_scenes.StillWait, True),
    (reuse_scenes.Holds, True),
    (reuse_scenes.ReplacedDisplay, False),
    (reuse_scenes.PatchedCamera, False),
]


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
@pytest.mark.parametrize(
    ("scene_class", "reuses"),
    DELIVERED,
    ids=lambda value: getattr(value, "__name__", None),
)
def test_a_reused_frame_is_delivered_as_a_drawn_one_is(scene_class, reuses, draw_path):
    """Reuse replaces the drawing of a frame's pixels and nothing else: the
    painter of every frame delivered (``painter.deliveries``, what a movie's
    one painter is kept by) is the same, frame for frame, with reuse and
    without, and so are the bytes. A camera whose frames are described from
    its own captures (``Drawing.captured``, on ``ir_cairo``) reuses none.
    """
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")
    with deliveries() as delivered_on:
        scene, on = render_writes(scene_class, draw_path=draw_path)
    with deliveries() as delivered_off:
        scene_off, off = render_writes(
            scene_class, draw_path=draw_path, draw_reuse=False
        )
    assert delivered_on, "frames were delivered"
    assert delivered_on == delivered_off
    assert_same_writes(on, off)
    assert scene_off.renderer.draws.reused == 0
    if reuses:
        assert scene.renderer.draws.reused > 0, "the witness reuses frames"
    else:
        assert scene.renderer.draws.reused == 0, "the camera's code ran at each draw"
