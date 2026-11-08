"""A record draws a play's frames after the scene ran only when drawing
runs none of the scene's code: no method of a mobject's class or instance
the camera calls, no protocol method (``__hash__``, ``__iter__``) the
drawing invokes, and no such mobject joining mid-play. A play that meets
one mid-play has the scene executed again, drawing that play as it runs.
"""

from __future__ import annotations

import importlib
import inspect
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from arachne import (
    BLUE,
    LEFT,
    RED,
    RIGHT,
    DecimalNumber,
    Dot,
    MathTex,
    SVGMobject,
    config,
    tempconfig,
)
from arachne.camera.camera import Camera
from arachne.mobject.construction_caches import forget_construction_caches
from arachne.timeline import draw, recorder
from arachne.timeline.dispatch import draw_protocols
from arachne.timeline.routes import RecordRefused

from . import soundness_scenes
from .conftest import SMALL, render

SOUNDNESS_SCENES = Path(soundness_scenes.__file__)
#: The module, which the package hides behind a function of the same name.
record = importlib.import_module("arachne.timeline.record")

#: What the drawing code is read to invoke, at least.
_PROTOCOLS = (
    "__hash__",
    "__eq__",
    "__iter__",
    "__getitem__",
    "__len__",
    "__bool__",
    "__getattr__",
    "__getattribute__",
)
#: What a format or a message calls, and what making a class runs: no frame
#: depends on them.
_NOT_DRAWN = ("__repr__", "__str__", "__format__", "__init_subclass__")
#: What making an object or reading its class takes, not drawing it.
_NOT_INSTRUMENTED = frozenset(
    {"__class__", "__new__", "__init__", "__subclasshook__", "__init_subclass__"}
)


def test_the_draw_protocols_are_read_from_the_drawing_code():
    protocols = draw_protocols(Camera)
    for name in _PROTOCOLS:
        assert name in protocols, name
    for name in _NOT_DRAWN:
        assert name not in protocols, name


def _probe_class(called: set[str], recording: list[bool]) -> type:
    """A Dot whose every protocol method notes its name in ``called`` while
    ``recording[0]``, doing what Dot's does otherwise; those Dot leaves to
    Python's defaults (``__bool__``, ``__getattr__``, ``__contains__``,
    ``__index__``) do what the defaults do.
    """

    def noted(name, original):
        def method(self, *args, **kwargs):
            if recording[0]:
                called.add(name)
            return original(self, *args, **kwargs)

        return method

    members: dict[str, object] = {}
    for name in dir(Dot):
        if not (name.startswith("__") and name.endswith("__")):
            continue
        if name in _NOT_INSTRUMENTED:
            continue
        static = inspect.getattr_static(Dot, name)
        if isinstance(static, (staticmethod, classmethod, type)) or not callable(
            static
        ):
            continue
        members[name] = noted(name, getattr(Dot, name))

    def default_bool(self):
        return len(self) != 0

    def default_getattr(self, name):
        raise AttributeError(name)

    def default_contains(self, value):
        return any(item is value or item == value for item in iter(self))

    def default_index(self):
        raise TypeError("not an index")

    for name, original in (
        ("__bool__", default_bool),
        ("__getattr__", default_getattr),
        ("__contains__", default_contains),
        ("__index__", default_index),
    ):
        members.setdefault(name, noted(name, original))
    return type("Probe", (Dot,), members)


class _PlainDot(soundness_scenes.HashedDot):
    """``HashedDot`` with an engine Dot."""

    dot_class: type = Dot


def test_drawing_from_a_record_calls_no_protocol_outside_the_list():
    """Instrument every protocol method of a probe dot, draw it from the
    record (its refusal set aside), and list what drawing called.
    """
    called: set[str] = set()
    recording = [False]
    probed = type(
        "ProbedDot",
        (soundness_scenes.HashedDot,),
        {"dot_class": _probe_class(called, recording)},
    )
    enter, leave = draw.Drawing.__enter__, draw.Drawing.__exit__

    def entered(self):
        recording[0] = True
        return enter(self)

    def left(self, *args):
        recording[0] = False
        return leave(self, *args)

    with (
        mock.patch.object(draw.Drawing, "__enter__", entered),
        mock.patch.object(draw.Drawing, "__exit__", left),
        mock.patch.object(recorder, "draw_refusal", return_value=None),
        mock.patch.object(record, "draw_member_refusal", return_value=None),
    ):
        scene, frames = render(probed, 6)
    assert scene.record.refusal is None
    assert frames
    # The probe was drawn from the record, and drawing called its methods.
    assert called
    outside = called - draw_protocols(Camera) - set(_NOT_DRAWN)
    assert not outside, sorted(outside)


def test_a_class_defining_a_drawn_protocol_is_drawn_as_the_scene_runs(session_of):
    session = session_of("HashedDot", SOUNDNESS_SCENES)
    assert session.record.refusal.startswith("draw.user_override: HashDot.__hash__:")
    calls = session.module.HashDot.calls
    before = len(calls)
    with pytest.raises(RecordRefused, match="draw.user_override"):
        session.frame(Fraction(1, 3))
    with pytest.raises(RecordRefused):
        session.state_at(Fraction(1, 3))
    assert len(calls) == before
    _, frames = render(soundness_scenes.HashedDot, 6)
    _, plain = render(_PlainDot, 6)
    assert len(frames) == len(plain)
    for k, (frame, reference) in enumerate(zip(frames, plain, strict=True)):
        assert np.array_equal(frame, reference), k


def _drawn_serially(fps: int, plays: list[tuple[str, object]]) -> list[np.ndarray]:
    """Frames of ``play_fill_between(0.3, 0.6)`` plays, in the colour each
    names, and waits, drawn as CE draws them: the dot at each frame its
    tracker is inside, in its colour then.
    """
    cls = soundness_scenes.CountingFill
    frames: list[np.ndarray] = []
    with tempconfig(
        {**SMALL, "frame_rate": fps, "dry_run": True, "disable_caching": True}
    ):
        camera = soundness_scenes.TransientFill().renderer.camera
        camera.reset()
        blank = camera.pixel_array.copy()
        for kind, value in plays:
            if kind == "wait":
                frames.extend([blank] * round(fps * value))
                continue
            cls.colour = value
            for k in range(fps):
                if 0.3 <= k / fps < 0.6:
                    camera.reset()
                    camera.capture_mobjects([cls(radius=0.3)])
                    frames.append(camera.pixel_array.copy())
                else:
                    frames.append(blank)
    return frames


def _same_frames(frames, expected) -> None:
    assert len(frames) == len(expected)
    for k, (frame, reference) in enumerate(zip(frames, expected, strict=True)):
        assert np.array_equal(frame, reference), k


def test_a_mobject_that_joins_and_leaves_mid_play_is_drawn_as_the_scene_runs(
    session_of,
):
    session = session_of("TransientFill", SOUNDNESS_SCENES)
    assert session.record.refusal.startswith(
        "draw.user_override: CountingFill.get_fill_rgbas:"
    )
    calls = session.module.CountingFill.calls
    before = len(calls)
    with pytest.raises(RecordRefused, match="draw.user_override"):
        session.frame(Fraction(1, 2))
    assert len(calls) == before
    scene, frames = render(soundness_scenes.TransientFill, 6)
    execution = scene.record.execution
    assert execution.number == 2
    assert list(execution.known) == [0]
    assert execution.known[0].startswith("draw.user_override: CountingFill.")
    assert execution.divergence is None
    _same_frames(frames, _drawn_serially(6, [("fill", RED), ("wait", 0.5)]))


def test_every_play_from_the_first_refused_on_is_drawn_as_the_scene_runs():
    """Play 2 refuses as play 0 does, and is drawn as it runs in the same
    second execution.
    """
    scene, frames = render(soundness_scenes.TwoTransientFills, 6)
    execution = scene.record.execution
    assert execution.number == 2
    assert sorted(execution.known) == [0]
    assert execution.live_from == 0
    assert execution.divergence is None
    _same_frames(
        frames,
        _drawn_serially(6, [("fill", RED), ("wait", 0.5), ("fill", BLUE)]),
    )


def test_a_refusal_found_at_another_play_each_execution_costs_one_execution():
    """One more wait each run moves the one refusal a play later each
    execution: drawn as it runs from the first index found on, it costs one
    extra execution, not one per index.
    """
    runs = soundness_scenes.ShiftingTransientFill.runs
    runs.clear()
    scene, frames = render(soundness_scenes.ShiftingTransientFill, 6)
    execution = scene.record.execution
    distinct_refusals = 1
    assert execution.number == 1 + distinct_refusals
    assert len(runs) == execution.number
    assert list(execution.known) == [1]
    # This execution ran two waits before its fill.
    _same_frames(
        frames,
        _drawn_serially(6, [("wait", 0.5), ("wait", 0.5), ("fill", RED)]),
    )


def test_a_scene_catching_its_errors_is_executed_again_all_the_same():
    scene, frames = render(soundness_scenes.CaughtTransientFill, 6)
    assert scene.record.execution.number == 2
    _same_frames(frames, _drawn_serially(6, [("fill", RED), ("wait", 0.5)]))


def test_an_execution_that_begins_a_play_from_another_state_says_so():
    soundness_scenes.DriftingTransientFill.runs.clear()
    scene, _ = render(soundness_scenes.DriftingTransientFill, 6)
    execution = scene.record.execution
    assert execution.number == 2
    assert execution.divergence == "play 0 began from another state than in execution 1"


def test_a_retry_tears_down_only_the_execution_that_completes():
    """CE tears down no run that raised: the interrupted execution is set up
    and never torn down, and the one that completes is torn down once.
    """
    cls = soundness_scenes.SetUpTransientFill
    cls.steps.clear()
    cls.open.clear()
    scene, frames = render(cls, 6)
    assert scene.record.execution.number == 2
    assert scene.record.execution.known[0].startswith("draw.user_override:")
    assert cls.steps == ["setup", "setup", "tear_down"]
    assert cls.open == [1]
    _same_frames(frames, _drawn_serially(6, [("fill", RED)]))


class ConfiguringTransientFill(soundness_scenes.TransientFill):
    """``TransientFill`` setting the configured background colour first,
    which a camera is made with: the second execution's camera is made as the
    first's was.
    """

    def construct(self):
        config.background_color = BLUE
        super().construct()


def test_an_execution_again_finds_the_configuration_its_first_execution_found():
    scene, frames = render(ConfiguringTransientFill, 6)
    assert scene.record.execution.number == 2
    _same_frames(frames, _drawn_serially(6, [("fill", RED), ("wait", 0.5)]))


@pytest.mark.xfail(
    raises=AttributeError,
    strict=True,
    reason=(
        "a retry initialises the scene anew from its constructor arguments:"
        " an attribute its caller set between making it and rendering it"
        " (scene.anchor) is lost (debt)"
    ),
)
def test_a_retry_keeps_what_the_caller_set_on_the_scene():
    with tempconfig(
        {**SMALL, "frame_rate": 6, "dry_run": True, "disable_caching": True}
    ):
        scene = soundness_scenes.AnchoredTransientFill()
        scene.anchor = np.array([1.0, 0.0, 0.0])
        scene.render()
    assert scene.record.execution.number == 2


SVG = (
    Path(__file__).parents[1]
    / "test_graphical_units"
    / "img_svg_resources"
    / "arcs01.svg"
)


class ImageThenTransientFill(soundness_scenes.TransientFill):
    """``TransientFill`` after an image: executed again, it builds the image
    from the cache as its first execution did.
    """

    def construct(self):
        self.add(SVGMobject(SVG).scale(0.3))
        super().construct()


def test_an_execution_again_finds_the_images_its_first_execution_found():
    """One process renders several scenes, an earlier one leaving its images
    cached: a scene executed again builds its images as its first execution
    did, from the cache as that execution found it.
    """
    forget_construction_caches()
    for _ in range(2):  # the first parses the image, the second finds it cached
        scene, _ = render(ImageThenTransientFill, 6)
        execution = scene.record.execution
        assert execution.number == 2
        assert execution.divergence is None


class NumberThenFormulaThenTransientFill(soundness_scenes.TransientFill):
    """``TransientFill`` after a number and a formula whose image the number
    left cached: executed again, the formula is built as its first execution
    built it.
    """

    def construct(self):
        self.add(DecimalNumber(1, num_decimal_places=0).shift(LEFT))
        formula = MathTex("1").shift(RIGHT)
        formula.set_sheen(0.75, formula.get_sheen_direction())
        self.add(formula)
        super().construct()


def test_an_execution_again_finds_the_glyphs_and_images_its_first_execution_found():
    """A number fills the glyph cache and the image cache together: executed
    again, a scene starts from both as its first execution did.
    """
    forget_construction_caches()
    for _ in range(2):  # the first fills both caches, the second finds them filled
        scene, _ = render(NumberThenFormulaThenTransientFill, 6)
        execution = scene.record.execution
        assert execution.number == 2
        assert execution.divergence is None
