"""The draw-reads tracer (``tests/helpers/draw_reads.py``) itself: it finds a
read of what the record does not hold, made any way Python reads an
attribute, names its type, attribute and line, and changes nothing the
engine draws. The suite runs under it with ``ARACHNE_DRAW_READS`` set.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from arachne import LEFT, RIGHT, Circle, Create, Dot, Scene, Square, VMobject, config
from arachne._config.utils import ManimConfig
from arachne.camera.camera import Camera
from arachne.frame_ir.cairo_painter import CairoDrawer
from arachne.mobject.mobject import Mobject
from arachne.timeline.record import Record
from tests.helpers import draw_reads

from .conftest import render

HERE = "tests/timeline/test_draw_reads.py"

#: A mobject no scene holds, which the drawings below read.
OUTSIDE = Dot(LEFT)


@pytest.fixture
def tracing():
    """The process's tracer when the suite runs under one, else one of the
    test's own, removed after it.
    """
    shared = draw_reads._TRACER
    tracer = shared
    if tracer is None:
        tracer = draw_reads.Tracer()
        tracer.install()
    tracer.take()
    yield tracer
    tracer.take()
    if shared is None:
        tracer.uninstall()


def _replace_display(scene: Scene, reads) -> None:
    """Make the scene's camera call ``reads()`` each time it draws a
    vectorized mobject (a method replaced on the instance: the scene's own
    code, whose reads of the engine are traced as any are).
    """
    camera = scene.camera
    display = camera.display_vectorized

    def displaying(vmobject, ctx):
        reads()
        return display(vmobject, ctx)

    camera.display_vectorized = displaying


class ReadsOutside(Scene):
    def construct(self):
        def reads():
            OUTSIDE.points  # noqa: B018  # a property, reading _points
            getattr(OUTSIDE, "z_index")  # noqa: B009
            vars(OUTSIDE)

        _replace_display(self, reads)
        self.play(Create(Square()), run_time=0.2)


class ReadsAnOption(Scene):
    def construct(self):
        _replace_display(self, lambda: config.frame_rate)
        self.play(Create(Square()), run_time=0.2)


class ReplacesADisplayFunction(Scene):
    def construct(self):
        own = self.camera.display_funcs[VMobject]
        self.camera.display_funcs[VMobject] = lambda batch, pixels: own(batch, pixels)
        self.play(Create(Square()), run_time=0.2)


class ReplacesADisplayFunctionAndReadsAnOption(ReplacesADisplayFunction):
    """``ReplacesADisplayFunction``, and a read of an option the record does
    not hold, made by the scene's code in the camera's method of its own.
    """

    def construct(self):
        _replace_display(self, lambda: config.frame_rate)
        super().construct()


class FilledFromAnImage(Scene):
    def construct(self):
        circle = Circle(fill_opacity=1)
        circle.color_using_background_image(Image.new("RGBA", (4, 4), (255, 0, 0, 255)))
        self.play(circle.animate.shift(RIGHT), run_time=0.2)


class Plain(Scene):
    def construct(self):
        square, circle = Square(), Circle().shift(RIGHT)
        self.add(square)
        self.play(Create(circle), square.animate.shift(LEFT), run_time=0.4)
        self.wait(0.2)


class PlainKeepingItsTable(Plain):
    """``Plain``, keeping the camera's table of display functions as the
    scene made it.
    """

    def construct(self):
        self.table = self.camera.display_funcs
        super().construct()


def _found(tracer) -> set[tuple[str, str, str]]:
    return {(read.owner, read.attribute, read.why) for read in tracer.take()}


def test_a_read_of_a_mobject_the_record_does_not_hold_is_found_however_made(tracing):
    render(ReadsOutside, 15)
    reads = tracing.take()
    dot = "arachne.mobject.geometry.arc.Dot"
    by_name = {read.attribute: read for read in reads if read.owner == dot}
    # Through a property's getter, getattr, and vars().
    assert {"_points", "z_index", "__dict__"} <= set(by_name)
    for name in ("z_index", "__dict__"):
        assert by_name[name].site.startswith(HERE), by_name[name]
        assert by_name[name].why == "a mobject the record does not hold"
    assert by_name["_points"].site.startswith("arachne/mobject/mobject.py:")
    assert f"via {HERE}" not in by_name["_points"].site
    # Nothing the record holds is found: only the outside dot was read so.
    assert {read.owner for read in reads} == {dot}


def test_an_option_no_entry_lists_is_found(tracing):
    render(ReadsAnOption, 15)
    assert (
        "arachne._config.utils.ManimConfig",
        "frame_rate",
        "a configuration option no list entry covers",
    ) in _found(tracing)


_ENTRY = (
    "arachne.camera.camera.Camera",
    "display_funcs[VMobject]",
    "a display function the engine did not define",
)


def test_a_display_function_the_scene_replaced_is_not_found_once_the_record_reports_it(
    tracing,
):
    scene, _ = render(ReplacesADisplayFunction, 15)
    assert [r.name for r in scene.timeline.record.camera_replaced] == [
        "display_funcs[VMobject]"
    ]
    assert _ENTRY not in _found(tracing)


def test_a_display_function_the_record_does_not_report_is_found(tracing, monkeypatch):
    from arachne.timeline import camera_code

    monkeypatch.setattr(camera_code, "_replaced_displays", lambda camera: [])
    scene, _ = render(ReplacesADisplayFunction, 15)
    assert scene.timeline.record.camera_replaced == ()
    assert _ENTRY in _found(tracing)


def test_the_exemption_of_a_reported_display_function_covers_that_entry_only(tracing):
    render(ReplacesADisplayFunctionAndReadsAnOption, 15)
    found = _found(tracing)
    assert _ENTRY not in found
    assert (
        "arachne._config.utils.ManimConfig",
        "frame_rate",
        "a configuration option no list entry covers",
    ) in found


def test_an_image_the_record_keeps_by_reference_is_found(tracing, monkeypatch):
    render(FilledFromAnImage, 15)
    assert tracing.take() == []
    # Drawn with the scene's own image, not the cameras' reads of it.
    monkeypatch.setattr(Record, "models_image", lambda record, value: False)
    render(FilledFromAnImage, 15)
    assert (
        "arachne.mobject.geometry.arc.Circle",
        "background_image",
        "an image the record keeps by reference, which can be painted in place",
    ) in _found(tracing)


def test_a_scene_the_record_holds_whole_reads_nothing_unlisted(tracing):
    before = tracing.traced
    render(Plain, 15)
    assert tracing.traced > before + 9
    assert tracing.take() == []


#: What a traced frame hooks, as the engine defines it: each must be back
#: once the frame is done.
_UNHOOKED = {
    (Mobject, "__getattribute__"): None,
    (Camera, "__getattribute__"): None,
    (ManimConfig, "__getattribute__"): None,
    (Mobject, "__setattr__"): None,
    (Mobject, "__delattr__"): None,
    (ManimConfig, "__getitem__"): vars(ManimConfig)["__getitem__"],
}


def _assert_unhooked(scene) -> None:
    """Every hook a traced frame installs is gone, the scene's camera holding
    its own table of display functions again. (The tracer's audit hook
    stays, as every audit hook does, and sees nothing once no frame is
    traced.)
    """
    for (cls, name), original in _UNHOOKED.items():
        assert vars(cls).get(name) is original, (cls, name)
    assert vars(scene.camera)["display_funcs"] is scene.table


def _skip_without_core(draw_path: str) -> None:
    if draw_path == "ir_cairo":
        pytest.importorskip("arachne_core")


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_tracing_changes_no_frame_and_leaves_no_hook(tracing, draw_path):
    _skip_without_core(draw_path)
    tracing.uninstall()
    try:
        _, untraced = render(PlainKeepingItsTable, 15, draw_path=draw_path)
    finally:
        tracing.install()
    before = tracing.traced
    scene, traced = render(PlainKeepingItsTable, 15, draw_path=draw_path)
    assert tracing.traced > before
    assert len(traced) == len(untraced)
    for k, (a, b) in enumerate(zip(traced, untraced, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=str(k))
    _assert_unhooked(scene)


class _Failed(Exception):
    pass


@pytest.mark.parametrize("draw_path", ["cairo", "ir_cairo"])
def test_a_traced_frame_that_raises_leaves_no_hook(tracing, draw_path, monkeypatch):
    _skip_without_core(draw_path)
    made: list = []
    inside: list[bool] = []

    def init(scene, *args, **kwargs):
        made.append(scene)
        original_init(scene, *args, **kwargs)

    def fail(*args, **kwargs):
        inside.append(bool(tracing.frames))
        raise _Failed

    original_init = PlainKeepingItsTable.__init__
    monkeypatch.setattr(PlainKeepingItsTable, "__init__", init)
    if draw_path == "cairo":
        monkeypatch.setattr(Camera, "capture_mobjects", fail)
    else:
        monkeypatch.setattr(CairoDrawer, "draw", fail)
    with pytest.raises(_Failed):
        render(PlainKeepingItsTable, 15, draw_path=draw_path)
    assert inside == [True]
    assert tracing.frames == []
    _assert_unhooked(made[-1])
