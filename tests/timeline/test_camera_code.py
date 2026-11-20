"""The camera methods a scene's code replaced, found from the camera alone."""

from __future__ import annotations

import subprocess
import sys

from arachne import (
    Camera,
    ImageMobjectFromCamera,
    MovingCamera,
    MultiCamera,
    PMobject,
    Square,
    ThreeDCamera,
    VMobject,
)
from arachne.timeline.camera_code import (
    Replaced,
    display_funcs_rewired,
    drawn_cameras,
    engine_override,
    engine_own,
    replaced_methods,
)


def test_a_frame_is_drawn_with_its_camera_and_every_display_s_camera():
    inner = MovingCamera()
    outer = MovingCamera()
    nested = MultiCamera()
    nested.add_image_mobject_from_camera(ImageMobjectFromCamera(inner))
    main = MultiCamera()
    main.add_image_mobject_from_camera(ImageMobjectFromCamera(outer))
    main.add_image_mobject_from_camera(ImageMobjectFromCamera(nested))
    # Shown twice: one camera.
    main.add_image_mobject_from_camera(ImageMobjectFromCamera(outer))
    assert drawn_cameras(main) == (main, outer, nested, inner)
    camera = Camera()
    assert drawn_cameras(camera) == (camera,)


def _reversed(method):
    def get_mobjects_to_display(self, *args, **kwargs):
        return list(reversed(method(self, *args, **kwargs)))

    return get_mobjects_to_display


def test_a_camera_as_the_engine_made_it_has_nothing_replaced():
    assert replaced_methods(Camera()) == ()
    assert replaced_methods(ThreeDCamera()) == ()


def test_a_callable_the_instance_holds_in_a_method_s_place_is_found():
    camera = ThreeDCamera()
    shown = _reversed(Camera.get_mobjects_to_display).__get__(camera)
    camera.get_mobjects_to_display = shown
    camera.not_a_method = print  # no method of the class is shadowed
    assert replaced_methods(camera) == (
        Replaced("get_mobjects_to_display", "instance", ThreeDCamera, shown),
    )
    assert replaced_methods(camera)[0].qualname == (
        "ThreeDCamera.get_mobjects_to_display"
    )


def test_a_method_replaced_on_an_engine_camera_class_is_found(monkeypatch):
    patch = _reversed(Camera.get_mobjects_to_display)
    monkeypatch.setattr(Camera, "get_mobjects_to_display", patch)
    assert replaced_methods(ThreeDCamera()) == (
        Replaced("get_mobjects_to_display", "class", Camera, patch),
    )


def test_a_method_deleted_from_an_engine_camera_class_is_found(monkeypatch):
    monkeypatch.delattr(ThreeDCamera, "project_points")
    assert replaced_methods(ThreeDCamera()) == (
        Replaced("project_points", "class", ThreeDCamera, None),
    )


def test_an_entry_of_the_display_functions_the_scene_replaced_is_found():
    from arachne import VMobject

    camera = ThreeDCamera()
    shown = camera.display_funcs[VMobject]

    def display(mobjects, pixel_array):
        shown(mobjects, pixel_array)

    camera.display_funcs[VMobject] = display
    assert replaced_methods(camera) == (
        Replaced("display_funcs[VMobject]", "display_funcs", ThreeDCamera, display),
    )
    assert replaced_methods(camera)[0].qualname == (
        "ThreeDCamera.display_funcs[VMobject]"
    )


def test_a_display_function_bound_to_another_camera_is_replaced():
    from arachne import VMobject

    camera = Camera()
    camera.display_funcs[VMobject] = Camera().display_funcs[VMobject]
    (found,) = replaced_methods(camera)
    assert found.name == "display_funcs[VMobject]"


def test_a_table_of_display_functions_the_scene_replaced_whole_is_found():
    camera = Camera()
    table = {}
    camera.display_funcs = table
    assert replaced_methods(camera) == (
        Replaced("display_funcs", "display_funcs", Camera, table),
    )


def test_what_is_no_engine_camera_has_nothing_replaced():
    class Duck:
        def capture_mobjects(self, mobjects):
            pass

    duck = Duck()
    duck.capture_mobjects = print
    assert replaced_methods(duck) == ()


def test_display_funcs_as_the_engine_made_them_are_not_rewired():
    for camera in (Camera(), ThreeDCamera(), MovingCamera(), MultiCamera()):
        assert not display_funcs_rewired(camera)


def test_display_funcs_holding_the_scene_s_drawing_are_rewired():
    camera = ThreeDCamera()
    paint = camera.display_funcs[VMobject]

    def bordered(mobjects, pixel_array):
        paint(mobjects, pixel_array)

    camera.display_funcs[VMobject] = bordered
    assert display_funcs_rewired(camera)


def test_display_funcs_holding_another_camera_s_method_are_rewired():
    camera, other = ThreeDCamera(), ThreeDCamera()
    camera.display_funcs[VMobject] = other.display_multiple_vectorized_mobjects
    assert display_funcs_rewired(camera)


def test_display_funcs_with_a_kind_added_or_removed_are_rewired():
    camera = Camera()
    del camera.display_funcs[PMobject]
    assert display_funcs_rewired(camera)
    camera = Camera()
    camera.display_funcs[Square] = camera.display_multiple_vectorized_mobjects
    assert display_funcs_rewired(camera)


def test_a_function_that_says_it_wraps_the_engine_s_is_the_scene_s():
    """The engine's wrappers are found by the function objects it made, never
    by what a function says about itself: a scene's function can carry a
    ``__wrapped__`` and the engine's marker.
    """
    camera = Camera()
    paint = camera.display_funcs[VMobject]

    def wrapped(mobjects, pixel_array):
        return paint(mobjects, pixel_array)

    wrapped.__wrapped__ = paint
    camera.display_funcs[VMobject] = wrapped
    assert display_funcs_rewired(camera)
    camera.display_funcs[VMobject] = engine_own(wrapped)
    assert display_funcs_rewired(camera)
    (found,) = replaced_methods(camera)
    assert found.name == "display_funcs[VMobject]"


def test_a_wrapper_the_engine_made_stands_for_what_it_wraps():
    camera = Camera()
    paint = camera.display_funcs[VMobject]

    def wrapper(mobjects, pixel_array):
        return paint(mobjects, pixel_array)

    camera.display_funcs[VMobject] = engine_override(wrapper, paint)
    assert not display_funcs_rewired(camera)
    assert replaced_methods(camera) == ()
    # Over an entry that was not the engine's, it is not either.
    other = Camera()
    foreign = other.display_funcs[VMobject]
    camera.display_funcs[VMobject] = engine_override(wrapper, foreign)
    assert display_funcs_rewired(camera)


def test_every_wrapper_the_live_painter_installs_reads_as_the_engine_s():
    """The one place the engine puts wrappers in ``display_funcs`` is
    ``LivePainter._install``, for each camera it watches. Built for each
    engine camera class, the table it leaves is not rewired.
    """
    from arachne.frame_ir.live import LivePainter

    for camera in (Camera(), ThreeDCamera(), MovingCamera(), MultiCamera()):
        restore = LivePainter(draw=False)._install(camera)
        wrappers = [
            func
            for kind, func in camera.display_funcs.items()
            if kind.__name__ != "Mobject"
        ]
        assert len(wrappers) == 3
        assert not any(
            getattr(func, "__self__", None) is camera for func in wrappers
        ), "the entries were wrapped"
        assert not display_funcs_rewired(camera), type(camera).__name__
        assert replaced_methods(camera) == ()
        restore()
        assert not display_funcs_rewired(camera)


def test_an_entry_the_scene_put_under_the_live_painter_s_wrapper_is_still_flagged():
    """The wrapper stands for the entry it wraps, whoever's that is: a scene's
    function under it is the scene's.
    """
    from arachne.frame_ir.live import LivePainter

    camera = Camera()
    paint = camera.display_funcs[VMobject]

    def bordered(mobjects, pixel_array):
        paint(mobjects, pixel_array)

    camera.display_funcs[VMobject] = bordered
    restore = LivePainter(draw=False)._install(camera)
    try:
        assert "_xf_live" in vars(camera)
        assert display_funcs_rewired(camera)
        (found,) = replaced_methods(camera)
        assert found.name == "display_funcs[VMobject]"
        assert found.method is bordered
    finally:
        restore()


def test_the_places_that_put_functions_in_display_funcs_are_the_ones_known():
    """A new place that installs a display function must teach the detector
    its wrapper (:func:`engine_override`) and join this list.
    """
    import re
    from pathlib import Path

    import arachne

    root = Path(arachne.__file__).parent
    assign = re.compile(r"display_funcs(\[[^\]]*\])?(:[^=]*)?\s*=[^=]")
    found = {
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if assign.search(path.read_text())
    }
    # compose/stack.py gives a window's camera of its own size the camera's
    # own painters, rebound to it, and installs no wrapper.
    assert found == {"camera/camera.py", "compose/stack.py", "frame_ir/live.py"}


def test_the_detector_imports_no_core():
    code = (
        "import sys, arachne\n"
        "from arachne.timeline.camera_code import replaced_methods\n"
        "assert replaced_methods(arachne.ThreeDCamera()) == ()\n"
        "assert 'arachne_core' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


#: Scenes of a sized window, as the scene's camera sees it with nothing
#: replaced, with a method replaced on its instance, and with an entry of its
#: table of display functions replaced; and the same replacements without a
#: window.
WINDOWED = """
from arachne import *


def _replace_method(camera):
    held = camera.display_vectorized

    def display_vectorized(vmobject, ctx):
        return held(vmobject, ctx)

    camera.display_vectorized = display_vectorized


def _replace_entry(camera):
    paint = camera.display_funcs[VMobject]

    def bordered(mobjects, pixel_array):
        return paint(mobjects, pixel_array)

    camera.display_funcs[VMobject] = bordered


class _Shows(Scene):
    windowed = True
    replace = None

    def setup(self):
        if self.replace is not None:
            globals()[self.replace](self.renderer.camera)

    def construct(self):
        square = Square()
        shown = Window(square, width=4, height=2.25) if self.windowed else square
        self.add(shown)
        self.play(shown.animate.shift(LEFT))


class Window_(_Shows):
    pass


class WindowMethod(_Shows):
    replace = "_replace_method"


class AloneMethod(WindowMethod):
    windowed = False


class WindowEntry(_Shows):
    replace = "_replace_entry"


class AloneEntry(WindowEntry):
    windowed = False
"""


def _opened(tmp_path, scene: str):
    from arachne.session import Session

    path = tmp_path / "windowed.py"
    path.write_text(WINDOWED)
    return Session.open(
        path,
        scene,
        config={
            "pixel_width": 96,
            "pixel_height": 54,
            "frame_rate": 4,
            "media_dir": str(tmp_path / "media"),
        },
    )


def _names(found) -> list[tuple[str, str]]:
    return [(replaced.name, replaced.form) for replaced in found]


def test_a_window_s_camera_at_its_own_size_replaces_nothing(tmp_path):
    """The camera a window is drawn with at its own size holds the scene
    camera's own painters, rebound to it: nothing of it is taken for the
    scene's code, and the record of the scene notes nothing replaced.
    """
    from arachne.compose.stack import _camera_of_size

    session = _opened(tmp_path, "Window_")
    try:
        assert session.record.camera_replaced == ()
        inner = _camera_of_size(session.scene.renderer.camera, (40, 22))
        assert replaced_methods(inner) == ()
        assert not display_funcs_rewired(inner)
    finally:
        session.close()


def test_a_replacement_the_scene_made_is_found_with_a_window_as_without(tmp_path):
    """A method the scene's code replaced on its camera, and an entry of its
    table of display functions, are found as they are without a window: in
    the record, on the scene's camera, and on the camera the window is drawn
    with at its own size, which draws through them too.
    """
    from arachne.compose.stack import _camera_of_size

    for kind in ("Method", "Entry"):
        alone = _opened(tmp_path, f"Alone{kind}")
        try:
            recorded = _names(alone.record.camera_replaced)
            live = _names(replaced_methods(alone.scene.renderer.camera))
        finally:
            alone.close()
        assert live, kind
        windowed = _opened(tmp_path, f"Window{kind}")
        try:
            camera = windowed.scene.renderer.camera
            assert _names(windowed.record.camera_replaced) == recorded, kind
            assert _names(replaced_methods(camera)) == live, kind
            inner = _camera_of_size(camera, (40, 22))
            assert _names(replaced_methods(inner)) == live, kind
        finally:
            windowed.close()
