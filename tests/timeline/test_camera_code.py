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
    engine camera class, the table it leaves is not rewired (read once the
    painter's own marker is gone, as a camera outside a live capture is).
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
        vars(camera).pop("_xf_live")
        assert not display_funcs_rewired(camera), type(camera).__name__
        assert replaced_methods(camera) == ()
        vars(camera)["_xf_live"] = []
        restore()
        assert not display_funcs_rewired(camera)


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
    assert found == {"camera/camera.py", "frame_ir/live.py"}


def test_the_detector_imports_no_core():
    code = (
        "import sys, arachne\n"
        "from arachne.timeline.camera_code import replaced_methods\n"
        "assert replaced_methods(arachne.ThreeDCamera()) == ()\n"
        "assert 'arachne_core' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
