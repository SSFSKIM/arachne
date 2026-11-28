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


def test_a_wrapper_the_engine_installs_stands_for_what_it_wraps():
    camera = Camera()
    paint = camera.display_funcs[VMobject]

    def wrapped(mobjects, pixel_array):
        return paint(mobjects, pixel_array)

    wrapped.__wrapped__ = paint
    unmarked = Camera()
    unmarked.display_funcs[VMobject] = wrapped
    assert display_funcs_rewired(unmarked)
    camera.display_funcs[VMobject] = engine_own(wrapped)
    assert not display_funcs_rewired(camera)


def test_the_detector_imports_no_core():
    code = (
        "import sys, arachne\n"
        "from arachne.timeline.camera_code import replaced_methods\n"
        "assert replaced_methods(arachne.ThreeDCamera()) == ()\n"
        "assert 'arachne_core' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
