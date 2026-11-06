"""A camera's recorded inputs observed anew are the same as before when the
camera did not change, and are not when any one input did.
"""

from __future__ import annotations

import numpy as np

from arachne import Dot, Square
from arachne.camera.camera import Camera
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.timeline.draw import CameraExtras


def _observed(camera) -> CameraExtras:
    extras = CameraExtras.observe(camera, None)
    assert extras is not None
    return extras


def test_inputs_observed_anew_of_an_unchanged_camera_are_the_same():
    camera = ThreeDCamera()
    anchor = Dot()
    camera.add_fixed_in_frame_mobjects(Square())
    camera.add_fixed_orientation_mobjects(anchor)
    first, again = _observed(camera), _observed(camera)
    assert first is not again
    assert first.same(again)
    assert again.same(first)


def test_inputs_with_any_one_change_are_not_the_same():
    camera = ThreeDCamera()
    anchor = Dot()
    camera.add_fixed_orientation_mobjects(anchor)
    before = _observed(camera)

    camera.background_opacity = 0.5
    assert not before.same(_observed(camera))
    camera.background_opacity = 1.0
    assert before.same(_observed(camera))

    held = Square()
    camera.add_fixed_in_frame_mobjects(held)
    assert not before.same(_observed(camera))
    camera.remove_fixed_in_frame_mobjects(held)
    assert before.same(_observed(camera))

    anchor.shift(np.array([1.0, 0.0, 0.0]))
    assert not before.same(_observed(camera))


def test_a_supplied_background_is_the_same_only_as_itself():
    camera = Camera()
    before = _observed(camera)
    camera.set_background(np.zeros_like(camera.background))
    assert not before.same(_observed(camera))
