"""What a camera holds, compared around a call its code makes while drawing
(:func:`arachne.frame_ir.live._changed`): the change that makes its draw the
camera's own. Nothing is drawn here.
"""

from __future__ import annotations

import numpy as np

from arachne import ThreeDCamera
from arachne.frame_ir.live import _camera_state, _changed
from arachne.mobject.values import FrozenArray, adopt


def _around(camera, change, *, checksums=True):
    before = _camera_state(camera, "three_d", checksums)
    change(camera)
    return _changed(before, _camera_state(camera, "three_d", checksums))


def test_a_camera_left_alone_has_nothing_changed():
    assert _around(ThreeDCamera(), lambda camera: None) is None


def test_a_number_changed_and_an_entry_added_are_named():
    def wider(camera):
        camera.cairo_line_width_multiple *= 2

    assert _around(ThreeDCamera(), wider) == "cairo_line_width_multiple"

    def added(camera):
        camera.rgb_max_val = 160

    assert _around(ThreeDCamera(), added) == "rgb_max_val"


def test_a_value_changed_in_place_through_what_holds_it_is_named():
    def turned(camera):
        camera.phi_tracker.set_value(1.0)

    assert _around(ThreeDCamera(), turned) == "phi"

    def edited(camera):
        camera.rotation_matrix[0, 0] += 1.0

    assert _around(ThreeDCamera(), edited) == "rotation_matrix"


def test_nan_and_numpy_scalars_compare_by_value():
    camera = ThreeDCamera()
    camera.unset = float("nan")
    camera.scale = np.float64(2.0)

    def rebound(camera):
        camera.unset = float("nan")
        camera.scale = np.float64(2.0)

    assert _around(camera, rebound) is None

    def rescaled(camera):
        camera.scale = np.float64(3.0)

    assert _around(camera, rescaled) == "scale"


def test_a_tuple_rebound_to_an_equal_value_is_named():
    # Containers compare their members by identity: a rebinding is loud.
    camera = ThreeDCamera()
    camera.pair = (1.0, [2.0])

    def rebound(camera):
        camera.pair = (1.0, [2.0])

    assert _around(camera, rebound) == "pair"


def test_a_frozen_array_is_told_by_identity():
    camera = ThreeDCamera()
    camera.held = adopt(np.zeros(4), FrozenArray)

    def rebound(camera):
        camera.held = adopt(np.zeros(4), FrozenArray)

    assert _around(camera, lambda camera: None) is None
    assert _around(camera, rebound) == "held"


def test_a_large_array_written_into_is_seen_through_its_checksum():
    def drawn_on(camera):
        camera.pixel_array[0, 0, 0] ^= 0xFF

    assert _around(ThreeDCamera(), drawn_on) == "pixel_array"
    # Where the real draw is compared with the description, it is not
    # checksummed call by call.
    assert _around(ThreeDCamera(), drawn_on, checksums=False) is None
