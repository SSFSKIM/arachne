"""The light the core draws a 3D scene in, as ``core_light`` and the soft
light's parameters name it: from a config file's ``[CLI]`` section or set at
run time, the soft light by default, CE's selectable, a value the core does
not take refused.
"""

from __future__ import annotations

import numpy as np
import pytest

from arachne import DEGREES, RED, Sphere, tempconfig
from arachne._config.utils import ManimConfig, make_config_parser
from arachne.camera.three_d_camera import ThreeDCamera
from arachne.frame_ir.canonical import RATIO_GRID
from arachne.frame_ir.core_painter import CorePainter, light_model

arachne_core = pytest.importorskip("arachne_core")

SMALL = {"pixel_width": 192, "pixel_height": 108}


def test_a_config_files_cli_section_names_the_light_and_its_parameters(tmp_path):
    cfg = tmp_path / "light.cfg"
    cfg.write_text(
        "[CLI]\ncore_light = ce\ncore_light_ambient = 0.5\ncore_light_wrap = 0.25\n"
    )
    named = ManimConfig().digest_file(str(cfg))
    assert named.core_light == "ce"
    assert (named.core_light_ambient, named.core_light_wrap) == (0.5, 0.25)


def test_the_default_is_the_soft_light_at_the_cores_own_parameters():
    default = ManimConfig().digest_parser(make_config_parser())
    assert default.core_light == "soft"
    assert (default.core_light_ambient, default.core_light_wrap) == (None, None)
    with tempconfig({"core_light": "soft"}):
        assert light_model() == (1, arachne_core.SOFT_DIFFUSE)


def test_a_parameter_set_at_run_time_is_the_light_the_core_is_handed():
    ambient, wrap = arachne_core.SOFT_DIFFUSE
    with tempconfig({"core_light_ambient": 0.6}):
        assert light_model() == (1, (0.6, wrap))
    with tempconfig({"core_light_wrap": "0"}):
        assert light_model() == (1, (ambient, 0.0))
    with tempconfig({"core_light": "ce", "core_light_ambient": 0.6}):
        assert light_model() is None


@pytest.mark.parametrize(
    ("key", "value"),
    [("core_light", "phong"), ("core_light_ambient", "bright")],
)
def test_a_light_or_parameter_the_config_does_not_know_is_refused(key, value):
    config = ManimConfig().digest_parser(make_config_parser())
    with pytest.raises(ValueError, match=key):
        setattr(config, key, value)


def _painted(**light):
    """A red sphere drawn on the core in ``light``: the delta it was handed
    and the frame.
    """
    with tempconfig({**SMALL, **light}):
        sphere = Sphere(resolution=(8, 16), fill_color=RED, checkerboard_colors=False)
        camera = ThreeDCamera()
        camera.set_phi(70 * DEGREES)
        camera.set_theta(-30 * DEGREES)
        painter = CorePainter()
        frame = painter.paint(camera, [("moving", [sphere])], roots=[sphere])
    return painter.last, frame


def _face_group(delta):
    (record,) = [record for _, record in delta.set if record.kind == "face_group"]
    return record.fields


def test_a_surface_in_the_soft_light_carries_its_corners_smooth_normals():
    sent, _ = _painted()
    assert sent.view["light_model"] == (1, arachne_core.SOFT_DIFFUSE)
    fields = _face_group(sent)
    assert "normals" not in fields
    normals = np.array(fields["corner_normals"])
    assert normals.shape == (4 * len(fields["uv"]), 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(normals, np.round(normals / RATIO_GRID) * RATIO_GRID)


def test_a_surface_in_ces_light_is_handed_ces_normals_and_no_light_model():
    sent, _ = _painted(core_light="ce")
    assert "light_model" not in sent.view
    fields = _face_group(sent)
    assert "corner_normals" not in fields
    assert "normals" in fields


def test_a_higher_ambient_share_lightens_the_side_away_from_the_light():
    _, dim = _painted(core_light_ambient=0.1)
    _, bright = _painted(core_light_ambient=0.9)
    lit = dim[..., 3] > 0
    assert lit.sum() > 1000
    assert bright[lit, 0].astype(int).sum() > dim[lit, 0].astype(int).sum()


def test_an_ambient_share_outside_its_range_is_refused_by_the_core():
    with pytest.raises(arachne_core.XfError, match=r"ambient share in 0\.\.=1"):
        _painted(core_light_ambient=1.5)
