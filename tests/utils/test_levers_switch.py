"""The speed levers' off switch, and the frames comparison that draws every
graphical unit both ways.

``speed_levers = False`` makes each speed lever read as its scalar,
CE-style setting, whatever it is set to; that is the reference a levers-on
render is compared with. With ``ARACHNE_FRAMES_LEVERS_OFF=1``
(``frames_comparison.LEVERS_OFF_ENV``) each frames comparison draws its
scene again with the levers off and requires each frame's bytes to equal
the first drawing's; ``tests/frame_ir/test_units.py`` runs every graphical
unit that way.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from arachne import BLUE, RED, Circle, Create, Scene, Square, config, tempconfig
from arachne._config.utils import SPEED_LEVERS_OFF, ManimConfig, make_config_parser
from arachne.cli.render.commands import render
from arachne.renderer.cairo_renderer import CairoRenderer
from arachne.utils.testing.frames_comparison import (
    LEVERS_OFF_ENV,
    _config_test,
    _FrameDigests,
    _make_test_comparing_frames,
)

#: Every lever set away from its off setting.
ALL_ON = {
    "vectorize_trusted": True,
    "construction_memo": True,
    "draw_reuse": True,
    "hold_gaps": True,
    "render_workers": 4,
}


def test_every_lever_has_an_off_setting():
    levers = set(SPEED_LEVERS_OFF)
    assert levers == {*ALL_ON, "typeset_dir"}
    for name in levers:
        assert isinstance(getattr(ManimConfig, name), property)


def test_the_switch_off_reads_every_lever_as_off(config, tmp_path):
    with tempconfig({**ALL_ON, "typeset_dir": tmp_path, "speed_levers": False}):
        for name, off in SPEED_LEVERS_OFF.items():
            assert config[name] == off, name
            assert getattr(config, name) == off, name


def test_the_switch_keeps_each_setting_for_when_it_is_on_again(config, tmp_path):
    with tempconfig({**ALL_ON, "typeset_dir": tmp_path}):
        config.speed_levers = False
        config.speed_levers = True
        for name, value in ALL_ON.items():
            assert getattr(config, name) == value, name
        assert config.typeset_dir == tmp_path


def test_a_lever_set_while_the_switch_is_off_still_reads_off(config):
    with tempconfig({"speed_levers": False}):
        config.hold_gaps = True
        config.render_workers = "auto"
        assert config.hold_gaps is False
        assert config.render_workers == 1


def test_a_typeset_dir_the_environment_names_reads_none_with_the_switch_off(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARACHNE_TYPESET_DIR", str(tmp_path))
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text("[CLI]\nspeed_levers = False\n")
    assert ManimConfig().digest_file(cfg).typeset_dir is None


def test_the_switch_off_on_the_command_line_wins_over_lever_flags(tmp_path):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    result = CliRunner().invoke(
        render,
        [
            str(scene_file),
            "--jupyter",
            "--no-speed-levers",
            "--workers",
            "4",
            "--hold-gaps",
            "--typeset_dir",
            str(tmp_path),
        ],
        standalone_mode=False,
    )
    assert result.exception is None, result.output
    candidate = ManimConfig().digest_parser(make_config_parser())
    candidate.digest_args(result.return_value)
    for name, off in SPEED_LEVERS_OFF.items():
        assert getattr(candidate, name) == off, name


# The frames comparison, both ways.


def _comparison(tmp_path, construct, last_frame):
    """A frames comparison of ``construct`` that writes its control frames
    into ``tmp_path`` (``--set_test``), under the graphical units' own
    configuration.
    """
    settings = _config_test(last_frame)
    settings["text_dir"] = tmp_path
    settings["tex_dir"] = tmp_path
    if last_frame:
        settings["frame_rate"] = 1
        settings["dry_run"] = True
    test = _make_test_comparing_frames(
        file_path=tmp_path / "control.npz",
        base_scene=Scene,
        construct=construct,
        renderer_class=CairoRenderer,
        is_set_test_data_test=True,
        last_frame=last_frame,
        show_diff=False,
        size_frame=(settings["pixel_height"], settings["pixel_width"]),
    )

    def run():
        with tempconfig(settings):
            test()

    return run


def _lever_dependent(seen):
    """A unit whose last frame shows whether the speed levers were on."""

    def construct(scene):
        seen.append(config.speed_levers)
        scene.add(Square())
        scene.play(Create(Circle(color=RED if config.vectorize_trusted else BLUE)))

    return construct


def _plain(seen):
    def construct(scene):
        seen.append(config.speed_levers)
        scene.add(Square())
        scene.play(Create(Circle()))

    return construct


@pytest.mark.parametrize("last_frame", [True, False])
def test_with_the_switch_the_comparison_draws_again_with_the_levers_off(
    tmp_path, monkeypatch, last_frame
):
    monkeypatch.setenv(LEVERS_OFF_ENV, "1")
    seen = []
    _comparison(tmp_path, _plain(seen), last_frame)()
    assert seen == [True, False]


@pytest.mark.parametrize("last_frame", [True, False])
def test_a_unit_that_draws_otherwise_with_the_levers_off_fails(
    tmp_path, monkeypatch, last_frame
):
    monkeypatch.setenv(LEVERS_OFF_ENV, "1")
    seen = []
    with pytest.raises(AssertionError, match="speed levers off differ"):
        _comparison(tmp_path, _lever_dependent(seen), last_frame)()
    assert seen == [True, False]


def test_a_frame_written_once_for_a_run_compares_as_the_run_frame_by_frame():
    """The comparison is over the frame grid: a frame written with
    ``repeat=3`` compares equal to that frame written three times, so a lever
    that writes a run of equal frames at once is no difference.
    """
    frame = np.zeros((2, 2, 4), dtype=np.uint8)
    other = np.full((2, 2, 4), 255, dtype=np.uint8)
    run, one_by_one = _FrameDigests(None), _FrameDigests(None)
    run.check_frame(0, frame, repeat=3)
    run.check_frame(1, other)
    for number in range(3):
        one_by_one.check_frame(number, frame)
    one_by_one.check_frame(3, other)
    assert run.digests == one_by_one.digests
    assert len(run.digests) == 4


def test_without_the_switch_the_comparison_draws_once(tmp_path, monkeypatch):
    monkeypatch.delenv(LEVERS_OFF_ENV, raising=False)
    seen = []
    _comparison(tmp_path, _lever_dependent(seen), last_frame=False)()
    assert seen == [True]
