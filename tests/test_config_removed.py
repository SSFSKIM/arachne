"""Setting a configuration option Arachne doesn't have raises, naming it and
what replaces it, rather than being ignored: the options Manim CE 0.22
replaced (``write_to_movie`` became ``format``), the OpenGL renderer's
options, and any name that was never an option.

Deliberate break 7 (strict configuration: an unknown name raises, and each of
the 17 removed options names its replacement), approved 2026-11-02.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from arachne import tempconfig
from arachne._config.utils import REMOVED_OPTIONS, ManimConfig, make_config_parser
from arachne.cli.render.commands import render

#: Each removed option, a value a script would have set it to, and words its
#: error must carry.
CASES = [
    ("write_to_movie", False, 'format = "none"'),
    ("save_pngs", True, '"png-sequence"'),
    ("save_as_gif", True, '"gif"'),
    ("movie_file_extension", ".mov", "format"),
    ("force_window", True, "OpenGL renderer"),
    ("custom_folders", True, "media_dir"),
    ("flush_cache", True, "arachne cache clear"),
    ("ffmpeg_loglevel", "ERROR", "media_loglevel"),
    ("enable_gui", True, "OpenGL renderer"),
    ("enable_wireframe", True, "OpenGL renderer"),
    ("fullscreen", True, "OpenGL renderer"),
    ("gui_location", (0, 0), "OpenGL renderer"),
    ("use_projection_fill_shaders", True, "OpenGL renderer"),
    ("use_projection_stroke_shaders", True, "OpenGL renderer"),
    ("window_monitor", 0, "OpenGL renderer"),
    ("window_position", "UR", "OpenGL renderer"),
    ("window_size", "default", "OpenGL renderer"),
]


def test_every_removed_option_is_listed():
    assert {name for name, _, _ in CASES} == set(REMOVED_OPTIONS)


@pytest.mark.parametrize(("name", "value", "replacement"), CASES)
def test_setting_a_removed_option_raises_naming_its_replacement(
    config, name, value, replacement
):
    with pytest.raises(AttributeError, match=f"'{name}'") as raised:
        setattr(config, name, value)
    assert replacement in str(raised.value)
    with pytest.raises(AttributeError, match=f"'{name}'"):
        config[name] = value
    with pytest.raises(AttributeError, match=f"'{name}'"), tempconfig({name: value}):
        pass


@pytest.mark.parametrize(("name", "value", "replacement"), CASES)
def test_reading_a_removed_option_raises_naming_its_replacement(
    config, name, value, replacement
):
    assert name not in config
    assert not hasattr(config, name)
    with pytest.raises(AttributeError, match=replacement):
        getattr(config, name)


@pytest.mark.parametrize(("name", "value", "replacement"), CASES)
def test_a_config_file_setting_a_removed_option_raises(
    tmp_path, name, value, replacement
):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[CLI]\n{name} = {value}\n")
    with pytest.raises(AttributeError, match=f"'{name}'"):
        ManimConfig().digest_parser(make_config_parser(cfg))


@pytest.mark.parametrize(("section", "replacement"), [("ffmpeg", "[media]")])
def test_a_config_file_section_arachne_no_longer_reads_raises(
    tmp_path, section, replacement
):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[{section}]\nloglevel = ERROR\n")
    with pytest.raises(AttributeError, match=replacement):
        ManimConfig().digest_parser(make_config_parser(cfg))


def test_setting_a_name_that_is_no_option_raises(config):
    with pytest.raises(AttributeError, match="no configuration option 'frame_rat'"):
        config.frame_rat = 30
    with pytest.raises(AttributeError, match="no configuration option 'frame_rat'"):
        config["frame_rat"] = 30


@pytest.mark.parametrize("name", ["get", "copy", "update", "keys", "digest_args"])
def test_a_method_name_is_no_option(config, name):
    method = getattr(config, name)
    with pytest.raises(AttributeError, match=f"no configuration option '{name}'"):
        setattr(config, name, 42)
    with pytest.raises(AttributeError, match=f"no configuration option '{name}'"):
        config[name] = 42
    assert getattr(config, name) == method


def test_a_method_name_assigned_as_an_attribute_raises(config):
    with pytest.raises(AttributeError, match="no configuration option 'get'"):
        config.get = 42
    with pytest.raises(AttributeError, match="no configuration option 'copy'"):
        config.copy = 42
    with pytest.raises(AttributeError, match="no configuration option 'update'"):
        config.update = 42


def test_options_are_still_set(config):
    config.format = "none"
    config["frame_rate"] = 30
    with tempconfig({"pixel_width": 96}):
        assert config.pixel_width == 96
    assert config.format == "none"
    assert config.frame_rate == 30


@pytest.mark.parametrize(
    "flag", ["--write_to_movie", "--save_pngs", "--save_as_gif", "--flush_cache"]
)
def test_a_removed_flag_is_an_error(tmp_path, flag):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    result = CliRunner().invoke(
        render, [str(scene_file), "--jupyter", flag], standalone_mode=False
    )
    assert isinstance(result.exception, click.NoSuchOption)
    assert flag in str(result.exception)
