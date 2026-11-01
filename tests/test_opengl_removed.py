"""The OpenGL renderer is gone: every way of asking for a renderer says so."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import click
import pytest
from click.testing import CliRunner

from arachne import tempconfig
from arachne._config.utils import ManimConfig, make_config_parser
from arachne.cli.render.commands import render

REMOVED = "The OpenGL renderer was removed: Cairo is the renderer."


@pytest.mark.parametrize("value", ["opengl", "OpenGL", "cairo"])
def test_setting_the_renderer_raises(config, value):
    with pytest.raises(AttributeError, match=REMOVED):
        config.renderer = value


def test_the_renderer_option_no_longer_exists(config):
    assert "renderer" not in config
    assert not hasattr(config, "renderer")
    with pytest.raises(AttributeError, match=REMOVED):
        config["renderer"] = "opengl"


def test_tempconfig_with_a_renderer_raises(config):
    with (
        pytest.raises(AttributeError, match=REMOVED),
        tempconfig({"renderer": "opengl"}),
    ):
        pass


def test_a_config_file_naming_a_renderer_raises(tmp_path):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text("[CLI]\nrenderer = opengl\n")
    with pytest.raises(AttributeError, match=REMOVED):
        ManimConfig().digest_parser(make_config_parser(cfg))


@pytest.mark.parametrize("args", [["--renderer", "opengl"], ["--renderer=cairo"]])
def test_the_renderer_flag_raises(tmp_path, args):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter", *args],
        standalone_mode=False,
    )
    assert isinstance(result.exception, click.BadParameter)
    assert REMOVED in str(result.exception)


def test_importing_arachne_loads_no_opengl_module():
    probe = textwrap.dedent(
        """
        import sys
        import arachne
        gl = {"moderngl", "moderngl_window", "glcontext", "dearpygui", "screeninfo"}
        print(sorted(m for m in sys.modules if m.split(".")[0] in gl))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


def test_the_opengl_modules_are_gone():
    with pytest.raises(ModuleNotFoundError):
        import arachne.mobject.opengl  # noqa: F401
    with pytest.raises(ModuleNotFoundError):
        import arachne.renderer.opengl_renderer  # noqa: F401
