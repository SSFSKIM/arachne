from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from arachne import WHITE, Scene, Square, Tex, Text, tempconfig
from arachne._config.output import OutputFormat, OutputSpec
from arachne._config.render_session import resolve_render_session
from arachne._config.utils import ManimConfig, make_config_parser
from arachne.cli.render.commands import render
from arachne.renderer.protocol import RendererCapabilities
from tests.assert_utils import assert_dir_filled, assert_file_exists


def _resolve_session(config):
    return resolve_render_session(
        config,
        RendererCapabilities(live_preview=True),
        renderer_name="TestRenderer",
    )


def _resolve_output(config):
    return _resolve_session(config).output


def test_tempconfig(config):
    """Test the tempconfig context manager."""
    original = config.copy()

    with tempconfig({"frame_width": 100, "frame_height": 42}):
        # check that config was modified correctly
        assert config["frame_width"] == 100
        assert config["frame_height"] == 42

        # check that no keys are missing and no new keys were added
        assert set(original.keys()) == set(config.keys())

    # check that the keys are still untouched
    assert set(original.keys()) == set(config.keys())

    # check that config is correctly restored
    for k, v in original.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_allclose(config[k], v)
        else:
            assert config[k] == v


def test_max_files_cached_uses_minus_one_for_unlimited(config):
    config.max_files_cached = -1
    assert config.max_files_cached == -1

    with pytest.raises(ValueError, match="non-negative integer or -1"):
        config.max_files_cached = float("inf")


@pytest.mark.parametrize(
    ("format", "expected_file_extension"),
    [
        ("mp4", ".mp4"),
        ("webm", ".webm"),
        ("mov", ".mov"),
        ("gif", ".mp4"),
    ],
)
def test_resolve_segment_extensions(config, format, expected_file_extension):
    config.format = format
    assert _resolve_output(config).segment_extension == expected_file_extension


@pytest.mark.parametrize(
    ("format", "expected_format"),
    [
        ("auto", OutputFormat.MP4),
        ("none", OutputFormat.NONE),
        ("png", OutputFormat.PNG),
        ("png-sequence", OutputFormat.PNG_SEQUENCE),
        ("gif", OutputFormat.GIF),
    ],
)
def test_resolve_session_output(config, format, expected_format):
    config.format = format

    assert _resolve_output(config).format is expected_format


def test_transparent_auto_output_resolves_to_mov(config):
    config.format = "auto"
    config.transparent = True

    output = _resolve_output(config)

    assert output.format is OutputFormat.MOV
    assert output.fallback_to_still is True


def test_only_automatic_video_output_allows_still_fallback(config):
    config.format = "auto"
    assert _resolve_output(config).fallback_to_still is True

    config.format = "mp4"
    assert _resolve_output(config).fallback_to_still is False


def test_still_fallback_requires_video_output():
    with pytest.raises(ValueError, match="requires a video output format"):
        OutputSpec(
            OutputFormat.PNG,
            transparent=False,
            save_sections=False,
            fallback_to_still=True,
        )


def test_explicit_no_output_is_not_dry_run(config):
    config.format = "none"

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.dry_run is False


def test_live_preview_auto_output_resolves_to_none(config):
    config.format = "auto"
    config.live_preview = True

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.output.fallback_to_still is False
    assert session.presentation.live_preview is True
    assert session.dry_run is False


def test_live_preview_requires_renderer_capability(config):
    config.live_preview = True

    with pytest.raises(ValueError, match="does not support live preview"):
        resolve_render_session(
            config,
            RendererCapabilities(),
            renderer_name="TestRenderer",
        )


def test_preview_requires_output(config):
    config.format = "none"
    config.preview = True

    with pytest.raises(ValueError, match="requires a media artifact"):
        resolve_render_session(
            config,
            RendererCapabilities(),
            renderer_name="TestRenderer",
        )


def test_explicit_transparent_mp4_is_rejected(config):
    config.format = "mp4"
    config.transparent = True

    with pytest.raises(ValueError, match="does not support an alpha channel"):
        _resolve_output(config)


def test_dry_run_resolves_no_output_without_mutating_output_request(config):
    config.format = "gif"
    config.save_sections = True
    config.dry_run = True

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.output.save_sections is False
    assert session.output.fallback_to_still is False
    assert session.dry_run is True
    assert config.format == "gif"
    assert config.save_sections is True


def test_save_last_frame_resolves_to_still_output(config):
    config.format = "auto"
    config.save_last_frame = True

    assert _resolve_output(config).format is OutputFormat.PNG


def test_save_last_frame_alias_works_with_tempconfig(config):
    original_format = config.format

    with tempconfig({"save_last_frame": True}):
        assert config.format == "png"
        assert _resolve_output(config).is_still

    assert config.format == original_format


def test_sections_require_video_output(config):
    config.format = "png"
    config.save_sections = True

    with pytest.raises(ValueError, match="Section output requires"):
        _resolve_output(config)


def test_format_is_loaded_from_config_file(tmp_path, config):
    config_file = tmp_path / "output.cfg"
    config_file.write_text("[CLI]\nformat = png-sequence\n")

    config.digest_file(config_file)

    assert config.format == "png-sequence"


def test_render_session_resolves_configured_encoder_profile(config):
    config.format = "mov"
    config.video_codec = "qtrle"
    config.pixel_format = "argb"
    config.video_encoder_options = {"threads": "1"}

    encoder = _resolve_session(config).video_encoder

    assert encoder is not None
    assert encoder.codec == "qtrle"
    assert encoder.pixel_format == "argb"
    assert encoder.options == (("threads", "1"),)


def test_video_encoder_profile_is_loaded_from_config_file(tmp_path, config):
    config_file = tmp_path / "encoder.cfg"
    config_file.write_text(
        """\
[video_encoder]
codec = libx264
pixel_format = yuv444p

[video_encoder.options]
crf = 18
preset = veryslow
tune = animation, film: grain
""",
    )

    config.digest_file(config_file)

    assert config.video_codec == "libx264"
    assert config.pixel_format == "yuv444p"
    assert config.video_encoder_options == {
        "crf": "18",
        "preset": "veryslow",
        "tune": "animation, film: grain",
    }


def test_media_loglevel_is_loaded_from_media_section(tmp_path, config):
    config_file = tmp_path / "media.cfg"
    config_file.write_text("[media]\nloglevel = DEBUG\n")

    config.digest_file(config_file)

    assert config.media_loglevel == "DEBUG"
    with pytest.raises(AttributeError):
        _ = config.ffmpeg_loglevel


def test_video_encoder_options_are_copied(config):
    supplied = {"crf": "18"}
    config.video_encoder_options = supplied
    supplied["crf"] = "30"
    assert config.video_encoder_options == {"crf": "18"}

    returned = config.video_encoder_options
    returned["crf"] = "40"
    assert config.video_encoder_options == {"crf": "18"}


def test_encoder_cli_options_replace_config_file_map(tmp_path, config):
    scene_file = tmp_path / "trivial_scene.py"
    scene_file.write_text("# --jupyter returns before importing this file\n")
    config_file = tmp_path / "encoder.cfg"
    config_file.write_text(
        """\
[video_encoder]
codec = qtrle
pixel_format = argb

[video_encoder.options]
predictor = 1
""",
    )
    runner = CliRunner()
    common_args = [
        str(scene_file),
        "--jupyter",
        "--config_file",
        str(config_file),
    ]

    result = runner.invoke(
        render,
        [
            *common_args,
            "--video-codec",
            "libx264",
            "--pixel-format",
            "yuv420p",
            "--encoder-option",
            "crf=18",
            "--encoder-option",
            "tune=animation=film",
        ],
        standalone_mode=False,
    )
    assert result.exit_code == 0, result.output
    with tempconfig({}):
        config.digest_args(result.return_value)
        assert config.video_codec == "libx264"
        assert config.pixel_format == "yuv420p"
        assert config.video_encoder_options == {
            "crf": "18",
            "tune": "animation=film",
        }

    result = runner.invoke(render, common_args, standalone_mode=False)
    assert result.exit_code == 0, result.output
    with tempconfig({}):
        config.digest_args(result.return_value)
        assert config.video_codec == "qtrle"
        assert config.pixel_format == "argb"
        assert config.video_encoder_options == {"predictor": "1"}


@pytest.mark.parametrize(
    "options",
    [
        ["--encoder-option", "missing-separator"],
        ["--encoder-option", "=missing-key"],
        ["--encoder-option", "missing-value="],
        ["--encoder-option", "crf=18", "--encoder-option", "crf=20"],
    ],
)
def test_encoder_cli_options_reject_malformed_or_duplicate_entries(tmp_path, options):
    scene_file = tmp_path / "trivial_scene.py"
    scene_file.touch()

    result = CliRunner().invoke(render, [str(scene_file), *options])

    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_cli_distinguishes_preview_from_live_preview(tmp_path):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")

    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter", "--preview", "--live-preview"],
        standalone_mode=False,
    )

    assert result.exception is None
    assert result.return_value.preview is True
    assert result.return_value.live_preview is True


def test_absent_cli_output_options_preserve_config_file_values(tmp_path):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    config_file = tmp_path / "output.cfg"
    config_file.write_text(
        "[CLI]\n"
        "format = webm\n"
        "output_file = configured-name\n"
        "background_opacity = 0.5\n",
    )
    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter"],
        standalone_mode=False,
    )
    assert result.exception is None

    candidate = ManimConfig().digest_file(config_file)
    candidate.digest_args(result.return_value)

    assert candidate.format == "webm"
    assert candidate.output_file == "configured-name"
    assert candidate.transparent is True


class MyScene(Scene):
    def construct(self):
        self.add(Square())
        self.add(Text("Prepare for unforeseen consequencesλ"))
        self.add(Tex(r"$\lambda$"))
        self.wait(1)


def test_transparent(config):
    """Test the 'transparent' config option."""
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 255])

    config.transparent = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 0])


def test_transparent_by_background_opacity(config, dry_run):
    config.background_opacity = 0.5
    assert config.transparent is True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 127])
    assert config.transparent is True


def test_background_color(config):
    """Test the 'background_color' config option."""
    config.background_color = WHITE
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [255, 255, 255, 255])


def test_digest_file(tmp_path, config):
    """Test that a config file can be digested programmatically."""
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            media_dir = this_is_my_favorite_path
            video_dir = {media_dir}/videos
            sections_dir = {media_dir}/{scene_name}/prepare_for_unforeseen_consequences
            frame_height = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    assert config.get_dir("media_dir") == Path("this_is_my_favorite_path")
    assert config.get_dir("video_dir") == Path("this_is_my_favorite_path/videos")
    assert config.get_dir("sections_dir", scene_name="test") == Path(
        "this_is_my_favorite_path/test/prepare_for_unforeseen_consequences"
    )


def test_custom_dirs(tmp_path, config):
    config.media_dir = tmp_path
    config.save_sections = True
    config.log_to_file = True
    config.frame_rate = 15
    config.pixel_height = 854
    config.pixel_width = 480
    config.sections_dir = "{media_dir}/test_sections"
    config.video_dir = "{media_dir}/test_video"
    config.partial_movie_dir = "{media_dir}/test_partial_movie_dir"
    config.images_dir = "{media_dir}/test_images"
    config.text_dir = "{media_dir}/test_text"
    config.tex_dir = "{media_dir}/test_tex"
    config.log_dir = "{media_dir}/test_log"

    scene = MyScene()
    scene.render()
    tmp_path = Path(tmp_path)
    assert_dir_filled(tmp_path / "test_sections")
    assert_file_exists(tmp_path / "test_sections/MyScene.json")

    assert_dir_filled(tmp_path / "test_video")
    assert_file_exists(tmp_path / "test_video/MyScene.mp4")

    assert_dir_filled(tmp_path / "test_partial_movie_dir")
    assert_file_exists(tmp_path / "test_partial_movie_dir/partial_movie_file_list.txt")

    assert not (tmp_path / "test_images").exists()

    assert_dir_filled(tmp_path / "test_text")
    assert_dir_filled(tmp_path / "test_tex")
    assert_dir_filled(tmp_path / "test_log")


def test_pixel_dimensions(tmp_path, config):
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    # aspect ratio is set using pixel measurements
    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if not specified in the cfg file, frame_width is set using the aspect ratio
    np.testing.assert_allclose(config.frame_height, 8.0)
    np.testing.assert_allclose(config.frame_width, 8.0)


def test_frame_size(tmp_path, config):
    """Test that the frame size can be set via config file."""
    np.testing.assert_allclose(
        config.aspect_ratio, config.pixel_width / config.pixel_height
    )
    np.testing.assert_allclose(config.frame_height, 8.0)

    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            frame_height = 10
            frame_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if both are specified in the cfg file, the aspect ratio is ignored
    np.testing.assert_allclose(config.frame_height, 10.0)
    np.testing.assert_allclose(config.frame_width, 10.0)


def test_temporary_dry_run(config):
    """Test that tempconfig correctly restores after setting dry_run."""
    assert _resolve_output(config).is_video

    with tempconfig({"dry_run": True}):
        assert not _resolve_output(config).enabled

    assert _resolve_output(config).is_video


def test_dry_run_with_png_format(config, dry_run):
    """Test that there are no exceptions when running a png without output"""
    config.format = "png"
    config.disable_caching = True
    assert config.dry_run is True
    scene = MyScene()
    scene.render()


def test_dry_run_with_png_format_skipped_animations(config, dry_run):
    """Test that there are no exceptions when running a png without output and skipped animations"""
    config.format = "png"
    config.disable_caching = True
    assert config["dry_run"] is True
    scene = MyScene(skip_animations=True)
    scene.render()


def test_tex_template_file(tmp_path):
    """Test that a custom tex template file can be set from a config file."""
    tex_file = Path(tmp_path / "my_template.tex")
    tex_file.write_text("Hello World!")
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            f"""
            [CLI]
            tex_template_file = {tex_file}
            """,
        )

    custom_config = ManimConfig().digest_file(tmp_cfg.name)

    assert Path(custom_config.tex_template_file) == tex_file
    assert custom_config.tex_template.body == "Hello World!"


def test_from_to_animations_only_first_animation(config):
    config: ManimConfig
    config.from_animation_number = 0
    config.upto_animation_number = 0

    class SceneWithTwoAnimations(Scene):
        def construct(self):
            self.after_first_animation = False
            s = Square()
            self.add(s)
            self.play(s.animate.scale(2))
            self.renderer.update_skipping_status()
            self.after_first_animation = True
            self.play(s.animate.scale(2))

    scene = SceneWithTwoAnimations()
    scene.render()

    assert scene.after_first_animation is False


# The speed levers: one option each, and the switch that turns them all off
# (``speed_levers``; its tests are in ``tests/utils/test_levers_switch.py``).

#: Each lever option, what a config file writes to set it away from its
#: default, the render command's flags that do the same, and what it then
#: reads. ``{dir}`` stands for a directory of the test's own.
LEVER_SETTINGS = [
    ("speed_levers", "False", ["--no-speed-levers"], False),
    ("vectorize_trusted", "False", ["--no-vectorize-trusted"], False),
    ("construction_memo", "False", ["--no-construction-memo"], False),
    ("draw_reuse", "False", ["--no-draw-reuse"], False),
    ("hold_gaps", "True", ["--hold-gaps"], True),
    ("render_workers", "3", ["--workers", "3"], 3),
    ("typeset_dir", "{dir}", ["--typeset_dir", "{dir}"], Path("{dir}")),
]


def _filled(value, directory):
    if isinstance(value, str):
        return value.replace("{dir}", str(directory))
    if isinstance(value, Path):
        return Path(str(value).replace("{dir}", str(directory)))
    if isinstance(value, list):
        return [_filled(item, directory) for item in value]
    return value


def _cli_args(tmp_path, args):
    """The namespace ``arachne render`` hands the configuration for ``args``."""
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    result = CliRunner().invoke(
        render, [str(scene_file), "--jupyter", *args], standalone_mode=False
    )
    assert result.exception is None, result.output
    return result.return_value


def test_the_speed_levers_defaults(monkeypatch):
    """On: the levers that change speed and never a picture. Off or unset:
    those whose output a reader outside the engine sees (the file's frame
    timing, how many processes draw it, where typesetting is kept).
    """
    monkeypatch.delenv("ARACHNE_TYPESET_DIR", raising=False)
    defaults = ManimConfig().digest_parser(make_config_parser())
    assert defaults.speed_levers is True
    assert defaults.vectorize_trusted is True
    assert defaults.construction_memo is True
    assert defaults.draw_reuse is True
    assert defaults.hold_gaps is False
    assert defaults.render_workers == 1
    assert defaults.typeset_dir is None


@pytest.mark.parametrize(("name", "written", "args", "read"), LEVER_SETTINGS)
def test_a_lever_is_set_from_a_config_file(tmp_path, name, written, args, read):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[CLI]\n{name} = {_filled(written, tmp_path)}\n")
    assert getattr(ManimConfig().digest_file(cfg), name) == _filled(read, tmp_path)


@pytest.mark.parametrize(("name", "written", "args", "read"), LEVER_SETTINGS)
def test_a_lever_is_set_from_the_command_line(tmp_path, name, written, args, read):
    candidate = ManimConfig().digest_parser(make_config_parser())
    candidate.digest_args(_cli_args(tmp_path, _filled(args, tmp_path)))
    assert getattr(candidate, name) == _filled(read, tmp_path)


@pytest.mark.parametrize(("name", "written", "args", "read"), LEVER_SETTINGS)
def test_a_lever_is_set_in_a_tempconfig_and_restored(
    config, tmp_path, name, written, args, read
):
    before = getattr(config, name)
    with tempconfig({name: _filled(read, tmp_path)}):
        assert getattr(config, name) == _filled(read, tmp_path)
    assert getattr(config, name) == before


@pytest.mark.parametrize(("name", "written", "args", "read"), LEVER_SETTINGS)
def test_an_absent_lever_flag_keeps_the_config_file_value(
    tmp_path, name, written, args, read
):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[CLI]\n{name} = {_filled(written, tmp_path)}\n")
    candidate = ManimConfig().digest_file(cfg)
    candidate.digest_args(_cli_args(tmp_path, []))
    assert getattr(candidate, name) == _filled(read, tmp_path)


def test_a_lever_flag_turns_a_lever_back_on(tmp_path):
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text("[CLI]\ndraw_reuse = False\nhold_gaps = True\n")
    candidate = ManimConfig().digest_file(cfg)
    candidate.digest_args(_cli_args(tmp_path, ["--draw-reuse", "--no-hold-gaps"]))
    assert candidate.draw_reuse is True
    assert candidate.hold_gaps is False


@pytest.mark.parametrize(
    "name",
    ["vectorise_trusted", "construction_memos", "draw_cache", "workers", "typeset"],
)
def test_a_misspelt_lever_raises(config, name):
    """Deliberate break 7: a lever set under another name is never kept unread."""
    with pytest.raises(AttributeError, match=f"no configuration option '{name}'"):
        setattr(config, name, False)
    with (
        pytest.raises(AttributeError, match=f"no configuration option '{name}'"),
        tempconfig({name: False}),
    ):
        pass


def test_workers_auto_is_the_processors_this_process_may_run_on(config, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 2, 5}, raising=False)
    config.render_workers = "auto"
    assert config.render_workers == 3


def test_workers_auto_is_the_cpu_count_without_affinity(config, monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    config.render_workers = "auto"
    assert config.render_workers == 7


def test_workers_auto_from_a_config_file_and_the_command_line(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1, 3}, raising=False)
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text("[CLI]\nrender_workers = auto\n")
    assert ManimConfig().digest_file(cfg).render_workers == 2
    candidate = ManimConfig().digest_parser(make_config_parser())
    candidate.digest_args(_cli_args(tmp_path, ["--workers", "auto"]))
    assert candidate.render_workers == 2


@pytest.mark.parametrize("value", [0, -2, True, 2.5, "many", "2"])
def test_workers_is_a_positive_integer_or_auto(config, value):
    with pytest.raises(ValueError, match="render_workers"):
        config.render_workers = value


@pytest.mark.parametrize("value", ["0", "many"])
def test_the_workers_flag_refuses_what_is_not_a_count(tmp_path, value):
    scene_file = tmp_path / "scene.py"
    scene_file.touch()
    result = CliRunner().invoke(render, [str(scene_file), "--workers", value])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


@pytest.mark.parametrize("name", ["vectorize_trusted", "hold_gaps", "speed_levers"])
def test_a_lever_switch_is_boolean(config, name):
    with pytest.raises(ValueError, match=name):
        setattr(config, name, "yes")


def test_the_typeset_dir_is_unset_by_an_empty_value(config, tmp_path):
    config.typeset_dir = tmp_path
    assert config.typeset_dir == tmp_path
    config.typeset_dir = ""
    assert config.typeset_dir is None
    config.typeset_dir = str(tmp_path)
    assert config.typeset_dir == tmp_path
    config.typeset_dir = None
    assert config.typeset_dir is None


def test_the_typeset_dir_is_named_by_the_environment(tmp_path, monkeypatch):
    """``ARACHNE_TYPESET_DIR`` outranks a config file; the command line
    outranks both.
    """
    monkeypatch.setenv("ARACHNE_TYPESET_DIR", str(tmp_path / "env"))
    assert ManimConfig().digest_parser(make_config_parser()).typeset_dir == (
        tmp_path / "env"
    )
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[CLI]\ntypeset_dir = {tmp_path / 'file'}\n")
    candidate = ManimConfig().digest_file(cfg)
    assert candidate.typeset_dir == tmp_path / "env"
    candidate.digest_args(_cli_args(tmp_path, ["--typeset_dir", str(tmp_path / "cli")]))
    assert candidate.typeset_dir == tmp_path / "cli"


def test_an_empty_typeset_environment_variable_names_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ARACHNE_TYPESET_DIR", "")
    cfg = tmp_path / "arachne.cfg"
    cfg.write_text(f"[CLI]\ntypeset_dir = {tmp_path}\n")
    assert ManimConfig().digest_file(cfg).typeset_dir == tmp_path
