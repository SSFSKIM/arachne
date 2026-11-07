"""A scene's error, explained: its own frames, where in the engine it was
raised, and a hint at the fix for a name the scene got wrong (an old name,
a removed one, a misspelt one, or ``manim`` itself), with the messages that
already name their fix kept as they are.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest

import arachne
from arachne import UP, FadeIn, FadeOut, Square, tempconfig
from arachne._config.utils import RENDERER_REMOVED
from arachne._removed import RENAMED
from arachne.animation.fading import _Fade
from arachne.errors import SceneError, explain
from arachne.session import Session

SURFACE = json.loads(
    (Path(__file__).parents[1] / "data" / "ce_0_20_1_surface.json").read_text()
)

#: A small frame, so executing costs little.
SMALL = {"pixel_width": 96, "pixel_height": 54}


@pytest.fixture(autouse=True)
def forget_modules(tmp_path):
    """Forget the modules a test's scenes loaded: a helper stays in
    ``sys.modules`` under its plain name, where another test's helper of
    that name would find it.
    """
    yield
    root = os.path.realpath(tmp_path) + os.sep
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and os.path.realpath(file).startswith(root):
            del sys.modules[name]


def write(tmp_path: Path, source: str, name: str = "scene.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source).lstrip())
    return path


def line_of(path: Path, tag: str) -> int:
    """The line of ``path`` that carries the comment ``# @tag``."""
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def raised(path: Path, scene: str = "S") -> BaseException:
    """What opening ``path``'s scene ``scene`` raised."""
    with pytest.raises(Exception) as info:
        Session.open(path, scene, config=SMALL, probe=False)
    return info.value


def explained(path: Path, scene: str = "S", **options) -> SceneError:
    return explain(raised(path, scene), path, **options)


def test_an_old_name_used_bare_is_its_new_name(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                self.play(ShowCreation(Circle()))  # @old
        """,
    )
    error = explained(path)
    assert isinstance(error, SceneError)
    assert error.cls == "NameError"
    assert error.line == line_of(path, "old")
    assert error.path == str(path)
    assert "ShowCreation is Create" in error.hints


def test_importing_a_removed_opengl_class_names_the_removal_and_its_cairo_class(
    tmp_path,
):
    path = write(tmp_path, "from arachne import OpenGLVMobject  # @import\n")
    error = explained(path)
    assert error.cls == "ImportError"
    assert error.line == line_of(path, "import")
    [hint] = error.hints
    assert "removed with the OpenGL renderer" in hint
    assert "use VMobject" in hint


def test_importing_manim_says_to_import_arachne(tmp_path):
    path = write(tmp_path, "from manim import *  # @import\n")
    error = explained(path)
    assert error.cls == "ModuleNotFoundError"
    assert error.line == line_of(path, "import")
    assert error.hints == ("Arachne replaces manim: write `from arachne import *`",)


def test_a_misspelt_method_names_the_nearest(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                Circle().shiftt(UP)  # @typo
        """,
    )
    error = explained(path)
    assert error.cls == "AttributeError"
    assert error.line == line_of(path, "typo")
    assert error.hints == ("did you mean shift?",)


def test_a_misspelt_setter_names_the_nearest(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                Circle().set_colr(RED)  # @typo
        """,
    )
    error = explained(path)
    assert error.cls == "AttributeError"
    assert error.line == line_of(path, "typo")
    # Deliberate break 9: the error itself names the fix, so no hint repeats it.
    assert "did you mean set_color" in error.message
    assert error.hints == ()


def test_a_misspelt_name_names_the_nearest_in_the_scene_s_namespace(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                radius_of_disc = 2
                Circle(radius=radius_of_dsc)  # @typo
        """,
    )
    error = explained(path)
    assert error.cls == "NameError"
    assert error.hints == ("did you mean radius_of_disc?",)


def test_an_arachne_name_not_imported_says_how_to_import_it(tmp_path):
    path = write(
        tmp_path,
        """
        import arachne

        class S(arachne.Scene):
            def construct(self):
                self.add(Circle())  # @bare
        """,
    )
    error = explained(path)
    assert error.hints == ("Circle is arachne.Circle: write `from arachne import *`",)


def test_a_name_of_one_of_arachne_s_modules_is_no_import_hint(tmp_path):
    path = write(
        tmp_path,
        """
        import arachne

        class S(arachne.Scene):
            def construct(self):
                utils.foo()
        """,
    )
    assert explained(path).hints == ()


@pytest.mark.parametrize(
    ("expression", "hint"),
    [
        ("arachne.ShowCreation", "ShowCreation is Create"),
        ("arachne.Circlee", "did you mean Circle or Create?"),
        (
            "arachne.TrueDot",
            "TrueDot was removed with the OpenGL renderer (Cairo is the renderer): "
            "use Dot",
        ),
    ],
)
def test_a_wrong_name_on_arachne_s_module_is_explained(tmp_path, expression, hint):
    path = write(
        tmp_path,
        f"""
        import arachne

        class S(arachne.Scene):
            def construct(self):
                {expression}  # @wrong
        """,
    )
    error = explained(path)
    assert error.cls == "AttributeError"
    assert error.hints == (hint,)


def test_a_removed_module_is_named_removed(tmp_path):
    path = write(
        tmp_path,
        "from arachne.mobject.opengl.opengl_surface import OpenGLSurface\n",
    )
    [hint] = explained(path).hints
    assert hint.startswith(
        "arachne.mobject.opengl was removed with the OpenGL renderer"
    )


def test_a_report_shows_the_scene_s_last_frames_and_where_the_engine_raised(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                self.first(Square())  # @outer

            def first(self, square):
                self.second(square)  # @middle

            def second(self, square):
                self.third(square)  # @inner

            def third(self, square):
                square.points[0] = ORIGIN  # @write
        """,
    )
    error = explained(path)
    lines = [frame.line for frame in error.frames]
    # The last three of the scene's four frames.
    assert lines == [line_of(path, tag) for tag in ("middle", "inner", "write")]
    assert error.line == line_of(path, "write")
    assert error.raised_in is not None
    assert error.raised_in.path == "arachne/mobject/values.py"
    report = error.report()
    assert f"scene.py:{line_of(path, 'write')}: in third" in report
    assert "square.points[0] = ORIGIN" in report
    assert "(raised in arachne/mobject/values.py:" in report
    # Nothing of the session, the test or the scene's construct line.
    assert "session.py" not in report
    assert "test_errors.py" not in report
    assert f"scene.py:{line_of(path, 'outer')}" not in report
    assert report.endswith(f"ValueError: {error.message}")
    assert str(error) == report


def _module(path: Path):
    """``path`` imported by hand, as code outside the engine would."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("hand_imported_scene", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _called_directly(path: Path) -> SceneError:
    """The error of the scene's ``construct()``, called by this test rather
    than by the engine, explained.
    """
    module = _module(path)
    with tempconfig({**SMALL, "dry_run": True}):
        scene = module.S()
        with pytest.raises(Exception) as info:
            scene.construct()
    return explain(info.value, path)


def test_a_scene_called_directly_that_raises_keeps_its_frame_not_the_caller_s(
    tmp_path,
):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                self.add(ShowCreation(Circle()))  # @raise
        """,
    )
    error = _called_directly(path)
    assert [(frame.path, frame.line) for frame in error.frames] == [
        ("scene.py", line_of(path, "raise"))
    ]
    assert error.line == line_of(path, "raise")
    assert error.raised_in is None
    assert error.hints == ("ShowCreation is Create",)


def test_a_scene_called_directly_whose_engine_call_raises_keeps_its_frame(
    tmp_path,
):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                Square().points[0] = ORIGIN  # @write
        """,
    )
    error = _called_directly(path)
    assert [(frame.path, frame.line) for frame in error.frames] == [
        ("scene.py", line_of(path, "write"))
    ]
    assert error.line == line_of(path, "write")
    assert error.raised_in is not None
    assert error.raised_in.path == "arachne/mobject/values.py"
    assert "test_errors.py" not in error.report()


def test_an_updater_in_a_helper_module_keeps_the_helper_s_frame(tmp_path):
    write(
        tmp_path,
        """
        def follow(mobject, dt):
            mobject.shiftt(dt)  # @helper
        """,
        name="updaters.py",
    )
    path = write(
        tmp_path,
        """
        from arachne import *
        from updaters import follow

        class S(Scene):
            def construct(self):
                dot = Dot()
                dot.add_updater(follow)
                self.add(dot)
                self.wait(0.5)  # @wait
        """,
    )
    error = explained(path)
    # The scene's wait, then the helper the engine called from it.
    assert [(frame.path, frame.line) for frame in error.frames] == [
        ("scene.py", line_of(path, "wait")),
        ("updaters.py", line_of(tmp_path / "updaters.py", "helper")),
    ]
    assert error.line == line_of(path, "wait")
    assert error.hints == ("did you mean shift?",)


def test_a_read_only_points_write_keeps_its_setter_message(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                Square().points[0] = ORIGIN
        """,
    )
    original = raised(path)
    error = explain(original, path)
    assert error.message == str(original)
    assert "mob.set_points(points)" in error.message
    assert error.hints == ()


@pytest.mark.parametrize(
    ("body", "words"),
    [
        # Deliberate break 6: play() from an updater.
        (
            "dot = Dot(); self.add(dot); "
            "dot.add_updater(lambda m, dt: self.play(FadeOut(m))); self.wait(0.5)",
            "Call it from construct() once that play has ended",
        ),
        # Deliberate break 7: a removed configuration option.
        ("config.save_as_gif = True", 'format = "gif"'),
        # The renderer setting, set and read.
        ('config.renderer = "opengl"', RENDERER_REMOVED),
        ("config.renderer", RENDERER_REMOVED),
    ],
)
def test_a_message_that_names_its_fix_is_kept_without_hints(tmp_path, body, words):
    path = write(
        tmp_path,
        f"""
        from arachne import *

        class S(Scene):
            def construct(self):
                {body}  # @raise
        """,
    )
    original = raised(path)
    error = explain(original, path)
    assert error.message == str(original)
    assert words in error.message
    assert error.line == line_of(path, "raise")
    assert error.hints == ()


def test_a_syntax_error_has_its_line(tmp_path):
    path = write(
        tmp_path,
        """
        from arachne import *

        class S(Scene):
            def construct(self):
                self.add(Circle()  # @open
                self.wait()
        """,
    )
    error = explained(path)
    assert error.cls == "SyntaxError"
    assert error.line == line_of(path, "open")
    # Compiling raised it: the source's place, and no engine frame.
    assert [(frame.path, frame.line) for frame in error.frames] == [
        ("scene.py", error.line)
    ]
    assert error.raised_in is None


def test_an_error_in_a_helper_module_shows_the_helper_and_the_scene_s_line(tmp_path):
    write(
        tmp_path,
        """
        def broken(mobject):
            return mobject.colr  # @helper
        """,
        name="helpers.py",
    )
    path = write(
        tmp_path,
        """
        from arachne import *
        from helpers import broken

        class S(Scene):
            def construct(self):
                broken(Circle())  # @call
        """,
    )
    error = explained(path)
    assert error.line == line_of(path, "call")
    assert [(frame.path, frame.line) for frame in error.frames] == [
        ("scene.py", line_of(path, "call")),
        ("helpers.py", line_of(tmp_path / "helpers.py", "helper")),
    ]
    # Mobject's own attribute lookup raised it.
    assert error.raised_in is not None
    assert error.raised_in.path == "arachne/mobject/mobject.py"
    assert error.raised_in.function == "__getattr__"
    # The nearest first, then up to two more.
    assert error.hints == ("did you mean color or set_color or get_color?",)


def test_the_scene_clock_s_time_is_carried(tmp_path):
    path = write(tmp_path, "from manim import *\n")
    error = explain(raised(path), path, t=Fraction(1))
    assert error.t == 1
    assert "at t = 1 s" in error.report()


# -- The tables are of what Arachne has: every replacement exists, and every old
# -- name is one neither Arachne nor Manim CE 0.20.1 has.


def _resolve(dotted: str) -> object:
    value: object = arachne
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


@pytest.mark.parametrize("old", RENAMED)
def test_an_old_name_is_no_arachne_or_ce_name_and_its_replacement_is(old):
    assert not hasattr(arachne, old)
    assert old not in SURFACE["star"]
    assert not any(old in names for names in SURFACE["modules"].values())
    for name in RENAMED[old].uses:
        _resolve(name)


def test_the_fade_replacements_take_the_arguments_their_hints_name():
    FadeIn(Square(), shift=UP)
    FadeIn(Square(), scale=2)
    FadeOut(Square(), shift=UP)
    assert {"shift", "scale"} <= set(inspect.signature(_Fade.__init__).parameters)
