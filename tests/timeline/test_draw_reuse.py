"""Draw reuse: a frame whose recorded state is a frame's drawn before takes
that frame's pixels instead of being drawn again (``draw_reuse``).

What a frame is drawn from is its recorded state: the version of every
mobject it reaches, the scene's lists, the camera's inputs, whether the
play's static picture holds, and the pixel size. Two frames reuse each
other's pixels only when every one of those is the same object (the
camera's inputs: the same values), never because a content key agrees.
Every frame is still written, one write per frame as before; the renderer
reports the runs of frames that reused the frame before them.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from arachne import tempconfig
from arachne.manager import Manager
from arachne.session import Session
from arachne.timeline.draw import Drawing, DrawReuse

from . import reuse_scenes
from .conftest import SMALL
from .test_video_output import CASES

FPS = 15
SCENES = Path(reuse_scenes.__file__)
#: A session opened with reuse asked for (it is off otherwise).
REUSING = {**SMALL, "frame_rate": FPS, "draw_reuse": True}


def render_writes(
    scene_class: type, **settings
) -> tuple[object, list[tuple[np.ndarray, int]]]:
    """:func:`render_as_configured` with reuse opted in (it is off unless
    asked for), as these tests are about what reuse does; pass
    ``draw_reuse=False`` for the same render without it.
    """
    return render_as_configured(scene_class, **{"draw_reuse": True, **settings})


def render_as_configured(
    scene_class: type, **settings
) -> tuple[object, list[tuple[np.ndarray, int]]]:
    """Render ``scene_class`` with nothing encoded; return the scene and
    every write its writer was handed, as (pixels, repeat), in the execution
    that ended the render. Nothing is set but ``settings``, and the test
    sizes: the default of every option is what the render runs with.
    """
    executions: list[list[tuple[np.ndarray, int]]] = []
    with tempconfig(
        {
            **SMALL,
            "frame_rate": FPS,
            "dry_run": True,
            "disable_caching": True,
            **settings,
        }
    ):
        scene = scene_class()
        manager = Manager(scene)

        def collect() -> None:
            writes: list[tuple[np.ndarray, int]] = []
            executions.append(writes)

            def write_frame(pixels, *, repeat=1):
                writes.append((pixels.copy(), repeat))

            scene.renderer.file_writer.write_frame = write_frame

        renew = manager._renew_scene

        def renewed() -> None:
            renew()
            collect()

        manager._renew_scene = renewed
        collect()
        scene.render()
    return scene, executions[-1]


def frames_of(writes: list[tuple[np.ndarray, int]]) -> list[np.ndarray]:
    return [pixels for pixels, repeat in writes for _ in range(repeat)]


def assert_same_writes(a, b) -> None:
    assert [repeat for _, repeat in a] == [repeat for _, repeat in b]
    for index, ((x, _), (y, _)) in enumerate(zip(a, b, strict=True)):
        assert x.tobytes() == y.tobytes(), f"write {index}"


# ── a render ─────────────────────────────────────────────────────────────────


def test_a_wait_whose_updater_changes_nothing_is_drawn_once_and_written_whole():
    scene, writes = render_writes(reuse_scenes.StillWait)
    renderer = scene.renderer
    (play,) = scene.timeline.record.plays
    assert not play.held, "drawn from the record frame by frame"
    assert [repeat for _, repeat in writes] == [1] * FPS, "one write per frame"
    assert renderer.draws.drawn == 1
    assert renderer.draws.reused == FPS - 1
    assert renderer.frame_runs == {0: (FPS,)}
    first = writes[0][0]
    assert all(pixels.tobytes() == first.tobytes() for pixels, _ in writes)


def test_a_change_of_line_joint_alone_is_drawn_again():
    """The triangle's corners turn round halfway: its version changes, so the
    frame is drawn, though only an enum attribute differs.
    """
    scene, writes = render_writes(reuse_scenes.JointTurns)
    frames = frames_of(writes)
    assert len(frames) == FPS
    runs = scene.renderer.frame_runs[0]
    assert len(runs) == 2, runs
    turned = runs[0]
    assert 0 < turned < FPS
    assert frames[turned - 1].tobytes() != frames[turned].tobytes(), (
        "a round joint draws other pixels than a mitred one"
    )
    assert scene.renderer.draws.drawn == 2
    _, off = render_writes(reuse_scenes.JointTurns, draw_reuse=False)
    assert_same_writes(writes, off)


def test_a_camera_moving_over_still_mobjects_is_drawn_at_every_frame():
    scene, writes = render_writes(reuse_scenes.PanOverStill)
    (play,) = scene.timeline.record.plays
    count = len(frames_of(writes))
    assert count > 2
    assert scene.renderer.frame_runs[play.index] == (1,) * count
    assert scene.renderer.draws.reused == 0
    _, off = render_writes(reuse_scenes.PanOverStill, draw_reuse=False)
    assert_same_writes(writes, off)


def test_a_frozen_frame_is_reported_as_one_run():
    scene, writes = render_writes(reuse_scenes.Holds)
    record = scene.timeline.record
    held = next(play for play in record.plays if play.held)
    assert scene.renderer.frame_runs[held.index] == (FPS,)
    assert writes[-1][1] == FPS, "a frozen frame is still written once, repeated"


@pytest.mark.parametrize(
    "scene_class",
    [
        *(case.values[0] for case in CASES),
        reuse_scenes.StillWait,
        reuse_scenes.JointTurns,
        reuse_scenes.PanOverStill,
        reuse_scenes.Holds,
    ],
    ids=lambda scene_class: scene_class.__name__,
)
def test_the_writes_are_the_same_with_and_without_reuse(scene_class):
    """Every write, its repeat and its bytes, is the same whether frames are
    reused or drawn: on every scene the video output tests write, on each
    route and each way a play comes to be drawn as it runs.
    """
    _, on = render_writes(scene_class)
    _, off = render_writes(scene_class, draw_reuse=False)
    assert on
    assert_same_writes(on, off)


def _assert_replaced_display(scene, entry: str) -> None:
    """The scene's camera is the scene's code by its ``display_funcs``
    (``user_camera`` names the table, as the draw route does), and the entry
    replaced is ``entry``.
    """
    from arachne.frame_ir.painter import user_camera
    from arachne.timeline.camera_code import replaced_methods

    camera = scene.renderer.camera
    subject, _ = user_camera(camera)
    assert subject == f"{type(camera).__name__}.display_funcs"
    assert [r.name for r in replaced_methods(camera) if r.form == "display_funcs"] == [
        entry
    ]


def test_a_camera_whose_display_function_the_scene_replaced_draws_every_frame():
    """An entry of the camera's ``display_funcs`` replaced by the scene's code
    draws through that code at every draw (here it marks each frame with its
    count): such a camera reuses nothing.
    """
    from arachne.frame_ir.painter import user_camera

    scene, on = render_writes(reuse_scenes.ReplacedDisplay)
    _, off = render_writes(reuse_scenes.ReplacedDisplay, draw_reuse=False)
    assert scene.timeline.record.refusal is None, "drawn from the record"
    assert user_camera(scene.renderer.camera) is not None
    assert [int(pixels[0, 0, 0]) for pixels, _ in off] == list(range(1, len(off) + 1))
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_a_frame_sorted_by_a_mobject_outside_the_scene_is_drawn():
    """A mobject's depth can be read from another mobject (its public
    ``z_index_group``) that the frame does not reach: what that one holds is
    no recorded state of the frame, so the frame is drawn every time.
    """
    scene, on = render_writes(reuse_scenes.OutsideDepth)
    _, off = render_writes(reuse_scenes.OutsideDepth, draw_reuse=False)
    centres = [tuple(pixels[27, 48, :3]) for pixels, _ in off]
    assert centres[0] != centres[-1], "the order changed between the plays"
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_an_engine_camera_s_display_functions_are_its_own():
    from arachne.camera.camera import Camera
    from arachne.camera.moving_camera import MovingCamera
    from arachne.camera.three_d_camera import ThreeDCamera
    from arachne.frame_ir.painter import user_camera

    for kind in (Camera, MovingCamera, ThreeDCamera):
        assert user_camera(kind()) is None, kind.__name__


def test_a_display_function_of_another_camera_is_the_scene_s_code():
    """A display function bound to another camera draws with that camera's
    settings, which change without a recorded state of this one: the camera
    is the scene's code, and reuses nothing.
    """
    scene, on = render_writes(reuse_scenes.BorrowedDisplay)
    _, off = render_writes(reuse_scenes.BorrowedDisplay, draw_reuse=False)
    _assert_replaced_display(scene, "display_funcs[VMobject]")
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_a_wrapped_do_nothing_display_function_is_the_scene_s_code():
    """A wrapper of the engine's do-nothing display function keeps its name
    and module (``functools.wraps``) but runs the scene's code: the camera
    reuses nothing.
    """
    scene, on = render_writes(reuse_scenes.WrappedNothing)
    _, off = render_writes(reuse_scenes.WrappedNothing, draw_reuse=False)
    _assert_replaced_display(scene, "display_funcs[Mobject]")
    assert [int(pixels[0, 0, 0]) for pixels, _ in off] == list(range(1, len(off) + 1))
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


@pytest.mark.parametrize(
    ("scene_class", "subject"),
    [
        (reuse_scenes.ForwardedDisplay, "display_funcs[VMobject]"),
        (reuse_scenes.ForwardedNothing, "display_funcs[Mobject]"),
    ],
    ids=["method", "nothing"],
)
def test_an_object_forwarding_to_the_engine_s_display_function_is_the_scene_s_code(
    scene_class, subject
):
    """An object that answers every attribute read as the engine's display
    function does (its code, its camera, its function) still runs its own
    code when called: only the engine's function or bound method itself is
    the engine's.
    """
    scene, on = render_writes(scene_class)
    _, off = render_writes(scene_class, draw_reuse=False)
    _assert_replaced_display(scene, subject)
    assert [int(pixels[0, 0, 0]) for pixels, _ in off] == list(range(1, len(off) + 1))
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_a_shallow_copy_of_a_camera_draws_through_the_original_s_functions():
    """A copy shares the original's display functions, bound to the original:
    it draws with the original's settings, so it counts as the scene's code.
    """
    import copy

    from arachne.camera.camera import Camera
    from arachne.frame_ir.painter import user_camera

    original = Camera()
    assert user_camera(original) is None
    assert user_camera(copy.copy(original)) is not None


def test_a_frame_showing_another_camera_s_live_picture_is_drawn():
    """A display of a camera outside the scene shows that camera's pixels as
    they are when the frame is drawn, which no recorded state holds: a frame
    that reaches such a display is drawn every time.
    """
    scene, on = render_writes(reuse_scenes.OtherCameraShown)
    _, off = render_writes(reuse_scenes.OtherCameraShown, draw_reuse=False)
    assert scene.timeline.record.refusal is None, "drawn from the record"
    centres = [tuple(pixels[27, 48, :3]) for pixels, _ in off]
    assert centres[0] != centres[-1], "the picture changed between the plays"
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_a_camera_whose_drawing_the_scene_replaced_draws_every_frame():
    """A capture replaced on the camera instance runs the scene's code at each
    draw, which no recorded state holds (here it marks each frame with its
    count): such a camera reuses nothing.
    """
    scene, on = render_writes(reuse_scenes.PatchedCamera)
    _, off = render_writes(reuse_scenes.PatchedCamera, draw_reuse=False)
    assert scene.timeline.record.refusal is None, "drawn from the record"
    assert [int(pixels[0, 0, 0]) for pixels, _ in off] == list(range(1, len(off) + 1))
    assert scene.renderer.draws.reused == 0
    assert_same_writes(on, off)


def test_a_background_built_again_from_another_image_is_drawn(tmp_path):
    """A background image is found by its file name in the assets directory:
    the same name in another directory builds another background with the
    same camera values. Frames drawn over it are drawn, not reused.
    """
    from PIL import Image

    from arachne import Scene, Square, Wait, config

    red, blue = tmp_path / "red", tmp_path / "blue"
    for folder, colour in ((red, (255, 0, 0, 255)), (blue, (0, 0, 255, 255))):
        folder.mkdir()
        Image.new("RGBA", (96, 54), colour).save(folder / "same.png")

    class RebuiltBackground(Scene):
        def construct(self):
            square = Square(fill_opacity=0.5)
            square.add_updater(lambda m, dt: None)
            self.add(square)
            self.camera.background_image = "same.png"
            wait = Wait(run_time=0.4)
            for folder in (red, blue):
                config.assets_dir = str(folder)
                self.camera.init_background()
                self.camera.reset()
                self.play(wait)

    scene, on = render_writes(RebuiltBackground)
    # Within each play the frames still reuse; across the rebuild they don't.
    assert scene.renderer.draws.reused == 2 * (len(on) // 2 - 1)
    _, off = render_writes(RebuiltBackground, draw_reuse=False)
    corners = [tuple(pixels[0, 0, :3]) for pixels, _ in off]
    assert corners[0] == (255, 0, 0)
    assert corners[-1] == (0, 0, 255)
    assert_same_writes(on, off)


@pytest.mark.parametrize("supplied", [False, True], ids=["built", "supplied"])
def test_a_camera_background_refuses_an_in_place_write(supplied):
    """A camera's background, built or supplied, can't be written in place
    or made writable: a change makes a new array.
    """
    # The reuse key holds the background by identity: it rests on this.
    from arachne.camera.camera import Camera

    camera = Camera()
    if supplied:
        camera.set_background(np.zeros_like(camera.background))
    background = camera.background
    with pytest.raises(ValueError, match="read-only"):
        background[..., :3] = 200
    with pytest.raises(ValueError, match="read-only"):
        background.flags.writeable = True


def test_a_frame_at_another_configured_pixel_size_is_drawn_again():
    """The camera draws a point's thickness from the configured pixel size
    (``Camera.adjusted_thickness``), which the scene's code can change
    between plays: a frame drawn under another is drawn, not reused.
    """
    from arachne import config

    scene, _ = render_writes(reuse_scenes.StillPoint)
    record = scene.timeline.record
    (play,) = record.plays
    first, second = play.samples[0], play.samples[1]
    reuse = DrawReuse(1)
    with tempconfig({**SMALL, "frame_rate": FPS}):
        with Drawing(record, reuse=reuse) as drawing:
            before = drawing.sample(play, first, drawing.static(play))
        config.pixel_width = 10 * SMALL["pixel_width"]
        with Drawing(record, reuse=reuse) as drawing:
            after = drawing.sample(play, second, drawing.static(play))
            assert not drawing.reused
        with Drawing(record) as drawing:
            fresh = drawing.sample(play, second, drawing.static(play))
    assert reuse.reused == 0
    assert after.tobytes() == fresh.tobytes()
    assert after.tobytes() != before.tobytes(), "the thickness follows the setting"


def test_a_frame_lit_by_a_light_the_camera_no_longer_holds_is_drawn_not_reused():
    """A frame's recorded camera inputs include the light that lit it, which
    the camera holds only while it is drawn: the key is made of it as it is
    then, not of the light the camera holds when the drawing is made.
    """
    scene, _ = render_writes(reuse_scenes.LightMovedThenReplaced, draw_reuse=False)
    record = scene.timeline.record
    (play,) = record.plays
    first, second = play.samples[1], play.samples[-1]
    reuse = DrawReuse(2)
    with tempconfig({**SMALL, "frame_rate": FPS}):
        with Drawing(record, reuse=reuse, static_layer=False) as drawing:
            before = drawing.sample(play, first, None)
        with Drawing(record, reuse=reuse, static_layer=False) as drawing:
            after = drawing.sample(play, second, None)
            assert not drawing.reused
        with Drawing(record, static_layer=False) as drawing:
            fresh = drawing.sample(play, second, None)
    assert after.tobytes() == fresh.tobytes()
    assert after.tobytes() != before.tobytes(), "the light moved the shading"


@pytest.mark.parametrize(
    "settings", [{"draw_reuse": False}, {"speed_levers": False}, {}]
)
def test_with_the_lever_off_every_frame_is_drawn(settings):
    """Off is how the option is set unasked (``{}``), by the option, and by
    the switch of every speed lever: no frame reuses another's pixels.
    """
    scene, writes = render_as_configured(reuse_scenes.StillWait, **settings)
    assert scene.renderer.draws.reused == 0
    assert scene.renderer.frame_runs == {0: (1,) * FPS}
    assert len(writes) == FPS


def test_an_opted_in_render_writes_the_bytes_an_unopted_one_does():
    """Asked for (``draw_reuse=True``), reuse takes frames; asked for or not,
    every write is the same bytes as one that reuses nothing.
    """
    scene, on = render_as_configured(reuse_scenes.StillWait, draw_reuse=True)
    plain, off = render_as_configured(reuse_scenes.StillWait)
    assert scene.renderer.draws.reused == FPS - 1
    assert plain.renderer.draws.reused == 0
    assert_same_writes(on, off)


# ── a session ────────────────────────────────────────────────────────────────


@pytest.fixture
def held_session() -> Iterator[Session]:
    session = Session.open(SCENES, "Holds", config=REUSING)
    try:
        yield session
    finally:
        session.close()


def held_instants(session: Session) -> tuple[Fraction, Fraction]:
    held = next(play for play in session.record.plays if play.held)
    return held.start + Fraction(1, 10), held.start + Fraction(7, 10)


def test_two_frames_inside_one_hold_are_drawn_once(held_session):
    first_t, second_t = held_instants(held_session)
    draws = held_session.draws
    first = np.asarray(held_session.frame(first_t))
    drawn = draws.drawn
    second = np.asarray(held_session.frame(second_t))
    assert draws.drawn == drawn
    assert draws.reused == 1
    np.testing.assert_array_equal(first, second)


def test_clear_draws_makes_the_next_frame_draw(held_session):
    first_t, second_t = held_instants(held_session)
    first = np.asarray(held_session.frame(first_t))
    held_session.clear_draws()
    drawn = held_session.draws.drawn
    second = np.asarray(held_session.frame(second_t))
    assert held_session.draws.drawn == drawn + 1
    assert held_session.draws.reused == 0
    np.testing.assert_array_equal(first, second)


def test_a_frame_at_another_quality_is_drawn(held_session):
    first_t, second_t = held_instants(held_session)
    held_session.frame(first_t)
    drawn = held_session.draws.drawn
    small = held_session.frame(second_t, quality="low_quality")
    assert held_session.draws.drawn == drawn + 1
    assert small.size == (854, 480)


def test_an_evaluated_frame_is_drawn_every_time(held_session):
    """A trusted play's frame between its captures is evaluated there, a state
    the record does not hold: it is neither kept nor taken.
    """
    moving = held_session.record.plays[0]
    assert moving.route == "trusted"
    t = moving.start + Fraction(1, 7 * FPS)
    assert moving.sample_at(t) is None
    first = np.asarray(held_session.frame(t))
    drawn = held_session.draws.drawn
    second = np.asarray(held_session.frame(t))
    np.testing.assert_array_equal(first, second)
    assert held_session.draws.reused == 0
    assert held_session.draws.drawn == drawn, "nothing kept for it"


def test_a_session_keeps_a_few_frames_and_lets_the_oldest_go():
    session = Session.open(SCENES, "Holds", config=REUSING)
    try:
        moving = session.record.plays[0]
        instants = [moving.start + Fraction(i, FPS) for i in range(7)]
        for t in instants:
            session.frame(t)
        assert len(session.draws) == min(len(instants), session.draws.capacity)
        assert session.draws.capacity < len(instants)
        reused = session.draws.reused
        session.frame(instants[-1])
        assert session.draws.reused == reused + 1
        session.frame(instants[0])
        assert session.draws.reused == reused + 1, "the oldest was let go"
    finally:
        session.close()


def test_a_described_frame_reused_between_two_drawn_ones_changes_nothing_after():
    """On a draw path that describes frames, a reused frame sends nothing to
    the stream; the next frame drawn is described against the stream's own
    last frame, and its pixels are those of a session that reuses nothing.
    """
    pytest.importorskip("arachne_core")
    from arachne.frame_ir.painter import painter_for

    found = []
    for reuse in (True, False):
        settings = {**SMALL, "frame_rate": FPS, "draw_path": "ir_cairo"}
        session = Session.open(
            SCENES, "Holds", config={**settings, "draw_reuse": reuse}
        )
        try:
            moving = session.record.plays[0]
            a, b, c = (moving.start + Fraction(i, FPS) for i in range(3))
            session.frame(a)
            session.frame(b)
            painter = painter_for(session.record, "ir_cairo")
            assert painter is not None
            sent, last = painter.frames, painter.last
            again = np.asarray(session.frame(a))
            if reuse:
                assert session.draws.reused == 1
                assert painter.frames == sent, "no delta sent"
                assert painter.last is last
            found.append((again, np.asarray(session.frame(c))))
        finally:
            session.close()
    (a_on, c_on), (a_off, c_off) = found
    np.testing.assert_array_equal(a_on, a_off)
    np.testing.assert_array_equal(c_on, c_off)


def test_a_session_reuses_no_frame_unless_asked():
    session = Session.open(SCENES, "Holds", config={**SMALL, "frame_rate": FPS})
    try:
        first_t, second_t = held_instants(session)
        session.frame(first_t)
        session.frame(second_t)
        assert session.draws.reused == 0
    finally:
        session.close()


def test_a_reload_starts_with_no_frames():
    session = Session.open(SCENES, "Holds", config=REUSING)
    try:
        session.frame(held_instants(session)[0])
        assert len(session.draws) == 1
        session.reload(force=True)
        assert len(session.draws) == 0
    finally:
        session.close()


# ── no content key ───────────────────────────────────────────────────────────

#: Entry points of the record's content key and of the execution summary's
#: state key: (module, attribute path). Each that exists is refused while a
#: frame is drawn or reused; the first three must exist.
KEY_ENTRY_POINTS = [
    ("arachne.timeline.record", "content_fields"),
    ("arachne.timeline.record", "RegisteredState.content_key"),
    ("arachne.timeline.execution", "summarize"),
    ("arachne.timeline.record", "_value_key"),
    ("arachne.timeline.record", "covered_fields"),
    ("arachne.timeline.record", "PlayRecord.sample_keys"),
    ("arachne.timeline.keys", "value_key"),
    ("arachne.timeline.execution", "play_keys"),
    ("arachne.timeline.execution", "summary_key"),
    ("arachne.timeline.execution", "_node_key"),
    ("arachne.timeline.execution", "_fields_key"),
    ("arachne.timeline.execution", "_fields"),
]
REQUIRED = 3


@contextlib.contextmanager
def keys_refused_while_drawing(monkeypatch) -> Iterator[list[str]]:
    """Within the block, every content-key and summary-key entry point raises
    if it is called while :meth:`Drawing.sample` runs, which is where a
    frame's reuse is decided, looked up and kept; outside it they work, so
    the rest of a render (its self-check, a summary) is not hindered. Each
    function is replaced wherever a module of the engine holds it (a name
    imported elsewhere included), each method on its class. Yields the
    entry points found.
    """
    drawing = [0]
    found: list[str] = []

    def refusing(name, original):
        def refused(*args, **kwargs):
            if drawing[0]:
                raise AssertionError(f"{name} was called while a frame was drawn")
            return original(*args, **kwargs)

        return refused

    for index, (module_name, path) in enumerate(KEY_ENTRY_POINTS):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            assert index >= REQUIRED, module_name
            continue
        owner_name, _, attribute = path.rpartition(".")
        owner = getattr(module, owner_name) if owner_name else module
        original = vars(owner).get(attribute)
        if original is None:
            assert index >= REQUIRED, f"{module_name}.{path}"
            continue
        name = f"{module_name}.{path}"
        found.append(name)
        replacement = refusing(name, original)
        if owner_name:
            monkeypatch.setattr(owner, attribute, replacement)
            continue
        for held in list(sys.modules.values()):
            if getattr(held, "__name__", "").startswith("arachne"):
                for key, value in list(vars(held).items()):
                    if value is original:
                        monkeypatch.setattr(held, key, replacement)

    sample = Drawing.sample

    def sampling(self, *args, **kwargs):
        drawing[0] += 1
        try:
            return sample(self, *args, **kwargs)
        finally:
            drawing[0] -= 1

    monkeypatch.setattr(Drawing, "sample", sampling)
    yield found


def test_reuse_reads_no_content_key(monkeypatch):
    with keys_refused_while_drawing(monkeypatch) as found:
        assert len(found) >= REQUIRED
        scene, writes = render_writes(reuse_scenes.StillWait)
        assert scene.renderer.draws.reused == FPS - 1, "frames were reused"
        session = Session.open(SCENES, "Holds", config=REUSING)
        try:
            first_t, second_t = held_instants(session)
            session.frame(first_t)
            session.frame(second_t)
            assert session.draws.reused == 1, "a frame was reused"
        finally:
            session.close()
