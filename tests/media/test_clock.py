"""The media clock, with no GPU: ``state_at`` reports every media object's
own time and frame at the instant it answers for, under each program; a
clock shown past its clip's end is refused at the line that ran it there,
and the ways that end a clip cleanly build; a running clip shown only
through a window is refused with its fix; the four calls refuse what lies
outside the clip; and a scene whose media stand still runs no driver.
"""

from __future__ import annotations

import math
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest

from arachne import VideoMobject, hold_media, media_driver
from arachne.lint.codes import CODES
from arachne.server.dispatch import Dispatcher
from arachne.server.protocol import REFUSED, SCENE_ERROR, RpcError
from arachne.timeline.ticking import scene_updater_loads_dt

from .clock_programs import (
    CLIP,
    PROGRAMS,
    SPRITE,
    START,
    clip_time,
    frame_of,
    sprite_time,
)
from .conftest import SMALL, line_of

SCENES = Path(__file__).parent / "clock_scenes.py"


class _Server:
    """A session server with one scene of the clock scenes open."""

    def __init__(
        self, tmp_path: Path, scene: str, scenes: Path = SCENES, **settings: object
    ) -> None:
        self.server = Dispatcher()
        config = {**SMALL, "media_dir": str(tmp_path / "media"), **settings}
        self.server.handle(
            "open", {"file": str(scenes), "scene": scene, "config": config}
        )

    def media(
        self, t: Fraction | float
    ) -> tuple[Fraction, dict[str, dict[str, float]]]:
        """The instant ``state_at(t)`` answers for, and each media object's
        quantities then, by its type.
        """
        state = self.server.handle("state_at", {"t": float(t)})["state"]
        found: dict[str, dict[str, float]] = {}
        for node in state["nodes"]:
            if node["kind"] in ("VideoMobject", "Sprite", "Still"):
                numbers = state["quantities"].get(node["id"], [])
                assert not [n for n in numbers if "unreadable" in n], numbers
                found[node["kind"]] = {n["name"]: n["value"] for n in numbers}
        return Fraction(state["as_of_exact"]), found

    def close(self) -> None:
        self.server.handle("close", {})


@pytest.fixture
def server(tmp_path):
    opened: list[_Server] = []

    def open_scene(scene: str, scenes: Path = SCENES, **settings: object) -> _Server:
        opened.append(_Server(tmp_path, scene, scenes, **settings))
        return opened[-1]

    yield open_scene
    for each in opened:
        each.close()


def _refused(tmp_path: Path, scene: str, scenes: Path = SCENES) -> RpcError:
    server = Dispatcher()
    config = {**SMALL, "media_dir": str(tmp_path / "media")}
    with pytest.raises(RpcError) as raised:
        server.handle("open", {"file": str(scenes), "scene": scene, "config": config})
    assert server.handle("ping", {}) == {"generation": None}
    return raised.value


#: Instants each program is asked about: around its start, at instants on
#: and between the session's frames, and past its end.
INSTANTS = [Fraction(k, 20) for k in range(0, 48)] + [START, Fraction(1234, 1000)]


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_state_at_reports_each_media_object_s_time_and_frame(server, program):
    """``media.time`` is the program's time at the instant the answer is
    of, and ``media.frame`` the frame that time names: every answer here is
    on the session's frame grid, whose instants are ticks.
    """
    opened = server(program)
    for t in INSTANTS:
        as_of, media = opened.media(t)
        assert as_of * 60 == math.floor(as_of * 60), as_of
        clip, sprite = clip_time(program, as_of), sprite_time(program, as_of)
        assert media["VideoMobject"] == {
            "media.frame": frame_of(clip, CLIP),
            "media.time": float(clip),
        }, (t, as_of)
        assert media["Sprite"] == {
            "media.frame": frame_of(sprite, SPRITE),
            "media.time": float(sprite),
        }, (t, as_of)
        assert media["Still"] == {"media.frame": 0, "media.time": 0.0}


def test_the_frame_and_time_agree_by_the_frame_formula(server):
    """The displayed frame is the one the program's time names."""
    for program in sorted(PROGRAMS):
        opened = server(program)
        for t in INSTANTS:
            _, media = opened.media(t)
            for kind, of in (("VideoMobject", CLIP), ("Sprite", SPRITE)):
                numbers = media[kind]
                time = Fraction(numbers["media.time"]).limit_denominator(10**6)
                assert numbers["media.frame"] == frame_of(time, of)


# A stated limit, the timeline's and not the clock's: a play is evaluated at
# its ticks and its start, never at an end that falls between two ticks, and
# the scene's last capture is not ticked there. So at such an end the frame
# on display is the one the program named at the play's last tick, while
# media.time is the program's time at the end itself; and a clock that runs
# past its clip by less than a tick there is not refused. The two tests below
# pin that behaviour as it is: a timeline that comes to evaluate a play's
# end changes them.


def test_stated_limit_a_play_ending_between_ticks_shows_its_last_tick_s_frame(
    server,
):
    opened = server("ClockHalf")
    as_of, media = opened.media(10)
    assert as_of == START + 2
    last_tick = Fraction(math.floor(as_of * 60), 60)
    assert last_tick < as_of
    assert media["VideoMobject"] == {
        "media.frame": frame_of(clip_time("ClockHalf", last_tick), CLIP),
        "media.time": float(clip_time("ClockHalf", as_of)),
    }
    # The two then disagree by the frame formula: 23 shown, 24 named.
    assert frame_of(clip_time("ClockHalf", last_tick), CLIP) == 23
    assert frame_of(clip_time("ClockHalf", as_of), CLIP) == 24


def test_stated_limit_a_clock_past_its_end_by_less_than_a_tick_is_not_refused(
    server,
):
    opened = server("RunsOnPastAnOffTickEnd")
    end = Fraction(2) + Fraction(1, 120)
    as_of, media = opened.media(10)
    assert as_of == end
    assert media["VideoMobject"] == {"media.frame": 47, "media.time": float(end)}


def test_a_capture_s_content_key_tells_a_clock_out_of_range_from_one_in_range():
    """Two states of one clip on the same frame, one in range and one past
    the end, key apart: a frame of the second is refused.
    """
    from arachne.mobject.media.clock import Segment
    from arachne.timeline.record import NodeState, content_fields

    clip = VideoMobject("fixture:numbered-clip")
    clip._program = (Segment(Fraction(0), Fraction(0), Fraction(1)),)
    clip._follow_clock(Fraction(2))
    inside = dict(content_fields(NodeState.of(clip)))
    assert clip.frame == 47
    clip._follow_clock(Fraction(3))
    outside = dict(content_fields(NodeState.of(clip)))
    assert clip.frame == 47
    assert inside["_frame"] == outside["_frame"]
    # The flag is keyed; _ran_out, a frozen record, is keyed only while it is
    # None, a plain value, so it cannot be the field that tells them apart.
    assert inside["_out_of_range"] != outside["_out_of_range"]
    assert "_ran_out" not in outside


def test_a_copy_of_a_running_clip_stands_on_the_frame_it_was_copied_with(server):
    """Neither a copy the scene's code makes nor one an animation makes is
    set by the driver: each stands on the frame and at the media time it
    was copied with, the two agreeing, while the clip runs on.
    """
    opened = server("CopiesOfARunningClip")
    seen = set()
    for k in range(8, 23):
        t = Fraction(k, 15)
        state = opened.server.handle("state_at", {"t": float(t)})["state"]
        as_of = Fraction(state["as_of_exact"])
        numbers = {
            node["name"]: {
                n["name"]: n["value"] for n in state["quantities"].get(node["id"], [])
            }
            for node in state["nodes"]
            if node["kind"] == "VideoMobject"
        }
        assert numbers["clip"] == {
            "media.frame": frame_of(as_of, CLIP),
            "media.time": float(as_of),
        }
        assert numbers["twin"] == {"media.frame": 12, "media.time": 0.5}
        if "started" in numbers:
            assert numbers["started"] == {"media.frame": 12, "media.time": 0.5}
            seen.add(as_of)
    assert min(seen) == 1


def test_a_clip_s_time_is_read_by_the_scene_as_it_runs(tmp_path):
    from arachne.session import Session

    session = Session.open(SCENES, "ReadsItsTime", config=SMALL)
    try:
        assert session.scene.read == [0, Fraction(1, 4), Fraction(1, 2)]
    finally:
        session.close()


def test_a_scene_s_driver_reads_its_dt_and_holds_what_ran(tmp_path):
    from arachne.session import Session

    session = Session.open(SCENES, "ClockPlay", config=SMALL)
    try:
        scene = session.scene
        driver = media_driver(scene)
        assert driver is not None
        assert driver.updater in scene.updaters
        # The timeline's own judgement: it runs at every tick.
        assert scene_updater_loads_dt(driver.updater)
        assert {type(o).__name__ for o in driver.objects} == {"VideoMobject", "Sprite"}
    finally:
        session.close()


def test_a_scene_whose_media_stand_still_runs_no_driver(no_gpu):
    from arachne.session import Session

    standing = Session.open(SCENES, "StandingMedia", config=SMALL)
    images = Session.open(SCENES, "StandingImages", config=SMALL)
    try:
        assert media_driver(standing.scene) is None
        assert standing.scene.updaters == []
        assert [p.route for p in standing.record.plays] == [
            p.route for p in images.record.plays
        ]
        assert standing.route == "gpu"
        assert images.route == "cpu"
    finally:
        standing.close()
        images.close()


def _assert_past_end(error: RpcError, line: int) -> dict:
    assert error.code == REFUSED, error.data
    finding = error.data["finding"]
    assert finding["code"] == "media.clock_past_end"
    assert finding["severity"] == "error"
    assert finding["fix"] == CODES["media.clock_past_end"].fix
    assert finding["line"] == line
    detail = finding["detail"]
    assert detail["asset"] == "numbered-clip"
    assert detail["duration"] == 2.0
    assert detail["object"]
    return detail


def test_a_wait_one_tick_longer_than_the_clip_is_refused_at_its_line(tmp_path):
    error = _refused(tmp_path, "RunsOneTickLong")
    detail = _assert_past_end(error, line_of("one-tick-long", SCENES).line)
    assert detail["t"] == float(Fraction(121, 60))
    assert detail["media_time"] == float(Fraction(121, 60))


def test_a_fade_that_outlasts_the_clip_is_refused_at_its_line(tmp_path):
    """An object is on screen through its own fade out."""
    error = _refused(tmp_path, "FadesPastTheEnd")
    detail = _assert_past_end(error, line_of("fade-past", SCENES).line)
    assert 2 < detail["media_time"] <= 2.5


@pytest.mark.parametrize(
    ("scene", "held"),
    [
        ("FadeEndsWithTheClip", False),
        ("HeldBeforeTheFade", True),
        ("RunThenHold", True),
    ],
)
def test_the_ways_that_end_a_clip_build_and_show_its_last_frame(server, scene, held):
    opened = server(scene)
    through = []
    for k in range(0, 16):
        as_of, media = opened.media(Fraction(3, 2) + Fraction(k, 15))
        if "VideoMobject" not in media:
            # Its fade over, the clip has left the scene.
            assert as_of == (Fraction(5, 2) if held else 2)
            continue
        media_time = min(as_of, 2) if held else as_of
        assert media["VideoMobject"] == {
            "media.frame": frame_of(media_time, CLIP),
            "media.time": float(media_time),
        }
        if as_of >= 2:
            through.append(media["VideoMobject"]["media.frame"])
    # Held, it stands on its last frame all through the fade.
    assert through == ([47] * 7 if held else [])


def test_a_fade_that_ends_with_the_clip_shows_its_last_frame_last(server):
    """At 60 frames a second, the fade's last visible tick shows the clip's
    last frame; at the fade's end the clip has left the scene.
    """
    opened = server("FadeEndsWithTheClip", frame_rate=60)
    as_of, media = opened.media(Fraction(119, 60))
    assert as_of == Fraction(119, 60)
    assert media["VideoMobject"]["media.frame"] == 47
    _, media = opened.media(2)
    assert "VideoMobject" not in media


def test_a_clock_that_runs_out_offstage_is_refused_when_shown_again(tmp_path):
    error = _refused(tmp_path, "RanOutOffstage")
    detail = _assert_past_end(error, line_of("back-on", SCENES).line)
    assert detail["t"] >= 2.5
    assert detail["media_time"] == detail["t"]


def test_a_clock_that_runs_out_offstage_raises_nothing_while_it_is_off(server):
    opened = server("RanOutAddedLast")
    # Out of the scene from 1 s: not listed, and nothing refused.
    as_of, media = opened.media(2.25)
    assert as_of > 2
    assert "VideoMobject" not in media
    # Added back after the last play: listed, its clock past the end.
    _, media = opened.media(10)
    assert media["VideoMobject"]["media.time"] == 2.5


def test_a_running_clip_a_window_lists_and_the_scene_does_not_hold_is_refused(tmp_path):
    error = _refused(tmp_path, "ClipOnlyInWindow")
    assert error.code == SCENE_ERROR
    assert error.data["cls"] == "ValueError"
    assert "add it to the scene as well" in str(error.data)
    assert error.data["line"] == line_of("only-in-window", SCENES).line


def test_a_standing_clip_only_listed_by_a_window_is_not_listed(server):
    """The one case ``state_at`` does not list a media object: it stands
    still and a window alone shows it.
    """
    opened = server("StandingClipOnlyInWindow")
    _, media = opened.media(0.25)
    assert media == {}


def test_a_running_clip_the_scene_holds_and_a_window_lists_is_listed(server):
    opened = server("ClipInWindow")
    for t in (Fraction(1, 5), Fraction(1, 2), Fraction(14, 15)):
        as_of, media = opened.media(t)
        assert media["VideoMobject"] == {
            "media.frame": frame_of(as_of, CLIP),
            "media.time": float(as_of),
        }


CALLS = """\
from fractions import Fraction

from arachne import *


class Calls(Scene):
    def construct(self):
        clip = VideoMobject("fixture:numbered-clip")
        sprite = Sprite("fixture:stepper")
        self.add(clip, sprite)
        {call}  # @call
        self.wait(0.25)
"""


@pytest.mark.parametrize(
    ("call", "error", "says"),
    [
        ("clip.run(at=3)", "ValueError", "2 s"),
        ("clip.run(at=-0.5)", "ValueError", "2 s"),
        ("clip.loop(at=2.5)", "ValueError", "2 s"),
        ("clip.hold(at=Fraction(9, 4))", "ValueError", "2 s"),
        ("clip.hold(frame=48)", "ValueError", "0 to 47"),
        ("sprite.hold(frame=8)", "ValueError", "0 to 7"),
        ("clip.hold(at=1, frame=3)", "ValueError", "not both"),
        ("clip.hold(frame=1.5)", "TypeError", "index"),
        ("clip.scrub(to=2.5, over=1)", "ValueError", "2 s"),
        ("clip.scrub(to=1, over=0)", "ValueError", "more than 0"),
        ("clip.run(then='loop')", "ValueError", "hold"),
        ("clip.run(rate='fast')", "TypeError", "number"),
        ("clip.run(rate=float('nan'))", "ValueError", "finite"),
    ],
)
def test_a_call_outside_the_clip_is_refused_at_the_call(tmp_path, call, error, says):
    scene = tmp_path / "calls.py"
    scene.write_text(textwrap.dedent(CALLS.format(call=call)))
    found = _refused(tmp_path, "Calls", scene)
    assert found.code == SCENE_ERROR
    assert found.data["cls"] == error
    assert says in str(found.data)
    assert found.data["line"] == line_of("call", scene).line


def test_the_calls_need_a_running_scene():
    clip = VideoMobject("fixture:numbered-clip")
    for call in (clip.run, clip.loop, clip.hold, lambda: clip.scrub(1, 1)):
        with pytest.raises(RuntimeError, match="inside the scene's construct"):
            call()
    with pytest.raises(RuntimeError, match="inside the scene's construct"):
        clip.media_time  # noqa: B018
    # Nothing was declared: the clip stands on its first frame.
    assert clip.frame == 0


def test_only_media_objects_join_the_media_driver():
    from arachne import Scene, Square

    with pytest.raises(TypeError, match="media objects"):
        hold_media(Scene(), Square())
