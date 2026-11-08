"""The probe: each state-function play restored at two instants its record
kept, from its start or its last event before them, against what the record
kept there. A play whose restore differs keeps history its proof did not
see: it is demoted to route 3 for good, the scene is recorded again, and
the session reports where.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from textwrap import dedent

import pytest

from arachne import LEFT, RIGHT, Dot, Group
from arachne import session as session_module
from arachne.session import Session
from arachne.timeline import admission
from arachne.timeline import probe as probe_module
from arachne.timeline.execution import Divergence
from arachne.timeline.probe import (
    PlayProbe,
    Probe,
    _fields_difference,
    _paint_name,
    _tree_difference,
    _Walk,
    _walk,
)
from arachne.timeline.purity import Proof, Refused
from arachne.timeline.record import DepthReference, NodeState

from . import camera_scenes, probe_scenes, restore_scenes, route_scenes
from .conftest import SMALL, render

PROBE_SCENES = Path(probe_scenes.__file__)
RESTORE_SCENES = Path(restore_scenes.__file__)
ROUTE_SCENES = Path(route_scenes.__file__)
CAMERA_SCENES = Path(camera_scenes.__file__)


def _line(tag: str) -> int:
    for number, line in enumerate(PROBE_SCENES.read_text().splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


@pytest.fixture
def planted(monkeypatch):
    """The prover takes a :func:`.probe_scenes.planted` setter as proven, one
    that writes its own mobject and reads nothing: a forged proof, outside
    what admission vouches for, which the probe is the backstop of. A
    :func:`.probe_scenes.planted_later` setter is refused the first time
    and taken as proven after.
    """
    prove = admission.prove_setter_body
    seen: set[int] = set()

    def forged(function, receiver, scene, **kwargs):
        mark = getattr(function, "planted", None)
        if mark is True:
            return Proof(writes=[receiver])
        if mark == "later":
            line = function.__code__.co_firstlineno
            if line in seen:
                return Proof(writes=[receiver])
            seen.add(line)
            raise Refused("not admitted in its first execution")
        return prove(function, receiver, scene, **kwargs)

    monkeypatch.setattr(admission, "prove_setter_body", forged)


def _outcomes(session) -> dict[int, str]:
    return {found.play: found.outcome for found in session.probe.plays}


def test_a_planted_history_dependent_setter_is_demoted_recorded_again_and_reported(
    session_of, planted
):
    unprobed = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15, probe=False)
    first = unprobed.record.plays[0]
    # The patch admits it: without the probe it stays a state play.
    assert first.route == "state"

    session = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15)
    play = session.record.plays[0]
    assert session.probe.demoted == (0,)
    assert _outcomes(session)[0] == "demoted"
    # Recorded again, with the play on route 3 for the reason the probe gave.
    assert play.route == "ticked"
    assert play.admission.reasons[0].startswith("probe.demoted")
    assert session.record.plays[1].route == "trusted"
    # Its instants are answered as route 3 answers them.
    assert session.state_at(Fraction(1, 2) + Fraction(1, 60)).answer == "recorded"
    # The restore from the play's start missed the furthest the dot reached.
    found_play = session.probe.plays[0]
    assert found_play.bases == (play.start, play.start)

    probes = [d for d in session.divergences if d.kind == "probe"]
    assert len(probes) == 1
    found = probes[0]
    assert isinstance(found, Divergence)
    assert (found.code, found.play) == ("replay.diverged", 0)
    assert found.source_line == play.source_line
    assert found.source_line.line == _line("drift")
    assert found.source_line.path == "probe_scenes.py"
    assert "_points" in found.reason or "points" in found.reason

    # For good: a later session of the source ticks it from its first run.
    later = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15, probe=False)
    assert later.record.plays[0].route == "ticked"
    assert later.record.plays[0].admission.reasons[0].startswith("probe.demoted")


def test_a_frame_hook_writing_what_its_frames_show_is_demoted(session_of):
    """A restore from the play's start can't know what the hook wrote at
    the frames since.
    """
    session = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    assert session.record.plays[0].route == "ticked"
    assert [d.kind for d in session.divergences][:1] == ["probe"]


def test_an_execution_recorded_again_that_can_t_be_compared_is_a_divergence(
    session_of, monkeypatch
):
    """The execution recorded after a demotion, summarized as another
    interpreter's, can't be compared with the first: the session reports
    that as a divergence of the whole scene naming both interpreters, never
    as an agreement, and the demoted play is ticked all the same.
    """
    summarize = session_module.summarize
    seen: list[object] = []

    def foreign_after_the_first(record):
        summary = summarize(record)
        if seen and record is not seen[0]:
            return replace(summary, interpreter="other-99")
        seen.append(record)
        return summary

    monkeypatch.setattr(session_module, "summarize", foreign_after_the_first)
    session = session_of("FrameCounter", RESTORE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    assert session.record.plays[0].route == "ticked"
    (found,) = [d for d in session.divergences if d.kind == "execution"]
    assert (found.code, found.play, found.source_line) == (
        "replay.diverged",
        None,
        None,
    )
    assert sys.implementation.cache_tag in found.reason
    assert "other-99" in found.reason
    assert not session.stale


@pytest.mark.parametrize(
    ("path", "name"),
    [
        (ROUTE_SCENES, "Readouts"),
        (RESTORE_SCENES, "ClosureCell"),
        (RESTORE_SCENES, "DetachedTracker"),
        (RESTORE_SCENES, "SuccessionReadout"),
        (CAMERA_SCENES, "PureCentre"),
    ],
)
def test_a_clean_state_play_stays_on_route_2(session_of, path, name):
    unprobed = session_of(name, path, frame_rate=15, probe=False)
    session = session_of(name, path, frame_rate=15)
    routes = [play.route for play in session.record.plays]
    assert routes == [play.route for play in unprobed.record.plays]
    assert "state" in routes
    assert session.probe.demoted == ()
    assert session.divergences == []
    outcomes = _outcomes(session)
    assert "agreed" in outcomes.values()
    assert set(outcomes.values()) <= {"agreed", "unprobed"}
    assert sorted(outcomes) == [
        play.index
        for play in session.record.plays
        if play.route == "state" and not play.held
    ]


def test_the_probe_restores_the_first_and_last_interior_frames(session_of):
    """Neither the play's start nor an event: the Succession's event at
    0.65 s falls between frames, and the last frame is restored from it.
    """
    session = session_of("SuccessionReadout", RESTORE_SCENES, frame_rate=15)
    (found,) = session.probe.plays
    play = session.record.plays[found.play]
    assert found.instants == (Fraction(1, 15), Fraction(19, 15))
    assert found.bases == (play.start, Fraction(13, 20))


def test_an_event_on_a_frame_is_never_a_probed_instant(session_of):
    """Restoring across an event is no restore: the instant just before it is
    the last probed, and the play agrees.
    """
    session = session_of("EventOnLastFrame", PROBE_SCENES, frame_rate=15)
    play = session.record.plays[0]
    assert Fraction(3, 5) in play.events
    assert play.sample_at(Fraction(3, 5)) is not None
    (found,) = session.probe.plays
    assert found.outcome == "agreed"
    assert found.instants == (Fraction(1, 15), Fraction(8, 15))


def test_a_play_its_restore_refuses_is_counted_apart(session_of):
    """A global rebound after the play: no restore of it is sound, so the
    probe can't compare it, and nothing demotes it.
    """
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    found = {each.play: each for each in session.probe.plays}
    assert found[0].outcome == "unrestorable"
    assert found[0].reason == "restore.global_rebound"
    assert session.probe.demoted == ()
    assert session.record.plays[0].route == "state"


def test_plays_with_fewer_than_two_interior_frames_are_counted_unprobed(session_of):
    session = session_of("ShortPlays", PROBE_SCENES, frame_rate=15)
    routes = [play.route for play in session.record.plays]
    assert routes == ["state", "state", "trusted", "state", "trusted"]
    assert session.record.plays[2].held
    assert _outcomes(session) == {0: "unprobed", 1: "unprobed", 3: "agreed"}
    assert session.probe.unprobed == 2
    found = {each.play: each for each in session.probe.plays}
    assert found[3].instants == (Fraction(2, 5), Fraction(4, 3))


def test_geometry_agrees_within_1e_9_and_paint_by_its_exact_key(session_of, planted):
    """Both drift by 1e-12 an evaluation: points within the tolerance agree,
    a paint whose key differs at all does not.
    """
    moved = session_of("PlantedGeometryDrift", PROBE_SCENES, frame_rate=15)
    assert _outcomes(moved) == {0: "agreed"}
    assert moved.record.plays[0].route == "state"
    assert moved.divergences == []

    faded = session_of("PlantedPaintDrift", PROBE_SCENES, frame_rate=15)
    assert faded.probe.demoted == (0,)
    assert faded.record.plays[0].route == "ticked"
    (found,) = [d for d in faded.divergences if d.kind == "probe"]
    assert "fill" in found.reason


def test_a_render_never_reads_a_demotion(session_of, planted):
    session = session_of("PlantedDrift", PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    scene, _ = render(probe_scenes.PlantedDrift, 15)
    assert scene.record.plays[0].route == "state"


def test_a_cause_seen_in_several_passes_is_reported_once(session_of, planted):
    """Each pass demotes one play, and each recording again differs from the
    first at its first play (an unseeded draw): one execution divergence,
    however many recordings found it, and each probe divergence once.
    """
    session = session_of("TwoPassDrift", PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0, 1)
    assert [p.route for p in session.record.plays] == ["ticked", "ticked"]
    found = [(d.kind, d.play) for d in session.divergences]
    assert sorted(found) == [("execution", 0), ("probe", 0), ("probe", 1)]
    assert sorted((d.kind, d.play) for d in session.probe.divergences) == [
        ("probe", 0),
        ("probe", 1),
    ]
    (second,) = [d for d in session.divergences if d.play == 1]
    assert second.source_line.line == _line("second")


def test_a_point_cloud_s_colours_are_paint_compared_exactly(session_of):
    """A point cloud keeps its colours in an array of its own, outside any
    paint value: a hook fading it by 1e-12 a frame is still a paint change.
    """
    session = session_of("PointPaintHook", PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    (found,) = [d for d in session.divergences if d.kind == "probe"]
    assert "_rgbas" in found.reason


def test_a_re_pointed_shared_child_is_a_tree_difference():
    a, b = Dot(LEFT), Dot(RIGHT)
    second = Group(a)
    root = Group(Group(a, b), second)
    kept = {id(mob): state for mob, state in _walk([root], NodeState.of)}
    theirs = _Walk(_walk([root], lambda mob: kept[id(mob)]))
    second.submobjects[:] = [b]
    mine = _Walk(_walk([root], NodeState.of))
    lists = ([root], [], [])
    assert len(mine) == len(theirs)
    assert _tree_difference(mine, lists, theirs, lists) is not None


def test_each_walk_is_indexed_once_per_probed_instant(session_of, monkeypatch):
    """Where each mobject sits in a walk is read once for the instant, not
    once per node it is compared at: a scene of N mobjects costs the probe
    O(N) per instant, not O(N²).
    """
    indexed = []
    places = probe_module._places

    def counted(walk):
        indexed.append(len(walk))
        return places(walk)

    monkeypatch.setattr(probe_module, "_places", counted)
    session = session_of("SuccessionReadout", RESTORE_SCENES, frame_rate=15)
    probed = sum(len(found.instants) for found in session.probe.plays)
    assert probed == 2
    # The restored scene's walk and the kept capture's, once each.
    assert len(indexed) == 2 * probed


def test_a_hook_re_pointing_a_shared_child_is_demoted(session_of):
    session = session_of("SharedChildHook", PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)


#: A scene whose every execution plays one play more than the one before,
#: counting its executions in a file beside it, with a frame hook that
#: writes what its frames show, so the probe demotes a play every pass.
GROWING = """
from pathlib import Path

from arachne import Dot, Scene, ValueTracker


class GrowingPlays(Scene):
    def construct(self):
        counter = Path(__file__).with_name("counter.txt")
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        if count == 6:
            raise RuntimeError("executed seven times")
        for _ in range(count + 1):
            tracker = ValueTracker(0)
            self.mark = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
            self.add(self.mark)
            self.play(tracker.animate.set_value(2), run_time=0.4)  # @grow
            self.mark.clear_updaters()

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "mark"):
            self.mark.set_y(t)
"""


def _open_growing(tmp_path):
    scene = tmp_path / "growing.py"
    scene.write_text(dedent(GROWING))
    return Session.open(
        scene, "GrowingPlays", config={**SMALL, "frame_rate": 15}
    ), tmp_path / "counter.txt"


def test_executions_that_change_their_plays_stop_the_recording_again(tmp_path):
    """The second execution plays two plays of the line where the first
    played one, so which of them the demotion was for is unknown: the
    session records once more with every play of the line ticked, keeps
    that, reports where the executions differ, and probes no more.
    """
    admission.DEMOTED.clear()
    session, counter = _open_growing(tmp_path)
    try:
        assert counter.read_text() == "3"
        assert [p.route for p in session.record.plays] == ["ticked"] * 3
        assert not session.probe.settled
        kinds = sorted((d.kind, d.play) for d in session.divergences)
        assert kinds == [("execution", 1), ("probe", None)]
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_recording_again_stops_at_its_cap(tmp_path, monkeypatch):
    """Were neither a changed schedule to stop it nor a demotion to tick its
    play, the passes stop at twice the first execution's plays and one.
    """
    from arachne import session as session_module
    from arachne.timeline.recorder import Recorder

    monkeypatch.setattr(session_module, "_schedule_shift", lambda a, b: None)
    monkeypatch.setattr(Recorder, "_demoted", lambda self, play: False)
    admission.DEMOTED.clear()
    session, counter = _open_growing(tmp_path)
    try:
        assert counter.read_text() == "3"
        assert not session.probe.settled
    finally:
        session.close()
        admission.DEMOTED.clear()


#: A scene whose first execution plays only the play a frame hook writes
#: into, and whose later ones play a clean play before it, counting its
#: executions in a file beside it.
SHIFTED = """
from pathlib import Path

from arachne import Dot, Scene, ValueTracker


class ShiftedSchedule(Scene):
    def construct(self):
        counter = Path(__file__).with_name("counter.txt")
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        if count > 0:
            tracker = ValueTracker(0)
            clean = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
            self.add(clean)
            self.play(tracker.animate.set_value(1), run_time=0.4)  # @clean
            clean.clear_updaters()
        tracker = ValueTracker(0)
        self.written = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(self.written)
        self.play(tracker.animate.set_value(2), run_time=0.4)  # @written

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "written"):
            self.written.set_y(t)
"""


def _source_line(source: str, tag: str) -> int:
    for number, line in enumerate(dedent(source).splitlines(), start=1):
        if line.rstrip().endswith(f"# @{tag}"):
            return number
    raise LookupError(tag)


def test_a_demotion_follows_its_play_when_the_plays_shift(tmp_path):
    """The second execution plays a clean play first: the demotion found at
    index 0 of the first ticks the play it was found in, now index 1, and
    the clean play stays on route 2; the divergence names the play's index
    in the record the session holds.
    """
    admission.DEMOTED.clear()
    scene = tmp_path / "shifted.py"
    scene.write_text(dedent(SHIFTED))
    session = Session.open(scene, "ShiftedSchedule", config={**SMALL, "frame_rate": 15})
    try:
        clean, written = session.record.plays
        assert clean.source_line.line == _source_line(SHIFTED, "clean")
        assert written.source_line.line == _source_line(SHIFTED, "written")
        assert (clean.route, written.route) == ("state", "ticked")
        assert written.admission.reasons[0].startswith("probe.demoted")
        assert not session.probe.settled
        assert session.probe.demoted == (1,)
        (found,) = [d for d in session.divergences if d.kind == "probe"]
        assert found.play == 1
        assert found.source_line == written.source_line
        (also,) = session.probe.divergences
        assert also.play == 1
    finally:
        session.close()
        admission.DEMOTED.clear()


@pytest.mark.parametrize("name", ["CameraPaintHook", "ZoomedCameraPaintHook"])
def test_the_camera_s_paint_is_compared_exactly(session_of, name):
    session = session_of(name, PROBE_SCENES, frame_rate=15)
    assert session.probe.demoted == (0,)
    (found,) = [d for d in session.divergences if d.kind == "probe"]
    assert "background" in found.reason


CHANGING = """
from arachne import RIGHT, Dot, Scene


class Changing(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.play(dot.animate.shift(RIGHT), run_time=0.4)
"""


def test_executions_of_two_sources_have_no_schedule_divergence(tmp_path):
    """A source edited between executions is stale, never diverged."""
    from arachne.session import _schedule_change

    scene = tmp_path / "changing.py"
    scene.write_text(dedent(CHANGING))
    config = {**SMALL, "frame_rate": 15}
    first = Session.open(scene, "Changing", config=config, probe=False)
    scene.write_text(dedent(CHANGING) + "        self.wait(0.4)\n")
    second = Session.open(scene, "Changing", config=config, probe=False)
    try:
        assert first.record.identity.source_digest != (
            second.record.identity.source_digest
        )
        assert len(first.record.plays) != len(second.record.plays)
        assert _schedule_change(first.record, second.record) is None
        assert _schedule_change(first.record, first.record) is None
    finally:
        first.close()
        second.close()


#: Every field a mobject or a camera stores its paint in as a plain number,
#: found by a sweep of the engine's assignments; and fields that are not.
PAINT_NAMES = [
    "fill_opacity",
    "stroke_opacity",
    "background_stroke_opacity",
    "opacity",
    "original_fill_opacity",
    "bar_fill_opacity",
    "_background_opacity",
    "stroke_width",
    "background_stroke_width",
    "initial_stroke_width",
    "bar_stroke_width",
    "default_frame_stroke_width",
    "max_stroke_width_to_length_ratio",
    "sheen_factor",
    "cairo_line_width_multiple",
    "_background_color",
]
NOT_PAINT_NAMES = ["width", "radius", "angle", "tip_length", "frame_width", "number"]


def test_every_style_field_is_paint_and_no_geometry_field_is():
    assert [n for n in PAINT_NAMES if not _paint_name(n)] == []
    assert [n for n in NOT_PAINT_NAMES if _paint_name(n)] == []


SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
    '<rect width="20" height="20"/></svg>'
)

SVG_HOOK = """
from pathlib import Path

from arachne import Dot, Scene, SVGMobject, ValueTracker


class SvgOpacityHook(Scene):
    def construct(self):
        self.svg = SVGMobject(Path(__file__).with_name("tiny.svg"), opacity=1.0)
        tracker = ValueTracker(0)
        mover = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
        self.add(self.svg, mover)
        self.play(tracker.animate.set_value(2), run_time=0.4)

    def update_to_time(self, t):
        super().update_to_time(t)
        if hasattr(self, "svg"):
            self.svg.opacity -= 1e-12
"""


def test_a_stored_opacity_is_paint(tmp_path):
    admission.DEMOTED.clear()
    (tmp_path / "tiny.svg").write_text(SVG)
    scene = tmp_path / "svg_hook.py"
    scene.write_text(dedent(SVG_HOOK))
    session = Session.open(
        scene,
        "SvgOpacityHook",
        config={**SMALL, "frame_rate": 15, "media_dir": str(tmp_path / "media")},
    )
    try:
        assert session.probe.demoted == (0,)
        (found,) = [d for d in session.divergences if d.kind == "probe"]
        assert "opacity" in found.reason
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_a_probe_moved_to_another_execution_follows_its_plays():
    """A play the other execution does not play is left out, and its
    divergence names no play; the others take their indices there.
    """
    gone = Divergence("probe", 0, None, "it differed")
    kept = Divergence("probe", 1, None, "it differed too")
    found = Probe(
        (PlayProbe(0, "demoted"), PlayProbe(1, "demoted"), PlayProbe(2, "agreed")),
        (gone, kept),
    )
    moved = found.moved({0: None, 1: 3, 2: 0})
    assert [(p.play, p.outcome) for p in moved.plays] == [(0, "agreed"), (3, "demoted")]
    assert moved.demoted == (3,)
    assert [d.play for d in moved.divergences] == [None, 3]


#: A scene playing one play per role from one line, counting its executions
#: in a file beside it; a frame hook writes into the play of role "bad".
#: ``ROLES`` is filled in per test: the roles of the first execution, then
#: of every later one.
ROLES = """
from pathlib import Path

from arachne import Dot, Scene, ValueTracker

FIRST, LATER = {first!r}, {later!r}


class Occurrences(Scene):
    def construct(self):
        counter = Path(__file__).with_name("counter.txt")
        count = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(count + 1))
        self.role = None
        for role in FIRST if count == 0 else LATER:
            self.role = role
            tracker = ValueTracker(0)
            self.dot = Dot().add_updater(lambda m: m.set_x(tracker.get_value()))
            self.add(self.dot)
            self.play(tracker.animate.set_value(2), run_time=0.4)  # @role
            self.dot.clear_updaters()

    def update_to_time(self, t):
        super().update_to_time(t)
        if self.role == "bad":
            self.dot.set_y(t)
"""


def _open_occurrences(tmp_path, first, later):
    scene = tmp_path / "occurrences.py"
    scene.write_text(dedent(ROLES.format(first=first, later=later)))
    return Session.open(scene, "Occurrences", config={**SMALL, "frame_rate": 15})


@pytest.mark.parametrize(
    ("first", "later"),
    [
        (["bad", "clean"], ["clean"]),
        (["clean", "bad"], ["new", "clean", "bad"]),
    ],
    ids=["an earlier occurrence gone", "an earlier occurrence added"],
)
def test_a_demotion_at_a_line_whose_occurrences_changed_ticks_the_whole_line(
    tmp_path, first, later
):
    """Which play of the line the demotion was found in is unknown once its
    plays there change: every play of the line ticks, so the one that keeps
    history never restores, and the divergence names the line, no play.
    """
    admission.DEMOTED.clear()
    session = _open_occurrences(tmp_path, first, later)
    try:
        plays = session.record.plays
        assert len(plays) == len(later)
        assert [p.route for p in plays] == ["ticked"] * len(later)
        assert all(p.admission.reasons[0].startswith("probe.demoted") for p in plays)
        assert not session.probe.settled
        (found,) = [d for d in session.divergences if d.kind == "probe"]
        assert found.play is None
        assert found.source_line.line == _source_line(ROLES, "role")
        assert session.probe.demoted == ()
        assert [d.play for d in session.probe.divergences] == [None]
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_a_demotion_at_a_line_whose_occurrences_did_not_change_keeps_its_play(
    tmp_path,
):
    """Another line's plays changing leaves this line's correspondence."""
    admission.DEMOTED.clear()
    session = _open_occurrences(tmp_path, ["clean", "bad"], ["clean", "bad"])
    try:
        assert [p.route for p in session.record.plays] == ["state", "ticked"]
        assert session.probe.settled
        assert session.probe.demoted == (1,)
    finally:
        session.close()
        admission.DEMOTED.clear()


def test_a_depth_stand_in_is_compared_with_the_reference_point_the_record_kept():
    """A restore draws a mobject sorted by a ``z_index_group`` by a stand-in
    for the group (the point its capture found): the probe takes it for the
    kept group when its point is the one the kept capture found, and for a
    difference otherwise.
    """
    import numpy as np

    from arachne import Square, VectorizedPoint

    square = Square()
    group = VectorizedPoint([0.0, 0.0, -1.0])
    square.z_index_group = group
    kept = NodeState.of(square)
    square.z_index_group = DepthReference(np.array([0.0, 0.0, -1.0]))
    restored = NodeState.of(square)
    places = ({}, {})
    centre = np.array([0.0, 0.0, -1.0])
    assert _fields_difference(restored, kept, places, centre) is None
    moved = np.array([0.0, 0.0, 1.0])
    assert "reference point" in _fields_difference(restored, kept, places, moved)
    assert "reference point" in _fields_difference(restored, kept, places, None)
