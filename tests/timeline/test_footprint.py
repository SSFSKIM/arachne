"""Drawing a recorded frame again without named drawables: the drawables are
the Frame IR's objects, a footprint is where the frame without one differs,
the changed set comes from the record's versions, and two sessions feed the
comparator, whose planted defects it then finds.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arachne_compare")

from arachne_compare import compare

from arachne.frame_ir.extract import FrameStream
from arachne.timeline.draw import Drawing, static_list
from arachne.timeline.footprint import (
    ReplayMaps,
    ReplayUnsupported,
    SessionFrames,
    changed_between,
    drawables,
    frame_without,
    object_map,
)
from arachne.timeline.routes import _resolve

from . import footprint_scenes, restore_scenes

SCENES = Path(footprint_scenes.__file__)
RESTORE_SCENES = Path(restore_scenes.__file__)

#: Large enough for a cluster to need 26 pixels (0.02% of the frame).
FRAME = {"pixel_width": 480, "pixel_height": 270, "frame_rate": 15}


def _id(session, t, type_name: str) -> str:
    (node,) = [n for n in session.state_at(t).nodes if n.type == type_name]
    return node.id


def _frame(session, t) -> np.ndarray:
    return np.asarray(session.frame(t))


def _frame_ir_ids(session, t: Fraction) -> set[str]:
    """The ids of the object table the Frame IR's first frame writes for
    the frame at ``t``, from the lists a drawing paints it from.
    """
    record = session.record
    resolved = _resolve(record, t)
    capture = resolved.capture
    with Drawing(record) as drawing:
        roots = capture.draw_list()
        drawing.show(capture, [*roots, *capture.moving])
        if resolved.prepare is not None:
            resolved.prepare(drawing)
        camera = drawing.camera
        if resolved.play is None:
            groups = [("static", roots)]
        else:
            groups = [
                ("static", static_list(camera, roots, capture.moving)),
                ("moving", list(capture.moving) or roots),
            ]
        delta = FrameStream().frame(camera, groups)
    return {name for _, name, _ in delta.objects}


@pytest.mark.parametrize("t", [Fraction(1, 5), Fraction(2, 3)], ids=["play", "wait"])
def test_the_drawables_are_the_frame_irs_objects(session_of, t):
    """A group holding a group holding leaves, one leaf holding a dot of its
    own: the drawables are the leaves, that one and its dot both, as the
    Frame IR's object table lists them, and no group.
    """
    session = session_of("Nested", SCENES, **FRAME)
    state = session.state_at(t)
    groups = {n.id for n in state.nodes if n.type == "VGroup"}
    assert len(groups) == 2
    holder = next(n for n in state.nodes if n.type == "Square" and n.children)
    found = drawables(session, t)
    assert set(found) == _frame_ir_ids(session, t)
    assert not groups & set(found)
    assert {holder.id, *holder.children} <= set(found)
    assert len(found) == 4


def test_a_group_id_stands_for_the_drawables_under_it(session_of):
    session = session_of("Nested", SCENES, **FRAME)
    t = Fraction(1, 5)
    state = session.state_at(t)
    outer = state.roots[0]
    inner = state.node(outer).children[0]
    under = [n.id for n in state.nodes if n.type != "VGroup"]
    under_inner = set(under) - {state.node(outer).children[1]}
    assert np.array_equal(
        frame_without(session, t, [outer]), frame_without(session, t, under)
    )
    found = object_map(session, t, [inner])
    assert set(found.objects) == under_inner
    assert not found.complete
    # A leaf holding a dot stands for its own painting only.
    holder = next(n for n in state.nodes if n.type == "Square" and n.children)
    own = object_map(session, t, [holder.id])
    assert set(own.objects) == {holder.id}
    dot = object_map(session, t, [holder.children[0]]).objects[holder.children[0]]
    assert not np.any(
        own.objects[holder.id].mask & _placed(dot, own.objects[holder.id])
    )


def _placed(footprint, onto) -> np.ndarray:
    """``footprint``'s mask over ``onto``'s box."""
    x0, y0, x1, y1 = onto.box
    canvas = np.zeros((max(y1, footprint.box[3]), max(x1, footprint.box[2])), bool)
    fx0, fy0, fx1, fy1 = footprint.box
    canvas[fy0:fy1, fx0:fx1] = footprint.mask
    return canvas[y0:y1, x0:x1]


def test_a_restored_dots_footprint_is_its_pixels(session_of):
    """At an instant the record did not sample, a state-function play is
    restored there; the dot's footprint is where the frame shows it.
    """
    session = session_of("UpdatedDot", SCENES, **FRAME)
    (play,) = session.record.plays
    t = play.start + play.duration * Fraction(3, 7)
    assert play.sample_at(t) is None
    assert session.state_at(t).answer == "restored"
    dot = _id(session, t, "Dot")
    footprint = object_map(session, t, [dot]).objects[dot]
    frame = _frame(session, t)
    shown = np.any(frame[..., :3] != frame[0, 0, :3], axis=2)
    x0, y0, x1, y1 = footprint.box
    whole = np.zeros(shown.shape, dtype=bool)
    whole[y0:y1, x0:x1] = footprint.mask
    assert shown.any()
    assert np.array_equal(whole, shown)


@pytest.mark.parametrize(
    ("scene", "t"),
    [
        ("Nested", Fraction(1, 5)),  # recorded
        ("Nested", Fraction(1, 7)),  # a trusted play evaluated between captures
        ("Nested", Fraction(2, 3)),  # a wait
        ("UpdatedDot", Fraction(3, 7)),  # restored
    ],
)
def test_a_frame_without_an_unknown_id_is_the_sessions_frame(session_of, scene, t):
    session = session_of(scene, SCENES, **FRAME)
    assert np.array_equal(frame_without(session, t, ["no-such-id"]), _frame(session, t))
    found = object_map(session, t, ["no-such-id"]).objects["no-such-id"]
    assert found.examined
    assert found.empty


def test_a_replay_leaves_the_session_as_it_was(session_of):
    session = session_of("Nested", SCENES, **FRAME)
    instants = [Fraction(1, 7), Fraction(1, 5), Fraction(2, 3)]
    before = [_frame(session, t) for t in instants]
    states = [session.state_at(t) for t in instants]
    for t in instants:
        object_map(session, t, list(drawables(session, t)))
    changed_between(session, instants[0], instants[1])
    assert all(
        np.array_equal(_frame(session, t), frame)
        for t, frame in zip(instants, before, strict=True)
    )
    assert [session.state_at(t) for t in instants] == states


def test_an_object_under_an_opaque_one_has_no_footprint_where_covered(session_of):
    session = session_of("Covered", SCENES, **FRAME)
    t = Fraction(0)
    ids = list(drawables(session, t))
    hidden, half, cover = ids
    found = object_map(session, t, ids).objects
    assert found[hidden].examined
    assert found[hidden].empty
    assert found[half].mask.any()
    # Nothing shows through the cover; only its antialiased rim lets some by.
    under = _placed(found[half], found[cover])
    assert not under[1:-1, 1:-1].any()
    assert under.any()


def test_the_changed_set_follows_the_records_versions(session_of):
    session = session_of("Nested", SCENES, **FRAME)
    state = session.state_at(Fraction(1, 5))
    circle = next(n.id for n in state.nodes if n.type == "Circle")
    during = changed_between(session, Fraction(1, 15), Fraction(2, 15))
    assert during.ids == {circle}
    assert not during.undecided
    assert not during.camera
    waiting = changed_between(session, Fraction(9, 15), Fraction(10, 15))
    assert not waiting.ids
    assert not waiting.undecided
    # Between captures the moving circle is undecided; nothing else is.
    between = changed_between(session, Fraction(2, 15), Fraction(1, 7))
    assert between.undecided == {circle}
    assert not between.ids
    # Evaluated at one instant, equal at both: decided unchanged.
    same = changed_between(session, Fraction(1, 7), Fraction(1, 7))
    assert not same.ids
    assert not same.undecided


def test_a_drawable_drawn_at_one_instant_only_changed(session_of):
    session = session_of(
        "Timeline", Path(footprint_scenes.__file__).parent / "scenes.py"
    )
    record = session.record
    fade = record.plays[-1]
    before, after = fade.start - Fraction(1, 60), fade.end - Fraction(1, 60)
    found = changed_between(session, before, after)
    added = set(drawables(session, after)) - set(drawables(session, before))
    assert added
    assert added <= found.ids


class _Flashed:
    """Frames with one of them painted over, after it was drawn."""

    def __init__(self, frames, at: int, box) -> None:
        self.frames, self.at, self.box = frames, at, box

    def __len__(self) -> int:
        return len(self.frames)

    def instant(self, i: int) -> Fraction:
        return self.frames.instant(i)

    def frame(self, i: int) -> np.ndarray:
        frame = self.frames.frame(i)
        if i == self.at:
            frame = frame.copy()
            x0, y0, x1, y1 = self.box
            frame[y0:y1, x0:x1, :3] = 255
        return frame


def _codes(reference, candidate, *, flash=None):
    maps = ReplayMaps(reference, candidate)
    reference_frames = SessionFrames(reference)
    candidate_frames = SessionFrames(candidate)
    if flash is not None:
        candidate_frames = _Flashed(candidate_frames, *flash)
    verdict = compare(
        reference_frames,
        candidate_frames,
        mode="same",
        maps=maps,
        changed=maps.changed,
        plays=maps.frames["candidate"].plays,
        fps=maps.frames["candidate"].fps,
        seed=0,
        key="replay",
    )
    return verdict, {f.code for f in verdict.findings}


@pytest.mark.parametrize("left_out", ["line", "glyph", "dot"])
def test_a_small_object_left_out_of_a_crowd_is_missing(session_of, left_out):
    """A one-pixel line, a glyph and a dot, each under the cluster
    threshold, among 41 other objects: through replay each is missing, and
    nothing else is found.
    """
    reference = session_of("Crowded", SCENES, **FRAME)
    candidate = session_of(f"CrowdedWithout{left_out.title()}", SCENES, **FRAME)
    listed = drawables(reference, 0)
    assert len(listed) == 44
    (dropped,) = set(listed) - set(drawables(candidate, 0))
    verdict, codes = _codes(reference, candidate)
    assert codes == {"gate.presence.missing"}
    assert {f.object for f in verdict.findings} == {dropped}
    assert verdict.counts["groups_below"] > 0
    assert verdict.counts["largest_below"] <= 25


def test_a_shifted_region_is_a_cluster(session_of):
    reference = session_of("Panel", SCENES, **FRAME)
    candidate = session_of("PanelShifted", SCENES, **FRAME)
    _, codes = _codes(reference, candidate)
    assert codes == {"gate.cluster"}


def test_a_one_frame_flash_is_flicker(session_of):
    """A flash painted into one drawn frame, at pixels of no object, while a
    square moves: flicker, and no other code.
    """
    reference = session_of("Crowded", SCENES, **FRAME)
    candidate = session_of("Crowded", SCENES, **FRAME)
    verdict, codes = _codes(reference, candidate, flash=(3, (20, 230, 40, 250)))
    assert codes == {"gate.flicker"}
    assert {f.frame for f in verdict.findings} == {3, 4}
    clean, codes = _codes(reference, candidate)
    assert codes == set()
    assert clean.verdict == "pass"


def test_frames_as_a_render_writes_them(session_of):
    session = session_of("Nested", SCENES, **FRAME)
    frames = SessionFrames(session)
    assert frames.fps == 15
    # A play of half a second, then a wait of a fifth: 8 frames and 3.
    assert len(frames) == 11
    assert list(frames.plays) == [(0, 8), (8, 11)]
    assert frames.instant(8) == Fraction(8, 15)
    assert np.array_equal(frames.frame(8), _frame(session, Fraction(8, 15)))


def test_an_instant_the_record_cannot_draw_is_unsupported(session_of):
    session = session_of("ReboundGlobal", RESTORE_SCENES, frame_rate=15)
    first = session.record.plays[0]
    t = first.start + first.duration * Fraction(1, 7)
    with pytest.raises(ReplayUnsupported):
        drawables(session, t)


def test_another_draw_path_is_unsupported(session_of):
    pytest.importorskip("arachne_core")
    session = session_of("Nested", SCENES, draw_path="ir_cairo")
    with pytest.raises(ReplayUnsupported, match="ir_cairo"):
        frame_without(session, Fraction(1, 5), [])
    with pytest.raises(ValueError, match="cairo"):
        Drawing(
            session.record, draw_path="ir_cairo", leave_out=lambda h, leaves: leaves
        )


def test_a_scene_that_plays_nothing_is_its_final_frame(session_of):
    """A still saves its final frame: compared, a changed still blocks."""
    reference = session_of("Still", SCENES, **FRAME)
    candidate = session_of("StillShifted", SCENES, **FRAME)
    frames = SessionFrames(candidate)
    assert len(frames) == 1
    assert list(frames.plays) == [(0, 1)]
    assert np.array_equal(frames.frame(0), _frame(candidate, frames.instant(0)))
    verdict, codes = _codes(reference, candidate)
    assert verdict.verdict == "block"
    assert codes == {"gate.cluster"}
    same, _ = _codes(reference, session_of("Still", SCENES, **FRAME))
    assert same.verdict == "pass"


def test_frames_at_the_rate_the_camera_draws_at(session_of):
    session = session_of("OwnRate", SCENES, **FRAME)
    frames = SessionFrames(session)
    assert frames.fps == 10
    assert len(frames) == 5
    assert list(frames.plays) == [(0, 5)]


def test_a_pair_with_one_side_on_another_draw_path_is_refused_before_a_frame(
    session_of,
):
    cairo = session_of("Nested", SCENES, **FRAME)
    described = session_of("Nested", SCENES, **FRAME, draw_path="ir_cairo")
    with pytest.raises(ReplayUnsupported, match="ir_cairo"):
        ReplayMaps(cairo, described)
    with pytest.raises(ReplayUnsupported, match="ir_cairo"):
        SessionFrames(described)


def test_a_camera_drawing_into_displays_is_refused(session_of):
    session = session_of("Zoomed", SCENES)
    with pytest.raises(ReplayUnsupported, match="displays"):
        SessionFrames(session)
    with pytest.raises(ReplayUnsupported, match="displays"):
        ReplayMaps(session, session)
    with pytest.raises(ReplayUnsupported, match="displays"):
        drawables(session, Fraction(0))


def test_the_camera_is_changed_by_the_value_of_its_inputs(session_of):
    session = session_of("CameraMoves", SCENES, **FRAME)
    frame = session.scene.camera.frame
    record = session.record
    versions = {
        id(record.node_state(frame, record.capture_at(Fraction(k, 15))[1]))
        for k in (7, 8, 9)
    }
    assert len(versions) == 3  # new versions, equal values
    # Across the cut into the play that animates the frame to where it is,
    # and inside it: the camera's inputs are the same.
    assert not changed_between(session, Fraction(7, 15), Fraction(8, 15)).camera
    assert not changed_between(session, Fraction(8, 15), Fraction(9, 15)).camera
    # Its frame moving, between two frames and to an instant evaluated
    # between captures.
    assert changed_between(session, Fraction(16, 15), Fraction(17, 15)).camera
    assert changed_between(session, Fraction(16, 15), Fraction(8, 7)).camera
    assert not changed_between(session, Fraction(8, 7), Fraction(8, 7)).camera


def test_a_centre_function_reading_a_moving_dot_changes_the_camera(session_of):
    session = session_of("FollowedCentre", SCENES, **FRAME)
    assert changed_between(session, Fraction(1, 15), Fraction(2, 15)).camera


def test_restored_centres_are_the_ones_drawn(session_of):
    """Two instants restored from one kept capture, a fixed-orientation
    centre following a dot between them: the camera changed.
    """
    session = session_of("RestoredCentre", SCENES, **FRAME)
    (play,) = session.record.plays
    early, late = Fraction(3, 15) + Fraction(1, 100), Fraction(3, 15) + Fraction(3, 100)
    assert play.route == "state"
    assert play.base_at(early) is play.base_at(late)
    assert {session.state_at(t).answer for t in (early, late)} == {"restored"}
    assert changed_between(session, early, late).camera
    assert not changed_between(session, late, late).camera


def test_a_surface_painted_as_one_group_is_one_drawable(session_of):
    session = session_of("MovingSurface", SCENES, **FRAME)
    t = Fraction(2, 15)
    state = session.state_at(t)
    (surface,) = [n for n in state.nodes if n.type == "Surface"]
    (corner,) = [n.id for n in state.nodes if n.type == "Square"]
    found = drawables(session, t)
    assert set(found) == _frame_ir_ids(session, t) == {surface.id, corner}
    assert not set(surface.children) & set(found)
    footprints = object_map(session, t, [surface.id, corner]).objects
    assert footprints[surface.id].mask.any()
    without = frame_without(session, t, [surface.id])
    shown = np.any(without[..., :3] != without[0, 0, :3], axis=2)
    alone = np.zeros(shown.shape, dtype=bool)
    x0, y0, x1, y1 = footprints[corner].box
    alone[y0:y1, x0:x1] = footprints[corner].mask
    assert np.array_equal(shown, alone)
    moved = changed_between(session, Fraction(2, 15), Fraction(3, 15))
    assert surface.id in moved.ids
    assert not (moved.ids | moved.undecided) & set(surface.children)


def test_a_line_whose_points_are_cleared_is_missing(session_of):
    """Held but painting nothing, it is examined and empty, not untested."""
    reference = session_of("Crowded", SCENES, **FRAME)
    candidate = session_of("CrowdedLineCleared", SCENES, **FRAME)
    (cleared,) = set(drawables(reference, 0)) - set(drawables(candidate, 0))
    found = object_map(candidate, 0, [cleared]).objects[cleared]
    assert found.examined
    assert found.empty
    verdict, codes = _codes(reference, candidate)
    assert codes == {"gate.presence.missing"}
    assert {f.object for f in verdict.findings} == {cleared}


def test_a_change_of_paint_order_is_a_change(session_of):
    reference = session_of("Reordered", SCENES, **FRAME)
    candidate = session_of("Reordered", SCENES, **FRAME)
    frames = SessionFrames(candidate)
    (_, cut), _ = frames.plays
    found = changed_between(candidate, frames.instant(cut - 1), frames.instant(cut))
    assert len(found.ids) == 1
    assert found.ids <= set(drawables(candidate, frames.instant(cut)))
    verdict, codes = _codes(reference, candidate)
    assert codes == set()
    assert verdict.verdict == "pass"


def test_a_removed_mobject_a_play_still_paints_is_found(session_of):
    session = session_of("RemovedMidPlay", SCENES, **FRAME)
    t = Fraction(6, 15)
    dot = next(n.id for n in session.state_at(0).nodes if n.type == "Dot")
    assert dot not in {n.id for n in session.state_at(t).nodes}
    assert dot in drawables(session, t)
    footprint = object_map(session, t, [dot]).objects[dot]
    assert footprint.mask.any()
    assert not np.array_equal(frame_without(session, t, [dot]), _frame(session, t))


def test_plays_drawn_at_different_frame_rates_are_refused(session_of):
    session = session_of("RateChanges", SCENES, **FRAME)
    assert [p.frame_rate for p in session.record.plays] == [15, 10]
    with pytest.raises(ReplayUnsupported, match="frame rates"):
        SessionFrames(session)
