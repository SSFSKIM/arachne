"""A declared illustration changes only at its declared rate, read from the
record at every instant a frame is delivered at, at 15, 30 and 60 frames
per second (60 is a session's 60 Hz record too): whatever moves it (a pose,
a whole-rig move, a fade, an animation of one part), beside smooth text,
off the grid, with line boil, and posed between plays. The drawing at a
delivered instant ``t`` is the one true at the last boundary ``k / 12`` at
or before ``t``, computed here from the scene's own terms.

The statement about ``state_at``'s ``as_of`` is kept apart (its own test),
and so are the delivered frames themselves and the last visible frame,
which are read off rendered frames (``test_rendered.py``).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from arachne import Rig, Text, Timing, timing_of

from .reading import (
    RATES,
    art,
    boundary,
    box_of,
    close,
    frame_captures,
    points_at,
    scene_box_of,
    value_at,
)
from .timing_scenes import BOIL, MOVE, OPEN, RATE


def _posed(value: float, shift=None) -> Rig:
    rig = Rig("fixture:lidded-box").pose(lid_open=value)
    return rig.shift(shift) if shift is not None else rig


@pytest.mark.parametrize("fps", RATES)
def test_a_pose_of_a_declared_rig_shows_the_drawing_of_its_last_boundary(opened, fps):
    session = opened("PosedBox", fps)
    box = box_of(session)
    seen = set()
    for t, capture in frame_captures(session, fps):
        want = OPEN * min(boundary(t), 1)
        assert value_at(session, box, capture) == pytest.approx(float(want)), t
        close(
            points_at(session, art(box), capture), [s.points for s in art(_posed(want))]
        )
        seen.add(float(want))
    # It moved: on the grid, not held still.
    assert len(seen) > 8


@pytest.mark.parametrize("fps", RATES)
def test_a_whole_rig_move_of_a_declared_rig_steps(opened, fps):
    session = opened("MovedBox", fps)
    box = box_of(session)
    for t, capture in frame_captures(session, fps):
        moved = float(boundary(t)) * MOVE
        close(
            points_at(session, art(box), capture),
            [s.points for s in art(_posed(0, moved))],
        )


def _fill_opacity(session, mobject, capture) -> float:
    paint = session.record.node_state(mobject, capture).get("_paint")
    return float(np.asarray(paint.fill_rgbas)[0, 3])


@pytest.mark.parametrize("fps", RATES)
def test_a_fade_in_of_a_declared_rig_steps_from_its_first_appearance(opened, fps):
    session = opened("FadedInBox", fps)
    box = box_of(session)
    body = art(box)[0]
    full = _fill_opacity(session, body, session.record.final)
    for t, capture in frame_captures(session, fps):
        assert _fill_opacity(session, body, capture) == pytest.approx(
            full * float(boundary(t))
        ), t


@pytest.mark.parametrize("fps", RATES)
def test_an_animation_of_a_part_steps_by_the_mark_on_its_ancestor(opened, fps):
    session = opened("PartOfBox", fps)
    box = box_of(session)
    lid = box.part("lid").submobjects[0]
    colours = {}
    for t, capture in frame_captures(session, fps):
        paint = session.record.node_state(lid, capture).get("_paint")
        colours.setdefault(boundary(t), set()).add(
            np.asarray(paint.fill_rgbas).tobytes()
        )
    # One colour a drawing, and a new one at each boundary.
    assert all(len(found) == 1 for found in colours.values())
    assert len({next(iter(found)) for found in colours.values()}) == len(colours)
    assert len(colours) == RATE


@pytest.mark.parametrize("fps", RATES)
def test_what_a_play_does_not_write_keeps_what_was_true(opened, fps):
    session = opened("PartsInTurn", fps)
    box = box_of(session)
    lid = box.part("lid").submobjects[0]
    final = session.record.node_state(lid, session.record.final).get("_paint")
    red = np.asarray(final.fill_rgbas)
    for t, capture in frame_captures(session, fps):
        paint = session.record.node_state(lid, capture).get("_paint")
        shown = np.asarray(paint.fill_rgbas)
        if boundary(t) >= Fraction(1, 2):
            # The lid's play ended at 0.45 s; from the next boundary its
            # final colour is shown, while the next play colours the body.
            assert np.array_equal(shown, red), t
        else:
            assert not np.array_equal(shown, red), t


@pytest.mark.parametrize("fps", RATES)
def test_the_rig_steps_while_the_text_in_its_group_moves_at_every_frame(opened, fps):
    session = opened("BoxBesideText", fps)
    box = box_of(session)
    words = next(m for m in session.record.mobjects() if isinstance(m, Text))
    body = art(box)[0]
    glyphs = words.family_members_with_points()
    drawn = []
    for t, capture in frame_captures(session, fps):
        assert _fill_opacity(session, body, capture) == pytest.approx(
            float(boundary(t)) * _fill_opacity(session, body, session.record.final)
        )
        drawn.append(
            b"".join(
                np.asarray(
                    session.record.node_state(g, capture).get("_points")
                ).tobytes()
                + np.asarray(
                    session.record.node_state(g, capture).get("_paint").stroke_rgbas
                ).tobytes()
                for g in glyphs
            )
        )
    # The text is written on every frame: no two delivered frames of the
    # write show it alike.
    assert len(set(drawn)) == len(drawn)


def _off_grid_value(t: Fraction) -> float:
    """The true opening of the off-grid pose at ``t``: from 0.37 s to 0.87 s."""
    start, end = Fraction(37, 100), Fraction(87, 100)
    if t <= start:
        return 0.0
    if t >= end:
        return float(OPEN)
    return float(OPEN * (t - start) / (end - start))


@pytest.mark.parametrize("fps", RATES)
@pytest.mark.parametrize("scene", ["OffGridThenWait", "OffGridEnding"])
def test_an_off_grid_play_shows_its_final_state_from_the_next_boundary(
    opened, fps, scene
):
    session = opened(scene, fps)
    box = box_of(session)
    finals = []
    for t, capture in frame_captures(session, fps):
        want = _off_grid_value(boundary(t))
        assert value_at(session, box, capture) == pytest.approx(want), t
        if want == OPEN:
            finals.append(t)
    if scene == "OffGridThenWait":
        # The pose ends at 0.87 s, between two boundaries: its final state
        # is first shown at 11/12, inside the wait.
        assert finals[0] == min(
            t for t, _ in frame_captures(session, fps) if t >= Fraction(11, 12)
        )
    else:
        # The scene ends before that boundary: no delivered frame shows it.
        assert finals == []
        # It is true at once, for the code that runs next and at the end.
        assert box.value("lid_open") == OPEN


@pytest.mark.parametrize("fps", RATES)
def test_a_pose_after_an_off_grid_move_is_drawn_where_the_move_left_the_rig(
    opened, fps
):
    session = opened("MovedThenPosed", fps)
    box = box_of(session)
    move_end = Fraction(9, 20)
    for t, capture in frame_captures(session, fps):
        at = boundary(t)
        moved = float(min(at / move_end, Fraction(1)))
        opening = float(OPEN * max(at - move_end, Fraction(0)) / Fraction(1, 2))
        close(
            points_at(session, art(box), capture),
            [s.points for s in art(_posed(opening, moved * np.array([1.0, 0, 0])))],
        )


@pytest.mark.parametrize("fps", RATES)
def test_the_scene_s_code_after_a_play_reads_what_is_true(opened, fps):
    session = opened("ShortWaitThenPose", fps)
    box = box_of(session)
    # The wait ends at 0.89 s, before the boundary at 11/12 that would show
    # the pose's end: the code still reads the pose as it ended.
    assert session.scene.read == OPEN
    start = Fraction(89, 100)
    for t, capture in frame_captures(session, fps):
        at = boundary(t)
        if at >= start:
            want = float(OPEN * (1 - min((at - start) / Fraction(1, 2), Fraction(1))))
        else:
            want = _off_grid_value(at)
        assert value_at(session, box, capture) == pytest.approx(want), t


@pytest.mark.parametrize("fps", RATES)
def test_boil_changes_only_on_its_own_grid_in_plays_and_waits(opened, fps):
    session = opened("Boiling", fps)
    box = box_of(session)
    lid = box.part("lid").submobjects[0]
    by_drawing: dict[tuple[Fraction, Fraction], set[bytes]] = {}
    for t, capture in frame_captures(session, fps):
        points = points_at(session, [lid], capture)[0]
        by_drawing.setdefault((boundary(t), boundary(t, BOIL.rate)), set()).add(
            points.tobytes()
        )
    assert all(len(found) == 1 for found in by_drawing.values())
    # In the wait (0.5 s to 1.5 s) the drawing holds still and the boil
    # changes at each of its boundaries, six times a second.
    waiting = {
        key: found
        for key, found in by_drawing.items()
        if Fraction(1, 2) <= key[0] < Fraction(3, 2)
    }
    boils = {next(iter(found)) for found in waiting.values()}
    assert len(boils) == 6


def test_boil_is_the_same_at_shared_delivered_instants_and_stays_small(opened):
    lids = {}
    for fps in RATES:
        session = opened("Boiling", fps)
        box = box_of(session)
        lid = box.part("lid").submobjects[0]
        lids[fps] = {
            t: points_at(session, [lid], capture)[0]
            for t, capture in frame_captures(session, fps)
        }
    for t, points in lids[15].items():
        # The same drawing and the same boil, to the rounding of the move.
        close([points], [lids[30][t]], tol=1e-9)
        close([points], [lids[60][t]], tol=1e-9)
    # Displaced from the drawing on display by at most the amplitude, in
    # reference pixels of the 1080-tall picture.
    session = opened("Boiling", 60)
    box = box_of(session)
    lid = box.part("lid").submobjects[0]
    unpaused = Rig("fixture:lidded-box").pose(lid_open=60).part("lid").submobjects[0]
    held = lids[60][Fraction(1)]
    moved = np.linalg.norm((held - unpaused.points)[:, :2], axis=1).max()
    assert 0 < moved * 1080 / 8 <= BOIL.amplitude + 1e-9


@pytest.mark.parametrize("fps", RATES)
def test_a_pose_by_the_scene_s_code_between_plays_is_shown_from_the_next_boundary(
    opened, fps
):
    session = opened("PosedBetweenPlays", fps)
    box = box_of(session)
    for t, capture in frame_captures(session, fps):
        at = boundary(t)
        # The first play ends at 0.45 s, between 5/12 and 6/12; the pose
        # made then is first shown at 6/12.
        assert value_at(session, box, capture) == (80 if at >= Fraction(1, 2) else 0), t
        if at < Fraction(9, 20):
            shift = float(at / Fraction(9, 20))
        else:
            shift = 1 - float((at - Fraction(9, 20)) / Fraction(1, 2))
        close(
            points_at(session, art(box), capture),
            [
                s.points
                for s in art(
                    _posed(80 if at >= Fraction(1, 2) else 0).shift(
                        shift * np.array([1.0, 0, 0])
                    )
                )
            ],
        )


@pytest.mark.parametrize("fps", RATES)
def test_the_record_holds_the_mark(opened, fps):
    session = opened("Boiling", fps)
    box = box_of(session)
    capture = session.record.capture_at(Fraction(1))[1]
    state = session.record.node_state(box, capture)
    assert timing_of(box) == Timing(RATE, BOIL)
    assert timing_of(box.part("lid")) is None
    # On a capture installed, the mark is the one declared then.
    state.install(box)
    assert timing_of(box) == Timing(RATE, BOIL)


def test_a_play_s_content_key_tells_declarations_apart(opened):
    # The same rig and wait, declared at another rate or with another boil:
    # the content key of the play's start state is another.
    keys = {
        name: opened(name, 15).record.plays[0].start_state.content_key()
        for name in (
            "KeyedAtRate",
            "KeyedAtOtherRate",
            "KeyedWithBoil",
            "KeyedWithOtherBoil",
        )
    }
    assert len(set(keys.values())) == len(keys), keys
    # And the same declaration keys alike.
    assert (
        opened("KeyedAtRate", 30).record.plays[0].start_state.content_key()
        == (keys["KeyedAtRate"])
    )


def test_a_morph_of_a_declared_rig_is_held(opened):
    # MorphParts suspends the stand-in it puts in the rig's place: the
    # engine's suspension, which is not refused.
    session = opened("MorphedRig", 15)
    assert [play.kind for play in session.record.plays] == ["play", "wait"]


def test_an_engine_transform_growing_a_declared_family_is_held(opened):
    # The second transform adds a shape to the group as it begins and
    # suspends it: the engine's suspension, which is not refused.
    session = opened("GrownByTransform", 15)
    shapes = next(m for m in session.record.mobjects() if timing_of(m) is not None)
    assert timing_of(shapes) == Timing(RATE)
    assert len(shapes.submobjects) == 2


@pytest.mark.parametrize("fps", RATES)
def test_a_copy_of_a_declared_rig_carries_its_timing_into_the_scene(opened, fps):
    session = opened("CopiedBox", fps)
    twin = next(
        m for m in session.record.mobjects() if getattr(m, "name", None) == "twin"
    )
    assert timing_of(twin) == Timing(RATE)
    for t, capture in frame_captures(session, fps):
        assert value_at(session, twin, capture) == pytest.approx(
            OPEN * float(boundary(t))
        ), t


@pytest.mark.parametrize("fps", RATES)
def test_a_rig_declared_again_takes_its_new_rate_from_then(opened, fps):
    session = opened("Redeclared", fps)
    box = box_of(session)
    for t, capture in frame_captures(session, fps):
        if t < Fraction(1, 2):
            want = 100 * float(boundary(t))
        else:
            # Four drawings a second from 0.5 s: 0.5, 0.75.
            want = 50 + 120 * float(boundary(t, 4) - Fraction(1, 2))
        assert value_at(session, box, capture) == pytest.approx(want), t


def test_two_declarations_over_one_family_are_a_plain_error(opened):
    with pytest.raises(ValueError, match="shares its family"):
        opened("NestedDeclarations", 15)


def test_two_declared_families_sharing_a_member_are_the_same_error(opened):
    with pytest.raises(ValueError, match="shares its family"):
        opened("SharedMember", 15)


def test_a_member_joining_a_second_declared_family_is_the_same_error(opened):
    with pytest.raises(ValueError, match="shares its family"):
        opened("MemberJoinsSecondFamily", 15)


@pytest.mark.parametrize("fps", RATES)
def test_a_succession_s_later_part_begins_from_what_is_true(opened, fps):
    # The second move begins at 0.45 s, between two boundaries, while the
    # drawing on display is the first move's at 5/12 with its boil: it
    # starts from where the first move left the rig, and no boil is kept.
    session = opened("SuccessionOfMoves", fps)
    twin = opened("SuccessionOfMovesUndeclared", 60)
    box, true_box = box_of(session), box_of(twin)
    for t, capture in frame_captures(session, fps):
        true = points_at(twin, art(true_box), twin.record.capture_at(boundary(t))[1])
        step = BOIL.step(t)
        want = [p + BOIL._wobble(step)(p) for p in true]
        close(points_at(session, art(box), capture), want, 1e-9)
    final = session.record.state_at(session.record.duration)
    assert final.named("box").scene_box == pytest.approx(
        twin.record.state_at(twin.record.duration).named("box").scene_box, abs=1e-12
    )


@pytest.mark.parametrize("fps", RATES)
def test_state_at_answers_the_last_frame_instant_of_the_grid_and_says_so(opened, fps):
    asked = Fraction(1, 12)
    state = opened("PosedBox", fps).state_at(asked)
    # A 15 and a 30 frames a second record answer from 1/15, the drawing of
    # the boundary 0; a 60 from 1/12 itself, the drawing of 1/12: both right.
    want = {15: Fraction(1, 15), 30: Fraction(1, 15), 60: Fraction(1, 12)}[fps]
    assert state.as_of == want
    drawn = _posed(OPEN * boundary(want))
    assert state.named("box").scene_box == pytest.approx(scene_box_of(drawn), abs=1e-9)
    if fps == 60:
        assert state.named("box").scene_box != pytest.approx(
            scene_box_of(_posed(0)), abs=1e-3
        )
