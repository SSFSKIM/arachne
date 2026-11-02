"""The stepping tick: a ticked play runs every callback at the ticks a frame
reads, its events, its last tick (and throughout a play with a stop
condition); every other tick steps only what integrates, the scene's
updaters, all interpolation, and the dt-free callbacks those may read.

What a skipped callback computes at a frame is then what CE computes: CE
calls it once per frame. History-free work comes out the same either way;
work that keeps history comes out as CE's per-frame evaluation does.
"""

from __future__ import annotations

import functools
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arachne import Dot, VGroup
from arachne.timeline.clock import frames_in
from arachne.timeline.ticking import (
    STRUCTURAL,
    UNBOUNDED,
    TickPlan,
    loads_dt,
    unaffected,
)

from . import stepping_scenes
from .conftest import render

STEPPING_SCENES = Path(stepping_scenes.__file__)

COLLAPSE_WARNING = "ignore:put_start_and_end_on has been called:UserWarning"


def _play(scene):
    (play,) = [p for p in scene.timeline.ensure_record().plays if p.kind == "play"]
    return play


def _same_as_every_tick(session_of, name: str, frames, fps: int) -> None:
    """``frames`` equal those of a session that runs every callback at every
    tick (no grid): skipping changed no frame.
    """
    session = session_of(name, STEPPING_SCENES, grid=None, frame_rate=fps)
    assert all(p.stepping_ticks == 0 for p in session.record.plays)
    for k, frame in enumerate(frames):
        pixels = np.asarray(session.frame(Fraction(k, fps)))
        assert np.array_equal(frame, pixels), (name, fps, k)


# What a stepping tick runs


@pytest.mark.parametrize("fps", [4, 15])
def test_dt_free_work_that_keeps_no_history_steps_and_changes_no_frame(session_of, fps):
    scene, frames = render(stepping_scenes.Unchanged, fps)
    play = _play(scene)
    assert play.route == "ticked"
    plan = play.tick_plan
    # The follower and the redraw wait for the frames; the clock and the
    # spinner integrate.
    assert len(plan.skipped) == 2
    assert plan.stepping == 2
    assert plan.pulled == 0
    assert play.full_ticks == fps
    assert play.stepping_ticks == 60 - fps
    _same_as_every_tick(session_of, "Unchanged", frames, fps)


@pytest.mark.parametrize("fps", [4, 15])
def test_a_trace_keeps_the_updaters_it_reads_on_the_tick(session_of, fps):
    """One trace reads its dot through a bound method, the other through a
    lambda: both dots' dt-free updaters step with them.
    """
    scene, frames = render(stepping_scenes.TracedReader, fps)
    plan = _play(scene).tick_plan
    assert plan.pulled == 2
    assert plan.skipped == frozenset()
    _same_as_every_tick(session_of, "TracedReader", frames, fps)


@pytest.mark.filterwarnings(COLLAPSE_WARNING)
def test_a_collapse_a_stepping_reader_sees_is_kept_at_every_rate():
    """``route_scenes.Collapse`` misses the collapse at 4 fps, as CE does; a
    trace of the path's end keeps its updater on the tick, so 4 fps sees
    the path collapse at 0.3 s as 60 fps does.
    """
    slow, _ = render(stepping_scenes.TracedCollapse, 4)
    fast, _ = render(stepping_scenes.TracedCollapse, 60)
    path = slow.mobjects[0]
    assert np.ptp(path.points, axis=0).max() == 0
    assert np.array_equal(path.points, fast.mobjects[0].points)


@pytest.mark.parametrize("fps", [4, 15])
def test_interpolation_runs_at_every_tick(fps):
    calls = stepping_scenes.EveryTick.calls
    calls.clear()
    scene, _ = render(stepping_scenes.EveryTick, fps)
    play = _play(scene)
    assert play.stepping_ticks == 60 - fps
    # Beginning the animation, the play's start, its 60 ticks, finishing it.
    assert len(calls) == 1 + 1 + 60 + 1


# A dt parameter the code never reads


def test_an_updater_integrates_only_when_its_code_reads_dt():
    def reads(m, dt):
        m.shift(dt)

    def reads_inside(m, dt):
        return [dt * k for k in range(2)]

    def ignores(m, dt):
        return m

    assert loads_dt(reads)
    assert loads_dt(reads_inside)
    assert loads_dt(lambda m, dt: m.shift(dt))
    assert not loads_dt(ignores)
    assert not loads_dt(lambda m, dt: m)
    assert not loads_dt(lambda m: m)
    # Code it can't read counts as reading it.
    assert loads_dt(functools.partial(reads, None))


@pytest.mark.parametrize("fps", [4, 15])
def test_a_guard_that_never_reads_dt_runs_with_the_redraw_it_checks(fps):
    """The guard runs after the redraw it checks and fails on a stale one:
    stepping alone, it would see a dot interpolation moved and a redraw
    left for the frame. Dead in dt, it runs at the full ticks only.
    """
    calls = stepping_scenes.StaleGuard.calls
    calls.clear()
    scene, _ = render(stepping_scenes.StaleGuard, fps)
    play = _play(scene)
    assert play.stepping_ticks == 60 - fps
    assert play.full_ticks == fps
    # The play's start, its full ticks, and the update after it finishes.
    assert len(calls) == 1 + fps + 1


# Callbacks that keep history


def _shown(state) -> list[tuple]:
    return [
        (node.type, node.z, node.box, node.color)
        for node in state.nodes
        if node.box is not None
    ]


def test_redraws_that_keep_history_are_built_at_the_frames_as_in_ce(session_of):
    """Each staircase's target keeps the members ``become`` padded it with, so
    what it shows depends on the instants it was built at. At 4 fps the
    record builds them at the frames, as CE does (the reference builds
    them by hand at the frames); built at every tick, they show otherwise.
    """
    fps = 4
    stepped = session_of("Staircases", STEPPING_SCENES, frame_rate=fps)
    everything = session_of("Staircases", STEPPING_SCENES, grid=None, frame_rate=fps)
    reference = session_of("StaircasesPerFrame", STEPPING_SCENES, frame_rate=fps)
    play = stepped.record.plays[-1]
    assert play.stepping_ticks == 60 - fps
    instants = frames_in(play.start, play.end, Fraction(fps))
    by_hand = [p.start for p in reference.record.plays]
    assert len(instants) == len(by_hand) == fps
    moved = {"low": False, "high": False}
    for t, at in zip(instants, by_hand, strict=True):
        shown = _shown(stepped.state_at(t))
        assert shown == _shown(reference.state_at(at)), t
        other = _shown(everything.state_at(t))
        for name, upper in (("low", True), ("high", False)):
            part = [n for n in shown if (n[2][1] > 0) == upper]
            if part != [n for n in other if (n[2][1] > 0) == upper]:
                moved[name] = True
    assert moved == {"low": True, "high": True}


# Reads that can't be bounded


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("DynamicRead", "uses getattr"),
        ("AliasedLookup", "holds getattr as a value"),
        ("OwnAnimationScene", "OwnAnimation is the scene's own animation class"),
    ],
)
def test_a_play_whose_reads_cant_be_bounded_ticks_everything(session_of, name, why):
    play = session_of(name, STEPPING_SCENES, frame_rate=4).record.plays[-1]
    assert play.route == "ticked"
    (reason,) = [r for r in play.admission.reasons if r.startswith(UNBOUNDED)]
    assert reason.startswith(f"{UNBOUNDED}: ")
    assert why in reason
    assert play.tick_plan.unbounded == reason
    assert play.stepping_ticks == 0
    assert play.full_ticks == 60


def test_getattr_by_a_constant_name_is_a_bounded_read(session_of):
    play = session_of("NamedRead", STEPPING_SCENES, frame_rate=4).record.plays[-1]
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.stepping_ticks == 60 - 4


def test_an_updater_that_joins_at_an_event_is_planned_for(session_of):
    """A Succession part begins at 0.5 s and copies a dot whose updater a
    trace reads: the event tick plans again, and stepping goes on.
    """
    fps = 4
    scene, frames = render(stepping_scenes.LateJoiner, fps)
    play = _play(scene)
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.stepping_ticks > 0
    _same_as_every_tick(session_of, "LateJoiner", frames, fps)


# Sessions


def test_a_session_records_on_its_render_s_grid(session_of):
    """By default a session steps as the render does, so its frames are the
    render's; without a grid every tick is full.
    """
    fps = 4
    _, frames = render(stepping_scenes.Staircases, fps)
    session = session_of("Staircases", STEPPING_SCENES, frame_rate=fps)
    assert session.record.plays[-1].stepping_ticks == 60 - fps
    for k, frame in enumerate(frames):
        assert np.array_equal(frame, np.asarray(session.frame(Fraction(k, fps)))), k


# Reads through what a stepping callback is handed, and through interpolation


@pytest.mark.parametrize(
    "name",
    ["AttributeChain", "PathReader", "FunctionReader"],
)
@pytest.mark.parametrize("fps", [4, 15])
def test_a_stepping_reader_keeps_what_it_reads_through_on_the_tick(
    session_of, name, fps
):
    """A dt updater reads a dot through an attribute of its own mobject,
    through ``MoveAlongPath``'s path or through an ``UpdateFromFunc``: the
    dt-free updater moving what it reads steps with it, and only the redraw
    nothing stepping reads waits for the frames.
    """
    scene, frames = render(getattr(stepping_scenes, name), fps)
    play = _play(scene)
    assert play.tick_plan.pulled == 1
    assert len(play.tick_plan.skipped) == 1
    assert play.stepping_ticks == 60 - fps
    _same_as_every_tick(session_of, name, frames, fps)


def test_a_fade_beside_a_dt_updater_that_reads_it_not_stays_stepping():
    scene, _ = render(stepping_scenes.FadeBeside, 4)
    play = _play(scene)
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.tick_plan.pulled == 0
    assert play.tick_plan.skipped
    assert play.stepping_ticks == 60 - 4


# The switch a differential sweep compares with


def _same_as_full_ticks(session_of, name: str, fps: int):
    """Every capture a stepping session keeps equals the one a session that
    ticks everything keeps at the same instant; the stepping session.
    """
    stepped = session_of(name, STEPPING_SCENES, frame_rate=fps)
    full = session_of(name, STEPPING_SCENES, frame_rate=fps, stepping=False)
    assert all(
        p.stepping_ticks == 0 and all(not plan.applied for _, plan in p.tick_plans)
        for p in full.record.plays
    )
    for ours, theirs in zip(stepped.record.plays, full.record.plays, strict=True):
        assert ours.sample_keys() == theirs.sample_keys(), (name, fps, ours.index)
    return stepped


@pytest.mark.parametrize("fps", [4, 15])
def test_stepping_off_ticks_everything_on_the_same_grid(session_of, fps):
    stepped = _same_as_full_ticks(session_of, "Unchanged", fps)
    assert stepped.record.plays[-1].stepping_ticks == 60 - fps


def test_the_captures_tell_history_kept_at_the_frames_from_every_tick(session_of):
    """Redraws that keep history differ between the two runs by design: CE
    builds them at its frames, which stepping keeps.
    """
    stepped = session_of("Staircases", STEPPING_SCENES, frame_rate=4)
    full = session_of("Staircases", STEPPING_SCENES, frame_rate=4, stepping=False)
    ours, theirs = stepped.record.plays[-1], full.record.plays[-1]
    assert [t for t, _ in ours.sample_keys()] == [t for t, _ in theirs.sample_keys()]
    assert ours.sample_keys() != theirs.sample_keys()


# Every path to what a stepping reader reads: resolved to a mobject's family,
# or counted as reaching anything


@pytest.mark.parametrize(
    "name",
    [
        "OwnerWrite",
        "ContainerWrite",
        "ReusedFactory",
        "ClassRead",
        "ArrayAlias",
        "ModuleAlias",
        "ConstantWrite",
    ],
)
@pytest.mark.parametrize("fps", [4, 15])
def test_a_producer_reached_through_any_path_steps_with_its_reader(
    session_of, name, fps
):
    """Through an attribute of the mobject it updates, a list it makes, a
    function another shares its code with, a class attribute, a view of an
    array or a module imported under another name: the dt-free updater
    stepping work reads steps, the dot nothing reads waits for the frames.
    """
    play = _same_as_full_ticks(session_of, name, fps).record.plays[-1]
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.tick_plan.pulled >= 1
    assert play.tick_plan.skipped
    assert play.stepping_ticks == 60 - fps


@pytest.mark.parametrize("name", ["PropertyRead", "MethodRead", "GroupRead"])
def test_a_read_of_another_mobject_pulls_only_its_producer(session_of, name):
    """A property, a method or a group's centre of a mobject the reader
    closes over: its producer steps, the idle dot does not.
    """
    play = _same_as_full_ticks(session_of, name, 4).record.plays[-1]
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert play.tick_plan.pulled == 1
    assert len(play.tick_plan.skipped) == 1
    assert play.stepping_ticks == 60 - 4


@pytest.mark.parametrize("name", ["AlphaReader", "TargetReader", "WiggleReader"])
@pytest.mark.parametrize("fps", [4, 15])
def test_interpolation_a_stepping_reader_reads_keeps_its_inputs_on_the_tick(
    session_of, name, fps
):
    """An ``UpdateFromAlphaFunc`` following a dot, a ``Transform`` into one,
    a ``Wiggle`` of a mobject its updater moves: what the interpolation reads
    steps with it, and the idle dot still waits for the frames.
    """
    play = _same_as_full_ticks(session_of, name, fps).record.plays[-1]
    assert not any(r.startswith(UNBOUNDED) for r in play.admission.reasons)
    assert len(play.tick_plan.skipped) == 1
    assert play.stepping_ticks > 0


def test_the_stepping_families_hold_what_stepping_reads_and_writes(session_of):
    session = session_of("MethodRead", STEPPING_SCENES, frame_rate=4)
    _, other, output, idle = session.scene.mobjects
    families = session.record.plays[-1].tick_plan.stepping_families()
    assert {id(other), id(output)} <= families
    assert id(idle) not in families


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("NewReader", "UpdateFromAlphaFunc"),
        ("ReaderRemoved", "UpdateFromAlphaFunc"),
        ("HiddenInstaller", "UpdateFromAlphaFunc"),
    ],
)
def test_a_play_whose_callbacks_change_its_updaters_ticks_everything(
    session_of, name, why
):
    """A callback that may add or remove an updater as the play runs would
    leave a stepping tick to meet work its plan never weighed, or keep
    stepping work its plan weighed gone: the play ticks everything.
    """
    play = _same_as_full_ticks(session_of, name, 4).record.plays[-1]
    (reason,) = [r for r in play.admission.reasons if r.startswith(STRUCTURAL)]
    assert reason.startswith(f"{STRUCTURAL}: {why} ")
    assert play.tick_plan.unbounded == reason
    assert play.stepping_ticks == 0
    assert play.full_ticks == 60


@pytest.mark.parametrize(
    "name", ["SkippedAdder", "MemberJoins", "SubmobjectAdded", "SubmobjectDropped"]
)
def test_a_skipped_callback_that_changes_the_structure_is_never_stepped_past(
    session_of, name
):
    """It adds a mobject carrying a dt updater, adds or drops a group's
    member carrying one, or puts into a group a reader reads a dot another
    callback moves: the play plans again after the frame it does so at, or
    it ticks everything for it from the start (when stepping work reads what
    it writes). Either way the frames are those of ticking everything.
    """
    play = _same_as_full_ticks(session_of, name, 15).record.plays[-1]
    first = play.tick_plans[0][1]
    if first.steps_only:
        assert [t for t, _ in play.tick_plans] == [play.start, Fraction(13, 60)]
    else:
        assert first.unbounded.startswith(f"{STRUCTURAL}: updater ")


def test_a_skipped_callback_that_adds_a_reader_has_the_play_plan_again(
    session_of,
):
    """It runs at full ticks only, so the reader it adds at a frame is
    planned for before the next stepping tick: the new plan finds the adder
    itself writing what the reader reads, and ticks everything from there.
    """
    play = _same_as_full_ticks(session_of, "SkippedInstaller", 15).record.plays[-1]
    # Before the frame at 0.2 s: its 12 ticks less the frames at 0, 1/15, 2/15.
    assert play.stepping_ticks == 12 - 3
    (reason,) = [r for r in play.admission.reasons if r.startswith(STRUCTURAL)]
    assert "updater install" in reason
    # The plans it stepped by: its first, then ticking everything from the
    # tick after the frame at 0.2 s.
    (start, first), (then, last) = play.tick_plans
    assert (start, then) == (play.start, Fraction(13, 60))
    assert first.steps_only
    assert last.unbounded == reason
    assert play.tick_plan is last


def test_a_guard_calling_getters_it_holds_steps_with_what_it_checks(session_of):
    play = _same_as_full_ticks(session_of, "GetterGuard", 4).record.plays[-1]
    assert not any(r.startswith("tick.") for r in play.admission.reasons)
    assert play.tick_plan.pulled == 1
    assert len(play.tick_plan.skipped) == 1
    assert play.stepping_ticks == 60 - 4


# Structure that changes where no plan looks; names that alias an operand;
# writes through a class or a setter


@pytest.mark.parametrize("name", ["StartInstall", "AliasAppend", "AliasAugment"])
def test_structure_changed_before_a_stepping_tick_plans_the_play_again(
    session_of, name
):
    """A skipped callback gives a dot a dt reader as the play first
    evaluates, or appends one to a group through a local name for its
    members: the next stepping tick finds the structure changed and plans
    again before it steps, so no step integrates stale input.
    """
    play = _same_as_full_ticks(session_of, name, 15).record.plays[-1]
    assert len(play.tick_plans) >= 2


@pytest.mark.parametrize("name", ["BoolAlias", "IfAlias"])
def test_a_name_bound_to_an_operand_changes_what_it_names(session_of, name):
    """``x = a or a`` and ``x = a if c else b`` bind ``x`` to an operand, not
    a new value: ``x += 1`` then changes in place something the plan can't
    name, which may be anything (the array a dt updater reads, a mobject's
    members), and the play ticks everything.
    """
    play = _same_as_full_ticks(session_of, name, 4).record.plays[-1]
    assert play.tick_plan.unbounded.startswith(f"{STRUCTURAL}: updater update ")


def test_become_padding_with_copies_that_update_plans_again(session_of):
    """``become`` pads a dot with copies of itself, its updater with them:
    the play plans for the copies before the next stepping tick, which skips
    them, so they move once a frame (as in CE) rather than once a tick, and
    no stepping tick meets work it never weighed.
    """
    session = session_of("BecomeRoot", STEPPING_SCENES, frame_rate=4)
    play = session.record.plays[-1]
    assert not any(r.startswith("tick.") for r in play.admission.reasons)
    assert [t for t, _ in play.tick_plans] == [play.start, Fraction(16, 60)]
    # Made at the frame at 0.25 s, the copies move there and at the frame at
    # 0.5 s: as a scene doing the same per frame by hand has them.
    box = session.state_at(Fraction(1, 2)).named("root").box
    assert (box[0] + box[2]) / 2 == pytest.approx(2 * 0.01)


@pytest.mark.parametrize("name", ["SkippedInstaller", "SubmobjectAdded", "Unchanged"])
def test_a_run_that_ticks_everything_plans_where_stepping_would(session_of, name):
    """With stepping off, a play still makes its plans, and makes them
    again at the ticks a stepping run would, for a comparison to read; it
    applies none (every tick full, no reason recorded on its admission).
    """
    stepped = session_of(name, STEPPING_SCENES, frame_rate=15).record.plays[-1]
    full = session_of(
        name, STEPPING_SCENES, frame_rate=15, stepping=False
    ).record.plays[-1]
    assert [t for t, _ in full.tick_plans] == [t for t, _ in stepped.tick_plans]
    for (_, ours), (_, theirs) in zip(stepped.tick_plans, full.tick_plans, strict=True):
        assert ours.applied
        assert not theirs.applied
        assert len(theirs.skipped) == len(ours.skipped)
        assert theirs.unbounded == ours.unbounded
    assert full.stepping_ticks == 0
    assert not any(r.startswith("tick.") for r in full.admission.reasons)


def test_on_step_sees_every_stepping_tick_and_no_full_one(session_of):
    """The diagnostic hook is called after each stepping tick, with the
    scene and the tick, and after no full tick.
    """
    seen: list[Fraction] = []
    session = session_of(
        "Unchanged",
        STEPPING_SCENES,
        frame_rate=4,
        on_step=lambda scene, tick: seen.append(tick),
    )
    play = session.record.plays[-1]
    assert len(seen) == play.stepping_ticks == 60 - 4
    assert not set(seen) & {Fraction(k, 4) for k in range(5)}
    full = session_of(
        "Unchanged",
        STEPPING_SCENES,
        frame_rate=4,
        stepping=False,
        on_step=lambda scene, tick: seen.append(tick),
    )
    assert full.record.plays[-1].stepping_ticks == 0
    assert len(seen) == 60 - 4


def _same_frames(session_of, name: str, fps: int):
    """Every frame of a stepping session equals that of one that ticks
    everything; the stepping session's last play.
    """
    stepped = session_of(name, STEPPING_SCENES, frame_rate=fps)
    full = session_of(name, STEPPING_SCENES, frame_rate=fps, stepping=False)
    for k in range(fps + 1):
        at = Fraction(k, fps)
        assert np.array_equal(
            np.asarray(stepped.frame(at)), np.asarray(full.frame(at))
        ), (name, at)
    return stepped.record.plays[-1]


@pytest.mark.parametrize("name", ["EventAdd", "AddRemove", "GroupBecome"])
def test_structure_a_plan_can_see_change_keeps_stepping_exact(session_of, name):
    """A part beginning adds a reader (the play plans again at its event);
    an add and a remove within one run leave nothing to plan for; a group
    ``become`` pads with copies of a member that carry its updater (the play
    plans again before the next stepping tick). The frames are those of
    ticking everything, and no tick everything was needed.
    """
    play = _same_frames(session_of, name, 4)
    assert not any(r.startswith("tick.") for r in play.admission.reasons)
    assert play.stepping_ticks > 0


def test_an_attribute_an_instance_shadows_steps_with_what_it_reads(session_of):
    """Rebinding the attribute on the instance a dt updater reads it on:
    the plan pulls the rebinding and the tracker it binds, and steps.
    """
    play = _same_as_full_ticks(session_of, "InstanceShadow", 4).record.plays[-1]
    assert play.tick_plan.steps_only


def test_a_member_swapped_where_no_stepping_reaches_keeps_the_plan(session_of):
    """A skipped redraw replaces a group's member at every frame, and
    nothing stepping reads the group: the plan stays true, and is kept.
    """
    play = _same_as_full_ticks(session_of, "LabelSwap", 4).record.plays[-1]
    assert len(play.tick_plans) == 1
    assert play.stepping_ticks == 60 - 4


def test_a_member_moved_into_what_stepping_reaches_is_a_change_to_plan_for():
    """A dot moved from a group nothing stepping reaches into one it does
    joins no family of the scene's anew, and still leaves the plan untrue;
    a dot swapped inside the first group does not.
    """
    mover, kept = Dot(), Dot()
    aside, read = VGroup(mover), VGroup(kept)
    scene = SimpleNamespace(updaters=[], mobjects=[aside, read])
    members = frozenset(map(id, (aside, mover, read, kept)))
    plan = TickPlan(
        skipped=frozenset({(0, 0)}),
        stepped_ids=frozenset(map(id, (read, kept))),
        roots=(id(aside), id(read)),
        members=members,
    )
    assert unaffected(plan, scene, [])
    aside.remove(mover)
    aside.add(Dot())
    assert unaffected(plan, scene, [])
    read.add(mover)
    assert not unaffected(plan, scene, [])


def test_a_play_that_gains_its_first_stepping_work_plans_for_it(session_of):
    """The play steps nothing (no dt updater) until a skipped callback gives
    the scene an updater that integrates: the next stepping tick plans for
    it, and steps it with what it reads.
    """
    play = _same_as_full_ticks(session_of, "SceneInstaller", 4).record.plays[-1]
    (_, first), (_, then) = play.tick_plans
    assert first.stepping == 0
    assert first.steps_only
    assert then.unbounded is not None or then.pulled >= 1


@pytest.mark.parametrize("name", ["NativeAlias", "EngineMethodAlias"])
def test_a_method_held_as_a_value_changes_what_it_is_bound_to(session_of, name):
    """``push = values.append`` or ``move = dot.set_x``, called under its own
    name: the call changes the list or the dot, which a dt updater reads,
    and the plan pulls the callback making it.
    """
    play = _same_as_full_ticks(session_of, name, 4).record.plays[-1]
    assert play.tick_plan.pulled == 1


@pytest.mark.parametrize("name", ["ClassShadow", "SetterWrite", "OwnSetter"])
def test_a_write_through_a_class_or_a_setter_pulls_its_writer(session_of, name):
    """A store binding an attribute on a class an instance inherits it
    through, or one running a property's setter, writes what a dt reader
    reads: the plan pulls the writer.
    """
    _same_as_full_ticks(session_of, name, 4)


@pytest.mark.xfail(
    strict=True,
    reason="known residual (tech-debt-tracker.md): the write check does not read "
    "a method of an object the callback made itself",
)
def test_a_method_of_an_object_the_callback_made_is_read_for_its_writes(session_of):
    """A dt-free updater makes a helper of the scene's own class at each run
    and calls its method, which moves a tracker a dt updater integrates.
    """
    _same_as_full_ticks(session_of, "FreshMethod", 4)
