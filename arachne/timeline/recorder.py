"""The forward run of a scene, recorded as it goes.

A scene owns one :class:`Recorder`. Each ``play`` (and ``wait``) runs
through :meth:`Recorder.play`: it begins the animations as CE does, then
calls the scene's ``play_internal``, whose body (:meth:`Recorder.play_body`)
lets the play's route evaluate it and finishes it (:mod:`.hooks`), and
leaves a :class:`PlayRecord` in the scene's :class:`Record` with every
capture the record keeps.
"""

from __future__ import annotations

__all__ = ["Bridge", "BridgeCall", "PlayRun", "Reading", "Recorder", "running"]

import inspect
import math
import os
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import FrameType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from arachne._config import config, logger
from arachne.camera.projection import configured_home

from .admission import (
    PlayIdentity,
    admit,
    camera_refusal,
    play_identity,
    whole_line,
)
from .bodies import Exit, Kept, bodies
from .clock import TICK_RATE, exact, frames_in, play_end
from .dispatch import draw_refusal, scene_hooks
from .draw import CameraExtras, LiveFrames, camera_nodes
from .events import EventPlan, plan_events
from .execution import Execution, divergence
from .hooks import FRAME_HOOKS, TIME_PROGRESSION, FrameInstant, SceneHookRefused
from .inputs import Inputs, engine_dirs, source_digest, source_files
from .origins import ORIGINS
from .record import Capture, Identity, Marker, PlayRecord, Record, Removal, States
from .reference import BRUTE_FORCE
from .restore import RestorePoint, restore_point
from .routes import ROUTE_CLASSES, Route, snapshot_animations
from .source import (
    SourceLine,
    names_from_frames,
    relative_path,
    scene_file_line,
    scene_frames,
    scene_location,
)
from .ticking import (
    TickPlan,
    current_structure,
    plan_ticks,
)

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene


#: The recorders whose scene's script is running, the innermost last.
_RUNNING: list[Recorder] = []


def running() -> Recorder | None:
    """The recorder of the scene whose script is running now, if any."""
    return _RUNNING[-1] if _RUNNING else None


@dataclass(frozen=True)
class Reading:
    """One evaluation of a bridge call: its instant, the scene file's line
    running then (``None`` where none of its frames was on the stack), the
    values the call's code measures (``None`` when there was nothing to
    measure: a probe that gave no point), and ``until``, the instant the
    state it read holds until: the frames in ``[t, until)`` show it.
    ``None`` for a reading at one instant only (a call's own, the scene's
    end).
    """

    t: Fraction
    line: SourceLine | None
    values: Mapping[str, Any] | None
    until: Fraction | None = None


@dataclass(frozen=True, eq=False)
class BridgeCall:
    """One call of the assertion bridge (:mod:`arachne.lint.bridge`), as the
    record keeps it.

    ``kind`` is ``"record"`` (a check at one instant), ``"watch"`` (a check
    at every kept capture from the call on), ``"clear"`` (a watch stopped:
    its operand is the watch's index among the calls), ``"sweep"`` and
    ``"apart"`` (checks of the scene as it stands at the call),
    ``"declare"`` (terms for the lint's rules) or ``"arm"`` (terms, and the
    checks made over time asked for). ``t`` is the scene time of the call
    and ``play`` the play it belongs to: the one running, or between plays
    the next to begin (one past the last after it). ``line`` is the call
    site, the scene file's line. ``operands`` are the points as values and
    the mobjects by stable id (a watched function has none: its values are
    its readings); ``reference`` the reference mobject's stable id.
    ``readings`` are the call's evaluations in time order: one for a
    record, a sweep or an apart (its findings), one at the call and then one
    per kept capture for a watch, and for a call stating exemptions what
    they name, at the call and then at each kept capture.
    """

    kind: str
    t: Fraction
    play: int
    line: SourceLine | None
    code: str | None = None
    operands: tuple[Any, ...] = ()
    reference: str | None = None
    margin: float = 0.0
    terms: Mapping[str, Any] = field(default_factory=dict)
    #: How a finding names what was checked.
    subject: str = ""
    readings: list[Reading] = field(default_factory=list)

    def reading_at(self, t: Fraction) -> Reading | None:
        """The last reading at or before ``t``: what was on screen then."""
        found = None
        for reading in self.readings:
            if reading.t > t:
                break
            found = reading
        return found


class Bridge:
    """The assertion bridge's calls of one recording, in call order, and
    those it reads at each kept capture.
    """

    def __init__(self) -> None:
        self.calls: list[BridgeCall] = []
        self._readers: list[
            tuple[BridgeCall, Callable[[], Mapping[str, Any] | None]]
        ] = []
        #: The calls (by id) whose unchanged readings are not kept.
        self._changes_only: set[int] = set()

    def add(
        self,
        call: BridgeCall,
        reader: Callable[[], Mapping[str, Any] | None] | None = None,
        *,
        changes_only: bool = False,
    ) -> BridgeCall:
        """Keep ``call``; with ``reader``, a call read at every kept capture
        from now on: a watch, or what a declaration's exemptions name. With
        ``changes_only``, a reading equal to the call's last is not kept.
        """
        self.calls.append(call)
        if reader is not None:
            self._readers.append((call, reader))
            if changes_only:
                self._changes_only.add(id(call))
        return call

    def of(self, play: int) -> tuple[BridgeCall, ...]:
        """The calls that belong to the play of index ``play``."""
        return tuple(call for call in self.calls if call.play == play)

    @property
    def watches(self) -> tuple[BridgeCall, ...]:
        """The watches, in the order they were declared."""
        return tuple(call for call in self.calls if call.kind == "watch")

    @property
    def watching(self) -> bool:
        """Whether a call is read at the kept captures."""
        return bool(self._readers)

    def read_now(self, call: BridgeCall) -> Mapping[str, Any] | None:
        """What ``call``'s reader reads now (``None`` for a call with none):
        for a call that read the scene, as it stands at another call.
        """
        for kept, reader in self._readers:
            if kept is call:
                return reader()
        return None

    def stop(self, call: BridgeCall) -> None:
        """Read the watch ``call`` at no capture from now on."""
        self._readers = [(kept, r) for kept, r in self._readers if kept is not call]

    def read(
        self, t: Fraction, line: SourceLine | None, until: Fraction | None = None
    ) -> None:
        """Read every call with a reader made at or before ``t`` there, in a
        state that holds until ``until``.
        """
        for call, reader in self._readers:
            if call.t <= t:
                self._read(call, reader, t, line, until)

    def read_changes(self, t: Fraction) -> None:
        """Read, at ``t``, every call read only where its reading changes
        (what a declaration's exemptions name): at a state the record keeps
        between frames, where a watch is never read.
        """
        for call, reader in self._readers:
            if call.t <= t and id(call) in self._changes_only:
                self._read(call, reader, t, None, None)

    def _read(
        self,
        call: BridgeCall,
        reader: Callable[[], Mapping[str, Any] | None],
        t: Fraction,
        line: SourceLine | None,
        until: Fraction | None,
    ) -> None:
        values = reader()
        if (
            id(call) in self._changes_only
            and call.readings
            and call.readings[-1].values == values
        ):
            return
        call.readings.append(Reading(t, line, values, until))


class PlayRun:
    """One play while its route evaluates it."""

    def __init__(
        self,
        recorder: Recorder,
        play: PlayRecord,
        route: Route,
        live: LiveFrames | None = None,
    ) -> None:
        self.recorder = recorder
        self.scene = recorder.scene
        self.play = play
        self.route = route
        self.start = play.start
        self.requested_end = play.requested_end
        self.held = play.held
        #: Where the play ended; the route sets it.
        self.end = play.requested_end
        self.live = live
        self._extras: CameraExtras | None = None
        #: Whether the play plans its ticks once evaluated at its start.
        self.plans = False
        #: The play's last frame-read instant, once it has one.
        self._last_frame: Fraction | None = None
        #: A state-function play's last restore point, which the next shares
        #: what is unchanged with.
        self._point: RestorePoint | None = None

    def set_clock(self, instant: Fraction) -> None:
        """Make the scene clock read ``instant``, drawing first whatever frames
        a live drawing still owes the instant before it.
        """
        if self.live is not None:
            self.live.flush(instant)
        self.scene.renderer.time = instant

    def frame(self, plan: EventPlan, instant: Fraction, dt: float) -> None:
        """Evaluate the play at ``instant``, which a frame reads, as CE
        evaluates a frame: through the scene's ``update_to_time``, shown the
        play-local time and the time since the play's last frame, its body
        evaluating the instant with the play's own ``dt`` (:mod:`.hooks`).
        """
        last = self._last_frame
        self._last_frame = instant
        frame = FrameInstant(
            plan,
            self.start,
            instant,
            dt,
            0.0 if last is None else float(instant - last),
        )
        recorder = self.recorder
        recorder.frame = frame
        try:
            self.scene.update_to_time(frame.t)
        finally:
            recorder.frame = None
        frame.check()

    def sample(self, instant: Fraction, *, until: Fraction, keep: bool = False) -> None:
        """The play has been evaluated at ``instant``, and holds until its next
        evaluated instant ``until``: capture it if the record keeps it (always
        when ``keep``).

        The camera's executable inputs are evaluated at every instant, kept
        or not, so a stateful one is called as often whatever the record keeps.
        """
        recorder = self.recorder
        self._extras = CameraExtras.observe(self.scene.renderer.camera, self._extras)
        if self.live is not None:
            self.live.pending(instant, until)
        if not (keep or recorder.keeps(instant, until)):
            return
        play = self.play
        capture = recorder.capture("sample", instant, play, self._extras, until)
        play.samples.append(capture)
        play._instants.append(instant)
        if not play.held:
            # What a play is evaluated from between its kept instants: a
            # trusted play's animations' state; a state-function play's
            # restore point (:mod:`.restore`), its animations' state included.
            if play.route == "trusted":
                play._animation_states[capture.seq] = snapshot_animations(
                    play._animations
                )
            elif play.route == "state":
                self._point = restore_point(self.scene, play, self._point)
                play._animation_states[capture.seq] = self._point
        # Last, so what the scene's watches read changes nothing kept.
        recorder.read_watches(instant, until)

    def progress(self, instants: list[Fraction]) -> Any:
        """A progress bar over ``instants``."""
        return self.scene._play_progress(instants)

    def stepping_ticks(self, ticks: list[Fraction]) -> frozenset[Fraction]:
        """The ticks of a ticked play that may only step (:mod:`.ticking`):
        none with a stop condition (it is read at every tick) or a plan that
        skips nothing; else every tick but the last and those its events are
        taken at. A tick a frame reads is full as well (``keeps``).
        """
        plan = self.play.tick_plan
        if self.scene.stop_condition is not None or plan is None or not plan.steps_only:
            return frozenset()
        full = {
            Fraction(math.ceil(event * TICK_RATE), TICK_RATE)
            for event in self.play.events
        }
        if ticks:
            full.add(ticks[-1])
        return frozenset(ticks) - full

    def stepping_plan(self, tick: Fraction) -> TickPlan | None:
        """The plan stepping tick ``tick`` steps by, or ``None`` when it runs
        in full. Before it steps, the structure is compared with the one the
        plan was made for (:func:`.current_structure`): anything that changed
        it since (an updater added or taken, a member joining or leaving, by
        whatever code) has the play plan again first. The new plan finds a
        play whose updaters no longer run where a tick could leave one out,
        which then runs this tick and every later one in full; otherwise the
        tick steps by the new plan, as it would have by the old: a structure
        change alone makes no tick full. A plan not
        :attr:`~.TickPlan.applied` is remade the same way, and the tick runs
        in full.
        """
        play = self.play
        plan = play.tick_plan
        if plan is None or not plan.steps_only:
            return None
        structure = current_structure(self.scene, play._animations)
        if structure != plan.structure:
            plan = plan_ticks(self.scene, list(play._animations))
            if plan.unbounded is not None:
                self._unbounded(plan.unbounded, tick)
                return None
            self._set_plan(plan, tick)
        return plan if plan.steps_only and self.recorder.stepping else None

    def _set_plan(self, plan: TickPlan, first: Fraction) -> None:
        self.play.set_tick_plan(
            plan if self.recorder.stepping else replace(plan, applied=False), first
        )

    def started(self) -> None:
        """The play has been evaluated at its start: show that state to the
        recorder's ``on_start`` hook, then plan which of its ticks only step
        (:func:`.plan_ticks`), from whatever the hook left. A run that ticks
        everything makes the plan all the same, for a comparison to read,
        and applies none of it.
        """
        recorder = self.recorder
        if recorder.on_start is not None:
            recorder.on_start(self.scene, self.start)
        if not self.plans:
            return
        play = self.play
        plan = plan_ticks(self.scene, list(play._animations))
        self._set_plan(plan, self.start)
        if plan.unbounded is not None and recorder.stepping:
            assert play.admission is not None
            play.admission = replace(
                play.admission, reasons=(*play.admission.reasons, plan.unbounded)
            )

    def stepped(self, tick: Fraction, *, full: bool) -> None:
        """Count a tick, and show it to the recorder's diagnostic hooks
        (``on_tick`` every tick, ``on_step`` a stepping one).
        """
        play = self.play
        recorder = self.recorder
        if recorder.on_tick is not None:
            recorder.on_tick(self.scene, tick, full)
        if full:
            play.full_ticks += 1
            return
        play.stepping_ticks += 1
        if recorder.on_step is not None:
            recorder.on_step(self.scene, tick)

    def _unbounded(self, reason: str, first: Fraction) -> None:
        """Tick everything from tick ``first``, for ``reason`` (the play's
        admission records it when the plan is applied).
        """
        play = self.play
        self._set_plan(TickPlan(unbounded=reason), first)
        if play.admission is not None and self.recorder.stepping:
            play.admission = replace(
                play.admission, reasons=(*play.admission.reasons, reason)
            )


class Recorder:
    """Records a scene's forward run.

    ``grid`` is the frame rate whose instants the record keeps (a render's),
    or ``None`` to keep every evaluated instant (a session's, which answers
    any time). ``draws`` says whether the renderer draws each play's frames
    from the record as soon as the play has run. ``stepping`` says how a
    ticked play runs its ticks no frame reads: ``True``, they only step, by
    the play's plan (:mod:`.ticking`); ``False``, every tick is full, its
    plays still making their plans, not applied (:attr:`.TickPlan.applied`);
    :data:`~.reference.BRUTE_FORCE`, a verification mode, not a production
    path: they step as decided tick by tick apart from any plan
    (:mod:`.reference`), which a stepping run's frames must equal.

    ``on_tick`` and ``on_step`` are diagnostics, called after each tick of a
    ticked play, as ``on_tick(scene, tick, full)`` (``full`` false for a
    stepping tick), and after each stepping tick only (one that ran the
    stepping set alone), as ``on_step(scene, tick)``: a comparison observes
    there what each tick changed, the record keeping no tick a frame does
    not read. ``on_start`` is called once for each ticked play, as
    ``on_start(scene, start)``, after the play's evaluation at its start and
    before its first tick and before the play plans its ticks: the state its
    first tick steps from, and what the plan is made from, whatever the hook
    changed. Neither ``on_tick`` nor ``on_step`` is called for the start.
    Unset, they cost nothing; set, ``on_tick`` and ``on_step`` observe and
    must change nothing: nothing checks it, and the frames of a play whose
    ``on_tick`` or ``on_step`` writes are undefined (``tech-debt-tracker.md``,
    Hooks).

    ``instants`` are scene instants a trusted or state-function play holding
    one evaluates and keeps besides its own, as a non-frame evaluation (the
    engine's bodies, never the scene's frame hooks): what a restore or an
    analytic answer there is compared with. A ticked play evaluates none.

    ``demoted`` are the identities of plays the probe demoted
    (:data:`.admission.PlayIdentity`, :mod:`.probe`): the play of each, or
    every play of a line demoted whole, ticks where it would be a
    state-function play. A session passes them; a render never does.

    ``exits`` says which plays' ends the record keeps where each body stands
    at (``Record.exits``, :mod:`.bodies`), which the lint's motion rule
    measures cuts from: ``"all"`` every play's (a forward run's record);
    ``"armed"`` each play ending after the scene arms the rule
    (:meth:`arm_motion`; a session's record); ``"none"`` none, armed or not
    (a render's, which no lint reads).

    ``states`` says, alike, from when the record keeps the states the lint's
    temporal rules read (``Record.states``, :class:`.record.States`): the
    scene as the script called each play and as each play ended, and as the
    script removed from it between plays, each a capture no frame draws,
    outside the record's history of drawn captures. ``"all"`` from the start (a lint's
    forward run that a temporal rule asked for); ``"armed"`` from
    :meth:`arm_states` on; ``"none"`` never.
    """

    def __init__(
        self,
        scene: Scene,
        *,
        grid: Fraction | float | None,
        draws: bool = True,
        stepping: bool | Literal["brute_force"] = True,
        on_step: Callable[[Scene, Fraction], object] | None = None,
        on_tick: Callable[[Scene, Fraction, bool], object] | None = None,
        on_start: Callable[[Scene, Fraction], object] | None = None,
        instants: Iterable[Fraction | float] = (),
        demoted: Iterable[PlayIdentity] = (),
        exits: Literal["none", "armed", "all"] = "none",
        states: Literal["none", "armed", "all"] = "none",
    ) -> None:
        if stepping not in (True, False, BRUTE_FORCE):
            raise ValueError(
                f"stepping is True, False or {BRUTE_FORCE!r}, not {stepping!r}"
            )
        if exits not in ("none", "armed", "all"):
            raise ValueError(f"exits is 'none', 'armed' or 'all', not {exits!r}")
        if states not in ("none", "armed", "all"):
            raise ValueError(f"states is 'none', 'armed' or 'all', not {states!r}")
        self.scene = scene
        self.grid = exact(grid) if grid is not None else None
        self.draws = draws
        self.stepping = stepping
        self.on_step = on_step
        self.on_tick = on_tick
        self.on_start = on_start
        self.instants = tuple(sorted({exact(t) for t in instants}))
        self.demoted = frozenset(demoted)
        self.exits = exits
        self.states = states
        self._kept_bodies = Kept()
        #: The frame the configuration describes, as ``(width, height)``:
        #: the unit of the record's home-frame boxes, whatever the camera
        #: does (:mod:`arachne.camera.projection`).
        self.home = configured_home(int(config.pixel_width), int(config.pixel_height))
        #: The configuration's pixel size, which a point cloud's thickness is
        #: scaled from as the scene draws.
        self.configured_pixels = (int(config.pixel_width), int(config.pixel_height))
        #: Each play's identity, counted as the plays are recorded.
        self._identities: list[PlayIdentity] = []
        self._line_counts: dict[SourceLine | None, int] = {}
        self.record: Record | None = None
        self._tagging = False
        #: Which execution of the scene this is, with the plays an earlier
        #: one found must be drawn as they run (:mod:`.execution`).
        self.execution = Execution()
        #: Why the camera's frames can't be drawn from the record, once found:
        #: from then on every play is drawn as it runs.
        self._camera_refusal: str | None = None
        #: The play :meth:`refusal` last answered for, and its answer.
        self._refusal_for: tuple[int, str | None] | None = None
        #: The earlier executions, as (number, play_keys), to compare with.
        self._earlier: list[tuple[int, tuple[bytes, ...]]] = []
        #: The frame-read instant whose hooks are running, if any (:mod:`.hooks`).
        self.frame: FrameInstant | None = None
        #: The play whose ``play_internal`` is running, until its body has run.
        self._body: _Body | None = None
        #: The files the script opens for reading while it runs.
        self._inputs: Inputs | None = None
        #: Whether a play is running: from when it is recorded until it has
        #: been evaluated.
        self._playing = False

    # The run

    def ensure_record(self) -> Record:
        """The scene's record, started when first needed."""
        if self.record is None:
            scene = self.scene
            scene_file = _scene_file(scene)
            self.record = Record(
                scene,
                grid=self.grid,
                scene_file=scene_file,
                # The source and inputs are known once the script has run
                # (:meth:`end_run`).
                identity=Identity(source_digest=None, seed=scene.random_seed),
            )
            self.record.execution = self.execution
            if self.exits == "all":
                self.record.exits = {}
            if self.states == "all":
                self.record.states = States(since=0, armed=Fraction(0))
            self.record.bridge = Bridge()
            self._camera_refusal = camera_refusal(scene.renderer.camera)
            self.record.refusal = self._camera_refusal
        return self.record

    def arm_motion(self) -> None:
        """Arm the motion rule: the record notes the first play to end from
        now on (``Record.armed_at``; the first arming counts), and, keeping
        armed exits (``exits="armed"``), keeps where each body stands as
        each such play ends. A record that keeps none is as it was.
        """
        if self.exits == "none":
            return
        record = self.ensure_record()
        if record.armed_at is None:
            # A play running now ends after the arming.
            record.armed_at = len(record.plays) - (1 if self._playing else 0)
        if self.exits == "armed" and record.exits is None:
            record.exits = {}

    def arm_states(self) -> None:
        """Keep, keeping armed states (``states="armed"``), the scene as the
        script calls each play from now on and as each such play ends, and
        as the script removes from it from now on (``Record.states``; the
        first arming counts). A record that keeps none is as it was.
        """
        if self.states != "armed":
            return
        record = self.ensure_record()
        if record.states is None:
            # A play running now was called before the arming: the first
            # play kept is the next one called.
            record.states = States(since=len(record.plays), armed=self.scene.time)

    def calling(self) -> None:
        """As the script calls a play, before the play adds or begins
        anything: keeping states, keep the scene as it stands.
        """
        if self.states == "all":
            self.ensure_record()
        states = self.record.states if self.record is not None else None
        if states is not None:
            assert self.record is not None
            states.calls[len(self.record.plays)] = self._state("call")

    @contextmanager
    def removing(self, kind: str, operands: tuple[Mobject, ...]) -> Iterator[None]:
        """Around a ``remove`` (or ``clear``) of the scene: keeping states,
        and the script calling it between plays, keep the scene as it was
        before it and its top level after (:class:`.record.Removal`).
        """
        if self.states == "all":
            self.ensure_record()
        record = self.record
        states = record.states if record is not None else None
        if states is None or self._playing:
            yield
            return
        assert record is not None
        before = self._state("removal")
        line = self.scene_line(sys._getframe(1))
        yield
        states.removals.append(
            Removal(
                before.instant,
                kind,
                operands,
                before,
                tuple(self.scene.mobjects),
                line,
                len(record.plays),
            )
        )

    def _state(self, kind: str) -> Capture:
        """The scene now, as a capture no frame draws, and what the scene's
        declarations name now (:meth:`Bridge.read_changes`).
        """
        camera = self.scene.renderer.camera
        record = self.ensure_record()
        record.bridge.read_changes(self.scene.time)
        return record._capture(
            kind,
            self.scene.time,
            camera_nodes(camera),
            CameraExtras.observe(camera, None),
            drawn=False,
        )

    def refusal(self) -> str | None:
        """Why the record can't draw the play about to run, so that it is
        drawn as it runs, if so: its camera, or a MultiCamera display its
        camera left, which shows the last picture drawn into it (and every
        play after it); a mobject it holds whose drawing runs the scene's
        code; or, from the earliest play an earlier execution found refusing
        mid-play on, that finding (:mod:`.execution`). The record keeps the first as its own
        (``Record.refusal``): it answers no instant.
        """
        record = self.ensure_record()
        index = len(record.plays)
        if self._refusal_for is not None and self._refusal_for[0] == index:
            return self._refusal_for[1]
        reason = self._camera()
        live_from = self.execution.live_from
        if reason is None and live_from is not None and index >= live_from:
            # From the earliest known refusal on, the record draws nothing:
            # the plays an execution runs may shift from the last one's.
            reason = self.execution.known[live_from]
        if reason is None:
            reason = self._draw_refusal()
        if record.refusal is None:
            record.refusal = reason
        self._refusal_for = (index, reason)
        return reason

    def _camera(self) -> str | None:
        """Why the camera's frames can't be drawn from the record, once
        found, for good.
        """
        if self._camera_refusal is None:
            # A display added since may draw through a camera the record can't.
            self._camera_refusal = camera_refusal(self.scene.renderer.camera)
        if self._camera_refusal is None:
            displays = vars(self.scene.renderer.camera).get(
                "image_mobjects_from_cameras"
            )
            record = self.record
            if (
                displays is not None
                and record is not None
                and record.display_left(displays) is not None
            ):
                self._camera_refusal = (
                    "camera.display_left: a zoomed display left its MultiCamera and"
                    " shows the last picture drawn into it, so the frames from then on"
                    " are drawn while the scene runs and the record answers no instant"
                )
        return self._camera_refusal

    def _draw_refusal(self) -> str | None:
        """:func:`.draw_refusal` of what the camera draws now and what the
        play about to run brings in (its animations' mobjects, Succession
        parts included).
        """
        from . import trusted

        scene = self.scene
        camera = scene.renderer.camera
        animated = [
            animation.mobject
            for animation in trusted.iter_animations(scene.animations or [])
            if animation.mobject is not None
        ]
        return draw_refusal(
            camera,
            [
                *scene.mobjects,
                *scene.foreground_mobjects,
                *animated,
                *camera_nodes(camera),
            ],
        )

    def reexecuting(
        self, known: Mapping[int, str], earlier: list[tuple[int, tuple[bytes, ...]]]
    ) -> None:
        """This run executes the scene again (:mod:`.execution`): the
        ``known`` plays are drawn as they run, and the run is compared with
        the ``earlier`` executions' plays (each ``(number, play_keys)``).
        """
        self.execution = Execution(number=len(earlier) + 1, known=dict(known))
        self._earlier = list(earlier)

    def begin_run(self) -> None:
        """The scene's script is about to run: tag the mobjects it makes, and
        watch the files it reads.
        """
        record = self.ensure_record()
        _RUNNING.append(self)
        if not ORIGINS.active:
            base = os.path.dirname(record.scene_file) if record.scene_file else None
            ORIGINS.start(os.path.realpath(base) if base else None, record.scene_file)
            self._tagging = True
        if record.scene_file is not None:
            self._inputs = Inputs(
                os.path.dirname(record.scene_file),
                excluded=engine_dirs(config),
                play=self._play_reading,
            )
            self._inputs.start()

    def _play_reading(self) -> int:
        """The index of the first play what the script reads now can change:
        the one running, or else the next to begin.
        """
        plays = len(self.ensure_record().plays)
        return plays - 1 if self._playing else plays

    def end_run(self) -> None:
        """The script has ended: capture what it left."""
        record = self.ensure_record()
        try:
            record.identity = self._identity(record)
            if record.refusal is None:
                # What the script added after its last play, or a camera
                # method it replaced since: the final frame, drawn from the
                # live scene.
                record.refusal = self._draw_refusal() or self._camera()
            if self.execution.number > 1:
                self.execution = record.execution = replace(
                    self.execution,
                    divergence=divergence(record, self._earlier),
                )
                if self.execution.divergence is not None:
                    logger.warning(
                        "Execution %(n)s of %(scene)s differs from an earlier one:"
                        " %(why)s",
                        {
                            "n": self.execution.number,
                            "scene": record.scene_class,
                            "why": self.execution.divergence,
                        },
                    )
            camera = self.scene.renderer.camera
            extras = CameraExtras.observe(camera, None)
            now = self.scene.time
            record.final = record._capture("final", now, camera_nodes(camera), extras)
            self.read_watches(now)
            record.duration = max(record.duration, now)
            if record.plays:
                record.final_names = record.plays[-1].names
            for play in record.plays:
                play.markers = record.markers_in(play.start, play.end)
        finally:
            self.abandon_run()

    def _identity(self, record: Record) -> Identity:
        """The execution's identity, now that its script has run: the source
        it ran (:func:`.source_files`), the files it read and its seed.
        """
        inputs = self._inputs
        if inputs is not None:
            inputs.stop()
        if record.scene_file is None:
            return record.identity
        sources = source_files(record.scene_file)
        try:
            digest = source_digest(sources, os.path.dirname(record.scene_file))
        except OSError:
            # A source file gone before the run ended: the source is unknown
            # (``None``, which compares with no execution), never a failed
            # render.
            digest = None
        return Identity(
            source_digest=digest,
            input_digests=()
            if inputs is None
            else inputs.input_digests(frozenset(sources)),
            seed=record.identity.seed,
        )

    def abandon_run(self) -> None:
        """The script stopped on an error: stop tagging and watching, and
        leave the record as it stands, without a final capture.
        """
        if self in _RUNNING:
            _RUNNING.remove(self)
        if self._inputs is not None:
            self._inputs.stop()
        if self._tagging:
            ORIGINS.stop()
            self._tagging = False

    # Plays

    def requested_end(self, start: Fraction | None = None) -> Fraction:
        """Where the play the scene has compiled, beginning at ``start`` (the
        scene clock unless given), ends unless a stop condition ends it first:
        its animations' longest run time, or what the scene's own
        ``get_run_time`` gave, as CE plays that span.
        """
        scene = self.scene
        if start is None:
            start = scene.time
        if scene_hooks(scene, ("get_run_time",)):
            return play_end(start, scene.duration)
        animations = scene.animations or []
        return play_end(
            start,
            max(animation._exact_run_time() for animation in animations),
        )

    def play(
        self,
        *,
        skipped: bool,
        cache_key: str | None,
        live: LiveFrames | None = None,
    ) -> PlayRecord:
        """Run the play the scene has compiled, and record it. ``live`` draws
        its frames while it runs, for a record that can't (``Record.refusal``).
        """
        from arachne.animation.animation import Wait

        scene = self.scene
        record = self.ensure_record()
        refusal = self.refusal()
        # What the captures find as mobjects enter them from here on.
        record.entry_refusal()
        animations = scene.animations or []
        start = scene.time
        requested_end = self.requested_end()
        frames = list(scene_frames(sys._getframe(1)))
        is_wait = len(animations) == 1 and isinstance(animations[0], Wait)
        play = PlayRecord(
            index=len(record.plays),
            start=start,
            end=requested_end,
            kind="wait" if is_wait else "play",
            route="",
            source_line=scene_location(frames[0], _base_dir(record))
            if frames
            else None,
            held=False,
            skipped=skipped,
            requested_end=requested_end,
            animations=tuple(str(animation) for animation in animations),
            names=names_from_frames(frames, scene),
            random_state=(random.getstate(), np.random.get_state()),  # noqa: NPY002
            cache_key=cache_key,
            _record=record,
            _animations=tuple(animations),
        )
        record.plays.append(play)
        if play.index == 0:
            self._identities.clear()
            self._line_counts.clear()
        self._identities.append(play_identity(play, self._line_counts))
        self._playing = True
        try:
            scene.begin_animations()
            play.moving = tuple(scene.moving_mobjects)
            play.static = tuple(scene.static_mobjects)
            play.held = bool(scene.is_current_animation_frozen_frame())
            play.frame_hooks = scene_hooks(scene, FRAME_HOOKS)
            play.play_internal_overridden = bool(scene_hooks(scene, ("play_internal",)))
            extras = CameraExtras.observe(scene.renderer.camera, None)
            body = _Body(play, refusal, live, extras)
            try:
                play.static_seq = self.capture("static", start, play, extras).seq
                if live is not None:
                    live.begin(play)
                if play.held:
                    # A frozen frame: CE calls none of the scene's hooks.
                    self._prepare(body)
                    self._run_play(body)
                else:
                    self._play_internal(body)
            finally:
                if body.plan is not None:
                    body.plan.release()
        finally:
            self._playing = False

        # The run that evaluated it, planned once play_internal's body began.
        final = body.run
        assert final is not None
        play.end = final.end
        record.duration = final.end
        if record.exits is not None:
            self._keep_exit(record, play)
        states = record.states
        if states is not None and play.index >= states.since:
            # Its animations finished and cleaned up, the script not yet run on.
            states.exits[play.index] = self._state("exit")
        if final.end < play.requested_end:
            self.mark(
                "wait_until",
                {"requested": play.requested_end - start, "actual": final.end - start},
                t=start,
                source_line=play.source_line,
            )
        return play

    def _keep_exit(self, record: Record, play: PlayRecord) -> None:
        """Keep where each body stands as ``play`` ends: its animations
        finished and cleaned up, the script not yet run on.
        """
        scene = self.scene
        shown = list(scene.mobjects)
        on = set(map(id, shown))
        shown += [m for m in scene.foreground_mobjects if id(m) not in on]
        assert record.exits is not None
        record.exits[play.index] = Exit(
            *bodies(
                shown,
                scene.renderer.camera,
                record.origin,
                home=self.home,
                configured_pixels=self.configured_pixels,
                kept=self._kept_bodies,
            )
        )

    def _prepare(self, body: _Body) -> PlayRun:
        """Plan the play over its span (``play.requested_end``): its events,
        its admission and route, and the run that evaluates it, kept in
        ``body`` and returned. A play is planned where its evaluation begins
        (``play_internal``'s body), from what the scene's code left by then:
        the run times, the span and every callback as the route will run them.
        """
        scene = self.scene
        play = body.play
        start, end = play.start, play.requested_end
        plan = body.plan = plan_events(scene.animations or [], start, end)
        record = self.ensure_record()
        admission = admit(
            scene,
            plan,
            held=play.held,
            base_dir=_base_dir(record),
            refusal=body.refusal,
            demoted=self._demoted(play),
        )
        play.admission = admission
        play.route = admission.route
        play.events = tuple(plan.instants(start, end + 1))
        run = body.run = PlayRun(
            self, play, ROUTE_CLASSES[admission.route](), body.live
        )
        run._extras = body.extras
        # Which of its ticks only step (those no frame reads) is planned
        # once the play has been evaluated at its start (``started``); a
        # brute-force run plans nothing (:mod:`.reference`).
        run.plans = (
            admission.route == "ticked"
            and not play.held
            and self.grid is not None
            and scene.stop_condition is None
            and self.stepping != BRUTE_FORCE
        )
        return run

    def _demoted(self, play: PlayRecord) -> bool:
        """Whether the probe demoted ``play``, or every play of its line."""
        if not self.demoted:
            return False
        identity = self._identities[play.index]
        return identity in self.demoted or whole_line(identity[0]) in self.demoted

    def _play_internal(self, body: _Body) -> None:
        """Call the scene's ``play_internal``, as CE's renderer does for a play
        that is not a frozen frame, and refuse one whose body did not run.
        """
        scene = self.scene
        overridden = scene_hooks(scene, TIME_PROGRESSION)
        if overridden:
            raise SceneHookRefused(
                "hook.time_progression",
                f"the scene's {', '.join(overridden)} chooses CE's frame instants,"
                " which the play's exact clock and the frame grid replace",
            )
        self._body = body
        try:
            scene.play_internal()
        finally:
            self._body = None
        if not body.ran:
            raise SceneHookRefused(
                "hook.play_internal.skipped",
                "the scene's play_internal returned without calling"
                " Scene.play_internal, whose body evaluates the play",
            )

    def play_body(self, skip_rendering: bool) -> None:
        """The body of ``Scene.play_internal``: evaluate the play, once."""
        body = self._body
        if body is None:
            raise RuntimeError("Scene.play_internal runs only inside a play")
        if body.ran:
            raise SceneHookRefused(
                "hook.play_internal.twice",
                "the scene's play_internal called Scene.play_internal twice, where"
                " a play is evaluated once",
            )
        if skip_rendering:
            raise SceneHookRefused(
                "hook.play_internal.skip_rendering",
                "the scene's play_internal asked to skip rendering: a play is"
                " evaluated whatever is drawn, and its frames are drawn after",
            )
        body.ran = True
        scene = self.scene
        assert scene.animations is not None
        # CE's second call: play_internal takes the play's length again, and
        # plays that one. The scene's code may have changed what the play is
        # since it was compiled (its own get_run_time, an animation's run
        # time), so the play is planned only now.
        scene.duration = scene.get_run_time(scene.animations)
        play = body.play
        play.requested_end = play.end = self.requested_end(play.start)
        self._prepare(body)
        self._run_play(body)

    def _run_play(self, body: _Body) -> None:
        """Evaluate the play through its route, then finish it, its end on the
        clock, as CE's clock reads it when a play finishes.
        """
        scene = self.scene
        run, plan, refusal = body.run, body.plan, body.refusal
        assert run is not None
        assert plan is not None
        play = run.play
        record = self.ensure_record()
        run.route.run(run, plan)
        if play.held:
            self._read_frozen(play, run.end)
        if run.live is not None:
            run.live.end(run.end)
        joined = record.entry_refusal()
        if refusal is None:
            # A mobject a callback added while the play ran, found as it
            # entered a capture or still there at the end; or a camera method
            # the scene's code replaced while it ran.
            joined = joined or self._draw_refusal() or self._camera()
            play.draw_refusal = joined
            if record.refusal is None:
                record.refusal = joined
        # The animations finish with CE's own event handling.
        plan.release()
        scene.renderer.time = run.end
        if not play.held:
            for animation in scene.animations or []:
                animation.finish()
                animation.clean_up_from_scene(scene)
            scene.update_mobjects(0)
        # The play's local time last evaluated: its end. A later
        # update_to_time(t) passes dt = t - last_t, as in CE.
        scene.last_t = float(run.end - play.start)

    def base_dt(self, name: str, dt: float) -> float:
        """The ``dt`` the body of the frame hook ``name`` runs with: the
        play's own at a frame-read instant (CE's frame ``dt`` is shown to the
        hooks only), else the one it is given.
        """
        frame = self.frame
        if frame is None:
            return dt
        frame.reach(name)
        return frame.dt

    def reads(self, instant: Fraction, until: Fraction) -> bool:
        """Whether a frame reads the state evaluated at ``instant`` that holds
        until ``until``, where the scene's frame hooks run: a frame of the
        grid, or without a grid every 60 Hz tick (a session's frames).
        """
        if self.grid is None:
            return instant < until and (instant * TICK_RATE).denominator == 1
        return self.keeps(instant, until)

    def keeps(self, instant: Fraction, until: Fraction) -> bool:
        """Whether the record keeps the state evaluated at ``instant`` that holds
        until ``until``: always without a grid, else when a grid instant reads it.
        """
        grid = self.grid
        if grid is None:
            return True
        return Fraction(math.ceil(instant * grid)) / grid < until

    def capture(
        self,
        kind: str,
        instant: Fraction,
        play: PlayRecord,
        extras: CameraExtras | None,
        until: Fraction | None = None,
    ) -> Any:
        """Capture the scene, reaching the play's draw list and the camera's
        own mobjects as well as the scene's.
        """
        record = self.ensure_record()
        roots: list[Mobject] = [*play.moving, *camera_nodes(self.scene.renderer.camera)]
        return record._capture(kind, instant, roots, extras, until)

    def location(self) -> SourceLine | None:
        """The line of the scene's code that called into the engine now."""
        return scene_location(sys._getframe(1), _base_dir(self.ensure_record()))

    # The assertion bridge

    def bridge_call(
        self,
        kind: str,
        caller: FrameType,
        *,
        reader: Callable[[], Mapping[str, Any] | None] | None = None,
        changes_only: bool = False,
        **fields: Any,
    ) -> BridgeCall:
        """Keep a call of the assertion bridge made now, from the frame
        ``caller``: its time, its play and its call site with ``fields``;
        with ``reader``, a call read at every kept capture from now on
        (with ``changes_only``, kept only where its reading changed).
        """
        record = self.ensure_record()
        call = BridgeCall(
            kind,
            self.scene.time,
            self._play_reading(),
            self.scene_line(caller),
            **fields,
        )
        return record.bridge.add(call, reader, changes_only=changes_only)

    def scene_line(self, frame: FrameType | None) -> SourceLine | None:
        """The line of the innermost frame from ``frame`` outwards that is
        the scene file's own (the line the X5 report gives a finding), else
        the innermost of the scene's code.
        """
        record = self.ensure_record()
        base = _base_dir(record)
        if record.scene_file is not None:
            line = scene_file_line(frame, record.scene_file)
            if line is not None:
                return SourceLine(relative_path(record.scene_file, base), line)
        return scene_location(frame, base)

    def read_watches(self, instant: Fraction, until: Fraction | None = None) -> None:
        """Read the scene's watches at ``instant``, a kept capture whose state
        holds until ``until``.
        """
        record = self.ensure_record()
        if record.bridge.watching:
            record.bridge.read(instant, self.scene_line(sys._getframe(1)), until)

    def _read_frozen(self, play: PlayRecord, end: Fraction) -> None:
        """Read the scene's watches at each frame of the frozen play
        ``play`` up to ``end``, with the clock at that frame's time, as a
        check run on every frame of a wait reads them: the play keeps one
        capture, and draws it on every frame.
        """
        record = self.ensure_record()
        if not record.bridge.watching or self.grid is None:
            return
        renderer = self.scene.renderer
        now = renderer.clock
        step = 1 / self.grid
        try:
            for t in frames_in(play.start, end, self.grid):
                renderer.time = t
                self.read_watches(t, t + step)
        finally:
            renderer.time = now

    # Markers

    def mark(
        self,
        kind: str,
        data: Mapping[str, Any],
        *,
        t: Fraction | None = None,
        source_line: SourceLine | None = None,
    ) -> Marker:
        """Place a marker on the timeline, at the scene clock unless ``t``."""
        record = self.ensure_record()
        if source_line is None:
            source_line = scene_location(sys._getframe(1), _base_dir(record))
        marker = Marker(
            kind,
            self.scene.time if t is None else exact(t),
            dict(data),
            source_line,
        )
        record.markers.append(marker)
        return marker


class _Body:
    """A play being run: why the record can't draw it if it can't, its live
    drawing and the camera's inputs at its start, its run and event plan once
    planned, and whether ``play_internal``'s body ran.
    """

    def __init__(
        self,
        play: PlayRecord,
        refusal: str | None,
        live: LiveFrames | None,
        extras: CameraExtras | None,
    ) -> None:
        self.play = play
        self.refusal = refusal
        self.live = live
        self.extras = extras
        self.run: PlayRun | None = None
        self.plan: EventPlan | None = None
        self.ran = False


def _scene_file(scene: Scene) -> str | None:
    try:
        found = inspect.getsourcefile(type(scene))
    except (TypeError, OSError):
        return None
    return os.path.realpath(found) if found and os.path.isfile(found) else None


def _base_dir(record: Record) -> str | None:
    return os.path.dirname(record.scene_file) if record.scene_file else None
