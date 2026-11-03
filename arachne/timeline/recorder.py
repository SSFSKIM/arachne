"""The forward run of a scene, recorded as it goes.

A scene owns one :class:`Recorder`. Each ``play`` (and ``wait``) runs
through :meth:`Recorder.play`: it begins the animations as CE does, lets the
play's route evaluate it, finishes it, and leaves a :class:`PlayRecord` in
the scene's :class:`Record` with every capture the record keeps.
"""

from __future__ import annotations

__all__ = ["PlayRun", "Recorder"]

import hashlib
import inspect
import math
import os
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from arachne._config import logger

from .admission import admit, camera_refusal
from .clock import TICK_RATE, exact, play_end
from .dispatch import draw_refusal
from .draw import CameraExtras, LiveFrames, camera_nodes
from .events import plan_events
from .execution import Execution, divergence
from .origins import ORIGINS
from .record import Identity, Marker, PlayRecord, Record
from .reference import BRUTE_FORCE
from .routes import ROUTE_CLASSES, Route, snapshot_animations
from .source import SourceLine, names_from_frames, scene_frames, scene_location
from .ticking import (
    TickPlan,
    current_structure,
    plan_ticks,
)

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene


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

    def set_clock(self, instant: Fraction) -> None:
        """Make the scene clock read ``instant``, drawing first whatever frames
        a live drawing still owes the instant before it.
        """
        if self.live is not None:
            self.live.flush(instant)
        self.scene.renderer.time = instant

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
        if play.held:
            return
        # The animations' state where a play is evaluated from: every kept
        # instant of a trusted play; a state-function play's start and events,
        # which the planned restore evaluates from.
        if play.route == "trusted" or (
            play.route == "state" and (instant == play.start or instant in play.events)
        ):
            play._animation_states[capture.seq] = snapshot_animations(play._animations)

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
            plan = plan_ticks(
                self.scene,
                list(play._animations),
                _base_dir(self.recorder.ensure_record()),
            )
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
        plan = plan_ticks(
            self.scene, list(play._animations), _base_dir(recorder.ensure_record())
        )
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
    ) -> None:
        if stepping not in (True, False, BRUTE_FORCE):
            raise ValueError(
                f"stepping is True, False or {BRUTE_FORCE!r}, not {stepping!r}"
            )
        self.scene = scene
        self.grid = exact(grid) if grid is not None else None
        self.draws = draws
        self.stepping = stepping
        self.on_step = on_step
        self.on_tick = on_tick
        self.on_start = on_start
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
                identity=Identity(
                    source_digest=_digest(scene_file),
                    seed=scene.random_seed,
                ),
            )
            self.record.execution = self.execution
            self._camera_refusal = camera_refusal(scene.renderer.camera)
            self.record.refusal = self._camera_refusal
        return self.record

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
        """The scene's script is about to run: tag the mobjects it makes."""
        record = self.ensure_record()
        if not ORIGINS.active:
            base = os.path.dirname(record.scene_file) if record.scene_file else None
            ORIGINS.start(os.path.realpath(base) if base else None)
            self._tagging = True

    def end_run(self) -> None:
        """The script has ended: capture what it left."""
        record = self.ensure_record()
        try:
            if record.refusal is None:
                # What the script added after its last play: the final frame,
                # drawn from the live scene.
                record.refusal = self._draw_refusal()
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
            record.duration = max(record.duration, now)
            if record.plays:
                record.final_names = record.plays[-1].names
            for play in record.plays:
                play.markers = record.markers_in(play.start, play.end)
        finally:
            self.abandon_run()

    def abandon_run(self) -> None:
        """The script stopped on an error: stop tagging, and leave the record
        as it stands, without a final capture.
        """
        if self._tagging:
            ORIGINS.stop()
            self._tagging = False

    # Plays

    def requested_end(self) -> Fraction:
        """Where the play the scene has compiled ends unless a stop condition
        ends it first.
        """
        animations = self.scene.animations or []
        return play_end(
            self.scene.time,
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

        scene.begin_animations()
        play.moving = tuple(scene.moving_mobjects)
        play.static = tuple(scene.static_mobjects)
        play.held = bool(scene.is_current_animation_frozen_frame())
        plan = plan_events(animations, start, requested_end)
        try:
            admission = admit(
                scene,
                plan,
                held=play.held,
                base_dir=_base_dir(record),
                refusal=refusal,
            )
            play.admission = admission
            play.route = admission.route
            play.events = tuple(plan.instants(start, requested_end + 1))
            route = ROUTE_CLASSES[admission.route]()
            camera = scene.renderer.camera
            run = PlayRun(self, play, route, live)
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
            run._extras = CameraExtras.observe(camera, None)
            play.static_seq = self.capture("static", start, play, run._extras).seq
            if live is not None:
                live.begin(play)
            route.run(run, plan)
            if live is not None:
                live.end(run.end)
            joined = record.entry_refusal()
            if refusal is None:
                # A mobject a callback added while the play ran, found as it
                # entered a capture or still there at the end.
                joined = joined or self._draw_refusal()
                play.draw_refusal = joined
                if record.refusal is None:
                    record.refusal = joined
        finally:
            plan.release()

        if not play.held:
            for animation in animations:
                animation.finish()
                animation.clean_up_from_scene(scene)
            scene.update_mobjects(0)
        play.end = run.end
        scene.renderer.time = run.end
        # The play's local time last evaluated: its end. A later
        # update_to_time(t) passes dt = t - last_t, as in CE.
        scene.last_t = float(run.end - start)
        record.duration = run.end
        if run.end < play.requested_end:
            self.mark(
                "wait_until",
                {"requested": play.requested_end - start, "actual": run.end - start},
                t=start,
                source_line=play.source_line,
            )
        return play

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


def _scene_file(scene: Scene) -> str | None:
    try:
        found = inspect.getsourcefile(type(scene))
    except (TypeError, OSError):
        return None
    return os.path.realpath(found) if found and os.path.isfile(found) else None


def _base_dir(record: Record) -> str | None:
    return os.path.dirname(record.scene_file) if record.scene_file else None


def _digest(path: str | None) -> str | None:
    if path is None:
        return None
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()
