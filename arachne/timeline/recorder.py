"""The forward run of a scene, recorded as it goes.

A scene owns one :class:`Recorder`. Each ``play`` (and ``wait``) runs
through :meth:`Recorder.play`: it begins the animations as CE does, then
calls the scene's ``play_internal``, whose body (:meth:`Recorder.play_body`)
lets the play's route evaluate it and finishes it (:mod:`.hooks`), and
leaves a :class:`PlayRecord` in the scene's :class:`Record` with every
capture the record keeps.
"""

from __future__ import annotations

__all__ = ["PlayRun", "Recorder"]

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

from arachne._config import config, logger

from .admission import admit, camera_refusal
from .clock import TICK_RATE, exact, play_end
from .dispatch import draw_refusal, scene_hooks
from .draw import CameraExtras, LiveFrames, camera_nodes
from .events import EventPlan, plan_events
from .execution import Execution, divergence
from .hooks import FRAME_HOOKS, TIME_PROGRESSION, FrameInstant, SceneHookRefused
from .inputs import Inputs, engine_dirs, source_digest, source_files
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
        #: The play's last frame-read instant, once it has one.
        self._last_frame: Fraction | None = None

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
        #: The frame-read instant whose hooks are running, if any (:mod:`.hooks`).
        self.frame: FrameInstant | None = None
        #: The play whose ``play_internal`` is running, until its body has run.
        self._body: _Body | None = None
        #: The files the script opens for reading while it runs.
        self._inputs: Inputs | None = None

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
        """The scene's script is about to run: tag the mobjects it makes, and
        watch the files it reads.
        """
        record = self.ensure_record()
        if not ORIGINS.active:
            base = os.path.dirname(record.scene_file) if record.scene_file else None
            ORIGINS.start(os.path.realpath(base) if base else None)
            self._tagging = True
        if record.scene_file is not None:
            self._inputs = Inputs(
                os.path.dirname(record.scene_file), excluded=engine_dirs(config)
            )
            self._inputs.start()

    def end_run(self) -> None:
        """The script has ended: capture what it left."""
        record = self.ensure_record()
        try:
            record.identity = self._identity(record)
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

    def _identity(self, record: Record) -> Identity:
        """The execution's identity, now that its script has run: the source
        it ran (:func:`.source_files`), the files it read and its seed.
        """
        inputs = self._inputs
        if inputs is not None:
            inputs.stop()
        if record.scene_file is None:
            return record.identity
        module = sys.modules.get(type(self.scene).__module__)
        sources = source_files(record.scene_file, getattr(module, "__package__", None))
        return Identity(
            source_digest=source_digest(sources, os.path.dirname(record.scene_file)),
            input_digests=()
            if inputs is None
            else inputs.input_digests(frozenset(sources)),
            seed=record.identity.seed,
        )

    def abandon_run(self) -> None:
        """The script stopped on an error: stop tagging and watching, and
        leave the record as it stands, without a final capture.
        """
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

        # The run that evaluated it, planned once play_internal's body began.
        final = body.run
        assert final is not None
        play.end = final.end
        record.duration = final.end
        if final.end < play.requested_end:
            self.mark(
                "wait_until",
                {"requested": play.requested_end - start, "actual": final.end - start},
                t=start,
                source_line=play.source_line,
            )
        return play

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
        if run.live is not None:
            run.live.end(run.end)
        joined = record.entry_refusal()
        if refusal is None:
            # A mobject a callback added while the play ran, found as it
            # entered a capture or still there at the end.
            joined = joined or self._draw_refusal()
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
