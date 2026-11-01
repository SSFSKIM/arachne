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
from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from .clock import exact
from .draw import CameraExtras, camera_nodes
from .origins import ORIGINS
from .record import Identity, Marker, PlayRecord, Record
from .routes import Route, choose_route
from .source import SourceLine, names_from_frames, scene_frames, scene_location

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.scene.scene import Scene


class PlayRun:
    """One play while its route evaluates it."""

    def __init__(self, recorder: Recorder, play: PlayRecord, route: Route) -> None:
        self.recorder = recorder
        self.scene = recorder.scene
        self.play = play
        self.route = route
        self.start = play.start
        self.requested_end = play.requested_end
        self.held = play.held
        #: Where the play ended; the route sets it.
        self.end = play.requested_end
        self._extras: CameraExtras | None = None

    def set_clock(self, instant: Fraction) -> None:
        """Make the scene clock read ``instant``."""
        self.scene.renderer.time = instant

    def sample(self, instant: Fraction, *, until: Fraction) -> None:
        """The play has been evaluated at ``instant``, and holds until its next
        evaluated instant ``until``: capture it if the record keeps it.

        The camera's executable inputs are evaluated at every instant, kept
        or not, so a stateful one is called as often whatever the record keeps.
        """
        recorder = self.recorder
        self._extras = CameraExtras.observe(self.scene.renderer.camera, self._extras)
        if not recorder.keeps(instant, until):
            return
        capture = recorder.capture("sample", instant, self.play, self._extras, until)
        self.play.samples.append(capture)
        self.play._instants.append(instant)

    def progress(self, instants: list[Fraction]) -> Any:
        """A progress bar over ``instants``."""
        return self.scene._play_progress(instants)


class Recorder:
    """Records a scene's forward run.

    ``grid`` is the frame rate whose instants the record keeps (a render's),
    or ``None`` to keep every evaluated instant (a session's, which answers
    any time). ``draws`` says whether the renderer draws each play's frames
    from the record as soon as the play has run.
    """

    def __init__(
        self, scene: Scene, *, grid: Fraction | float | None, draws: bool = True
    ) -> None:
        self.scene = scene
        self.grid = exact(grid) if grid is not None else None
        self.draws = draws
        self.record: Record | None = None
        self._tagging = False

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
        return self.record

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

    def play(self, *, skipped: bool, cache_key: str | None) -> PlayRecord:
        """Run the play the scene has compiled, and record it."""
        from arachne.animation.animation import Wait

        scene = self.scene
        record = self.ensure_record()
        animations = scene.animations or []
        start = scene.time
        requested = exact(scene.duration)
        frames = list(scene_frames(sys._getframe(1)))
        is_wait = len(animations) == 1 and isinstance(animations[0], Wait)
        play = PlayRecord(
            index=len(record.plays),
            start=start,
            end=start + requested,
            kind="wait" if is_wait else "play",
            route="",
            source_line=scene_location(frames[0], _base_dir(record))
            if frames
            else None,
            held=False,
            skipped=skipped,
            requested_end=start + requested,
            animations=tuple(str(animation) for animation in animations),
            names=names_from_frames(frames, scene),
            random_state=(random.getstate(), np.random.get_state()),  # noqa: NPY002
            cache_key=cache_key,
            _record=record,
        )
        record.plays.append(play)

        scene.begin_animations()
        play.moving = tuple(scene.moving_mobjects)
        play.static = tuple(scene.static_mobjects)
        play.held = bool(scene.is_current_animation_frozen_frame())
        route = choose_route(scene)
        play.route = route.name
        camera = scene.renderer.camera
        run = PlayRun(self, play, route)
        run._extras = CameraExtras.observe(camera, None)
        play.static_seq = self.capture("static", start, play, run._extras).seq

        route.run(run)

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
