"""A scene opened in process: executed once, recorded, then asked about.

:meth:`Session.open` imports a scene file, executes one of its scenes
forward with nothing written, and keeps the scene's :class:`Record` with
every instant the run evaluated. :meth:`Session.state_at` answers from the
record, and :meth:`Session.frame` draws one frame from it; neither runs the
scene's code again. Before either, the probe checks every state-function
play against what its record kept (:mod:`.timeline.probe`), and the scene is
recorded again with any play it demotes ticked. :meth:`Session.compare`
compares the session's execution with another of the same source, perhaps
from another process, and keeps where they differ in
:attr:`Session.divergences`.
"""

from __future__ import annotations

__all__ = ["Session"]

import contextlib
import importlib.util
import itertools
import os
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._config import config as global_config
from ._config import tempconfig
from .constants import QUALITIES
from .timeline import admission
from .timeline.admission import PlayIdentity, play_identities
from .timeline.clock import exact
from .timeline.draw import Drawing
from .timeline.execution import (
    Divergence,
    ExecutionSummary,
    Incomparable,
    changed_inputs,
    compare,
    summarize,
)
from .timeline.inputs import source_digest, source_files
from .timeline.origins import ORIGINS
from .timeline.probe import Probe
from .timeline.probe import probe as probe_record
from .timeline.recorder import Recorder, _scene_file
from .timeline.routes import draw
from .timeline.source import SourceLine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from PIL import Image

    from .scene.scene import Scene
    from .timeline.record import Record
    from .timeline.state import State
    from .typing import PixelArray, StrPath

_MODULE_NUMBERS = itertools.count()


class Session:
    """One scene of one file, executed and recorded.

    Open it with :meth:`open`; it holds the executed scene, whose camera
    draws its frames.
    """

    def __init__(self, scene: Scene, record: Record, module: ModuleType) -> None:
        self.scene = scene
        self.record = record
        self.module = module
        self._quality: str | None = None
        self._static: tuple[int, str | None, PixelArray | None] | None = None
        camera = scene.renderer.camera
        self._executed_shape = (camera.pixel_height, camera.pixel_width)
        #: Where the session's execution was found to differ from another of
        #: its source, in the order found (``replay.diverged``).
        self.divergences: list[Divergence] = []
        #: The record last summarised, and its summary.
        self._summary: tuple[Record, ExecutionSummary] | None = None
        #: What the probe found for the state-function plays, when the
        #: session was opened with it (:mod:`.timeline.probe`).
        self.probe: Probe | None = None

    @classmethod
    def open(
        cls,
        file: StrPath,
        scene: str,
        *,
        grid: Fraction | float | None | Literal["render"] = "render",
        config: dict[str, Any] | None = None,
        stepping: bool | Literal["brute_force"] = True,
        on_step: Callable[[Scene, Fraction], object] | None = None,
        on_tick: Callable[[Scene, Fraction, bool], object] | None = None,
        on_start: Callable[[Scene, Fraction], object] | None = None,
        instants: Iterable[Fraction | float] = (),
        probe: bool = True,
    ) -> Session:
        """Execute the scene class named ``scene`` from ``file`` and record it.

        ``grid`` is the frame rate whose instants the record keeps: by
        default the frame rate of the render the configuration describes, so
        the session's answers equal that render's frames (a ticked play's
        dt-free callbacks run at the instants a frame reads,
        :mod:`.timeline.ticking`); ``None`` keeps and fully evaluates every
        instant the run reaches. ``stepping`` is ``Recorder``'s switch:
        ``False`` runs every tick of a ticked play in full, and
        ``"brute_force"`` is the verification mode of
        :mod:`.timeline.reference`; ``on_step``,
        ``on_tick`` and ``on_start`` are ``Recorder``'s diagnostic hooks.
        ``instants`` are extra instants the trusted and state-function plays
        are evaluated and recorded at, between frames (``Recorder``'s): the
        forward evaluation a restore there is compared with. ``config``
        overrides the configuration the scene is executed under; nothing is
        written either way.

        ``probe`` probes every state-function play once the scene has run
        (:mod:`.timeline.probe`, :attr:`probe`). A play it demotes ticks in
        every session of the source from then on: the scene is recorded
        again with it on route 3, as a new execution compared with the
        first, and the probe's ``replay.diverged`` is kept in
        :attr:`divergences`, with any the comparison finds. The scene is
        recorded again until the probe demotes nothing more, unless an
        execution plays other plays than the one before it (another count,
        or another kind or line at an index): that is kept as a divergence
        of kind ``execution``, the session holds that execution, with every
        demotion found applied, and :attr:`.Probe.settled` is false. Past
        one more execution than the first had plays, it stops the same way.
        """
        path = Path(file).resolve()
        settings = {
            **(config or {}),
            "dry_run": True,
            "disable_caching": True,
        }
        options = {
            "grid": grid,
            "stepping": stepping,
            "on_step": on_step,
            "on_tick": on_tick,
            "on_start": on_start,
            "instants": tuple(instants),
        }
        demoted: set[PlayIdentity] = set()
        session = cls(*_execute(path, scene, settings, options, demoted))
        if not probe:
            return session
        # Probe the record. On a demotion, record the scene again with the
        # play ticked, a new execution compared with the first, and probe
        # that: each pass ticks at least one play more, while the executions
        # play the same plays of the same source. When they don't, the
        # session holds the last execution, its plays where the probe found
        # them (a demotion follows its play's identity, never its index).
        latest = found = probe_record(session.record)
        first: ExecutionSummary | None = None
        # A backstop: with the same plays in every execution, each pass ticks
        # one more of them, so this many executions always suffice.
        cap = len(session.record.plays) + 1
        executions = 1
        settled = True
        while True:
            for divergence in latest.divergences:
                session._keep(divergence)
            record = session.record
            identities = play_identities(record.plays)
            new = {identities[index] for index in latest.demoted} - demoted
            if not new:
                break
            digest = record.identity.source_digest
            for identity in new:
                demoted.add(identity)
                if digest is not None:
                    admission.demote(digest, record.scene_class, identity)
            if executions >= cap:
                settled = False
                break
            if first is None:
                first = session.summary()
            session._executed_again(*_execute(path, scene, settings, options, demoted))
            executions += 1
            with contextlib.suppress(Incomparable):
                session.compare(first)
            if not _same_source(record, session.record) or _schedule_shift(
                record, session.record
            ):
                # Another source is stale, never a divergence; the same one
                # playing other plays is where its executions differ.
                changed = _schedule_change(record, session.record)
                if changed is not None:
                    session._keep(changed)
                found = session._found_in(found, record)
                settled = False
                break
            latest = probe_record(session.record)
            found = found.then(latest)
        session.probe = replace(found, settled=settled)
        return session

    def state_at(self, t: Fraction | float) -> State:
        """The scene at ``t``, from the record: restored there when a
        state-function play holds ``t`` and the record did not sample it.
        """
        return self.record.state_at(t)

    def frame(self, t: Fraction | float, quality: str | None = None) -> Image.Image:
        """The frame at ``t``, drawn from the record as a render draws it;
        at ``quality`` (a key of :data:`~arachne.constants.QUALITIES`) when
        given, else at the pixel size the scene was executed with.

        A state-function play is restored at an instant it was not sampled
        at; raises :class:`~arachne.timeline.restore.Unrestorable` where it
        can't be, and :class:`~arachne.timeline.routes.RecordRefused` for a
        scene whose camera the record can't draw.
        """
        self._use_quality(quality)
        record = self.record
        with Drawing(record) as drawing:
            pixels = draw(
                drawing,
                record,
                exact(t),
                lambda play: self._static_picture(drawing, play),
            )
        return self.scene.renderer.camera.get_image(pixels)

    def summary(self) -> ExecutionSummary:
        """The summary of the session's execution, which another execution
        of its source, in this process or another, is compared with
        (:meth:`compare`, :meth:`.ExecutionSummary.to_json`).
        """
        if self._summary is None or self._summary[0] is not self.record:
            self._summary = (self.record, summarize(self.record))
        return self._summary[1]

    def compare(self, summary: ExecutionSummary) -> Divergence | None:
        """Compare the session's execution with ``summary``'s, an execution of
        the same source (:func:`.execution.compare`): the first play where they
        differ, also kept in :attr:`divergences` unless a divergence of that
        cause is there already, or ``None``. Raises
        :class:`.execution.Incomparable` for another source, or an unknown one.
        """
        found = compare(self.summary(), summary)
        if found is not None:
            self._keep(found)
        return found

    def _keep(self, divergence: Divergence) -> None:
        """Keep ``divergence`` in :attr:`divergences`, unless one of the same
        cause is there already (:attr:`.Divergence.cause`).
        """
        if all(kept.cause != divergence.cause for kept in self.divergences):
            self.divergences.append(divergence)

    def _found_in(self, found: Probe, before: Record) -> Probe:
        """``found``, the probe of ``before``, moved to the record held, an
        execution recorded after it: each play at its index there by its
        identity, a play it does not play left out and its divergence naming
        no play; the same for the probe's divergences kept.
        """
        places = {
            identity: index
            for index, identity in enumerate(play_identities(self.record.plays))
        }
        moved = {
            index: places.get(identity)
            for index, identity in enumerate(play_identities(before.plays))
        }
        self.divergences = [
            replace(d, play=moved.get(d.play))
            if d.kind == "probe" and d.play is not None
            else d
            for d in self.divergences
        ]
        return found.moved(moved)

    def _executed_again(self, scene: Scene, record: Record, module: ModuleType) -> None:
        """Hold the scene executed again, ``record`` its record, in place of
        the one held.
        """
        sys.modules.pop(self.module.__name__, None)
        self.scene, self.record, self.module = scene, record, module
        camera = scene.renderer.camera
        self._executed_shape = (camera.pixel_height, camera.pixel_width)
        self._quality = None
        self._static = None

    def _static_picture(self, drawing: Drawing, play: Any) -> PixelArray | None:
        """The play's static picture, kept for the next frame of the same play."""
        key = (play.index, self._quality)
        if self._static is not None and self._static[:2] == key:
            return self._static[2]
        static = drawing.static(play)
        self._static = (*key, static)
        return static

    def _use_quality(self, quality: str | None) -> None:
        if quality == self._quality:
            return
        camera = self.scene.renderer.camera
        if quality is None:
            height, width = self._executed_shape
        else:
            if quality not in QUALITIES:
                raise ValueError(
                    f"Unknown quality {quality!r}: one of {', '.join(QUALITIES)}."
                )
            height = QUALITIES[quality]["pixel_height"]
            width = QUALITIES[quality]["pixel_width"]
        # The frame keeps its size in scene units, as a render at that
        # quality has it; only the pixels change.
        camera.pixel_height, camera.pixel_width = height, width
        camera.init_background()
        camera.pixel_array_to_cairo_context.clear()
        camera.reset()
        self._quality = quality
        self._static = None

    def close(self) -> None:
        """Let go of the scene and its record."""
        sys.modules.pop(self.module.__name__, None)


def _execute(
    path: Path,
    scene: str,
    settings: dict[str, Any],
    options: dict[str, Any],
    demoted: set[PlayIdentity],
) -> tuple[Scene, Record, ModuleType]:
    """Import ``path`` afresh and execute its scene ``scene`` under
    ``settings``, recorded by a ``Recorder`` given ``options`` that ticks the
    ``demoted`` plays and those the probe demoted in an earlier session of
    the source (:func:`.admission.demoted`), which join ``demoted``.
    """
    base_dir = str(path.parent)
    with tempconfig(settings):
        ORIGINS.start(base_dir)
        added_path = base_dir not in sys.path
        if added_path:
            sys.path.insert(0, base_dir)
        try:
            module = _import(path)
            try:
                scene_class = getattr(module, scene, None)
                if not isinstance(scene_class, type):
                    raise LookupError(f"{path.name} defines no scene named {scene!r}.")
                instance: Scene = scene_class()
                demoted |= admission.demoted(
                    _source_digest(instance), type(instance).__name__
                )
                grid = options["grid"]
                instance.timeline = Recorder(
                    instance,
                    **{
                        **options,
                        "grid": global_config.frame_rate if grid == "render" else grid,
                    },
                    draws=False,
                    demoted=demoted,
                )
                instance.render()
            except BaseException:
                sys.modules.pop(module.__name__, None)
                raise
        finally:
            ORIGINS.stop()
            if added_path and base_dir in sys.path:
                sys.path.remove(base_dir)
    record = instance.record
    assert record is not None
    return instance, record, module


def _same_source(before: Record, after: Record) -> bool:
    """Whether two executions ran one known source."""
    digest = before.identity.source_digest
    return digest is not None and digest == after.identity.source_digest


def _schedule_shift(
    before: Record, after: Record
) -> tuple[int, SourceLine | None, str] | None:
    """Where ``after``, the execution recorded after ``before``, plays other
    plays (another kind or line at an index, or another count): the index,
    the line there and how.
    """
    for index, (a, b) in enumerate(zip(before.plays, after.plays, strict=False)):
        if (a.kind, a.source_line) != (b.kind, b.source_line):
            return (
                index,
                b.source_line,
                f"its play {index} is a {b.kind} at {b.source_line}, where the"
                f" execution before played a {a.kind} at {a.source_line}",
            )
    if len(before.plays) != len(after.plays):
        index = min(len(before.plays), len(after.plays))
        longer = after if len(after.plays) > index else before
        return (
            index,
            longer.plays[index].source_line,
            f"this execution ran {len(after.plays)} plays, and the one before"
            f" {len(before.plays)}",
        )
    return None


def _schedule_change(before: Record, after: Record) -> Divergence | None:
    """Where ``after``, an execution of ``before``'s source recorded after
    it, plays other plays, as a divergence; ``None`` when they play the same,
    and for two sources or an unknown one, which are stale, never diverged.
    """
    if not _same_source(before, after):
        return None
    shift = _schedule_shift(before, after)
    if shift is None:
        return None
    index, line, reason = shift
    return Divergence(
        "execution",
        index,
        line,
        reason,
        changed_inputs(before.identity, after.identity),
    )


def _source_digest(scene: Scene) -> str | None:
    """The digest of the source ``scene``'s class heads, as its record's
    identity will have it (:meth:`.Recorder.end_run`); ``None`` when it
    can't be read.
    """
    scene_file = _scene_file(scene)
    if scene_file is None:
        return None
    try:
        return source_digest(source_files(scene_file), os.path.dirname(scene_file))
    except OSError:
        return None


def _import(path: Path) -> ModuleType:
    """Import ``path`` under a name of its own, so two sessions of one file
    never share a module.
    """
    name = f"arachne_session_{next(_MODULE_NUMBERS)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} is not a Python file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
