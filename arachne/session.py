"""A scene opened in process: executed once, recorded, then asked about.

:meth:`Session.open` imports a scene file, executes one of its scenes
forward with nothing written, and keeps the scene's :class:`Record` with
every instant the run evaluated. :meth:`Session.state_at` answers from the
record, and :meth:`Session.frame` draws one frame from it; neither runs the
scene's code again. Before either, the probe checks every state-function
play against what its record kept (:mod:`.timeline.probe`), and the scene is
recorded again with any play it demotes ticked. :meth:`Session.answers`
answers a batch of instants, and what the record can't answer there a
forward run does: the scene executed once more, for the whole batch
(:mod:`.timeline.fallback`). :meth:`Session.compare` compares the session's
execution with another of the same source, perhaps from another process,
and keeps where they differ in :attr:`Session.divergences`.

A session's life: :meth:`Session.changed` says whether the files its
execution came from (the scene's source, and the files it read) hold
anything else now, by their content; :meth:`Session.reload` executes the
scene again from them, as :meth:`Session.open` does, as the session's next
:attr:`Session.generation`; a scene that raises raises its own error, as
it was raised, its explanation on it (``scene_error``, a
:class:`~arachne.errors.SceneError`); :meth:`Session.close` lets go of the
scene, its record and its module.
"""

from __future__ import annotations

__all__ = ["Session"]

import contextlib
import functools
import importlib.util
import itertools
import os
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from ._config import config as global_config
from ._config import tempconfig
from .constants import QUALITIES
from .errors import explain_on
from .mobject.construction_caches import forget_construction_caches
from .timeline import admission
from .timeline.admission import (
    PlayIdentity,
    line_plays,
    play_identities,
    whole_line,
)
from .timeline.clock import exact
from .timeline.draw import Drawing, DrawReuse
from .timeline.execution import (
    Divergence,
    ExecutionSummary,
    Incomparable,
    changed_inputs,
    compare,
    summarize,
)
from .timeline.fallback import Answer, ForwardRun, answer_instants
from .timeline.inputs import Read, file_digests, source_digest, source_files
from .timeline.loader import FromSource, imports
from .timeline.mapping import freeze_mapping_centre
from .timeline.origins import ORIGINS
from .timeline.probe import Probe
from .timeline.probe import probe as probe_record
from .timeline.recorder import Recorder, _scene_file
from .timeline.routes import _draw, _resolve, _Resolved
from .timeline.source import SourceLine

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from PIL import Image

    from ._config.utils import ManimConfig
    from .lint import LintResult
    from .lint.keyframes import KeyFrames, Storyboard
    from .lint.query import Query
    from .scene.scene import Scene
    from .timeline.record import PlayRecord, Record
    from .timeline.state import State
    from .typing import PixelArray, StrPath

_MODULE_NUMBERS = itertools.count()
_T = TypeVar("_T")
#: How many drawn frames a session keeps for reuse.
SESSION_DRAWS = 4


def _under_its_settings(method: Callable[..., _T]) -> Callable[..., _T]:
    """Run a public :class:`Session` operation under the configuration the
    session was opened with (:attr:`Source.settings`): whatever it reads of
    the configuration as it evaluates or draws (a callback building a graph
    across the frame, a camera drawing a point cloud) is the session's,
    whatever the process holds then. Entered once: an operation another
    calls runs inside the first's.
    """

    @functools.wraps(method)
    def run(self: Session, *args: Any, **kwargs: Any) -> _T:
        source = self._source
        if source is None or self._settled:
            # Not opened yet (``open`` runs under its settings itself), or
            # inside an operation that entered them.
            return method(self, *args, **kwargs)
        self._settled = True
        try:
            with tempconfig(source.settings):
                return method(self, *args, **kwargs)
        finally:
            self._settled = False

    run.__session_settings__ = True  # type: ignore[attr-defined]
    return run


def _enters_its_settings(function: Callable[..., _T]) -> Callable[..., _T]:
    """Mark an operation that enters the session's configuration itself:
    :meth:`Session.open`, which makes it.
    """
    function.__session_settings__ = True  # type: ignore[attr-defined]
    return function


class Session:
    """One scene of one file, executed and recorded.

    Open it with :meth:`open`; it holds the executed scene, whose camera
    draws its frames.
    """

    def __init__(
        self,
        scene: Scene,
        record: Record,
        module: ModuleType,
        recorder: Recorder,
        *,
        draw_path: str | None = None,
    ) -> None:
        self.scene = scene
        #: The draw path its frames take: the one it was executed under when
        #: opened, else the option's value at each frame.
        self.draw_path = draw_path
        self.record = record
        self.module = module
        # The recorder the session made: read from it, never from a name of
        # the scene's, which the scene's own code can take.
        self._recorder = recorder
        self._quality: str | None = None
        self._static: tuple[int, str | None, PixelArray | None] | None = None
        #: The frames it drew last, which a frame drawn from the same
        #: recorded state reuses (``draw_reuse``): of this execution only.
        self.draws = DrawReuse(SESSION_DRAWS)
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
        #: What :meth:`open` executed last, the execution the session holds,
        #: which a forward run executes again.
        self._source: Source | None = None
        # Whether an operation of the session's runs, under its settings.
        self._settled = False
        #: The forward runs that answered what the record could not, in order.
        self.forward_runs: list[ForwardRun] = []
        #: Whether the scene's source was found changed since the record was
        #: made (by a forward run, by the probe recording it again, or by a
        #: reload of changed files that did not complete): the record is of
        #: a program the file no longer is.
        self.stale = False
        #: Which of the session's executions it holds: 1 as opened, and one
        #: more for each :meth:`reload` that executed the scene again.
        self.generation = 1
        #: What :meth:`open` was asked, which :meth:`reload` asks again.
        self._opened: dict[str, Any] | None = None
        self._closed = False

    @classmethod
    @_enters_its_settings
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
        overrides the configuration the scene is executed under (its draw
        path too: the session's frames take the one it was opened with);
        nothing is written either way.

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
        An execution of another source than the one before it (the file
        edited meanwhile) stops it too, with no divergence: the session is
        :attr:`stale`.

        What an execution of the scene raises (importing the file, a scene
        it does not define, the scene's code, the engine refusing it) is
        raised as it was raised, its type, message and traceback unchanged;
        its explanation rides on it as ``scene_error``
        (:func:`~arachne.errors.explain`, at the scene clock's time once the
        scene had started), and its time and hints as a note a printed
        traceback shows.
        """
        return cls._open(
            Path(file).resolve(),
            scene,
            named=file,
            grid=grid,
            config=config,
            stepping=stepping,
            on_step=on_step,
            on_tick=on_tick,
            on_start=on_start,
            instants=instants,
            probe=probe,
        )

    @classmethod
    def _open(
        cls,
        path: Path,
        scene: str,
        *,
        named: StrPath,
        grid: Fraction | float | None | Literal["render"],
        config: dict[str, Any] | None,
        stepping: bool | Literal["brute_force"],
        on_step: Callable[[Scene, Fraction], object] | None,
        on_tick: Callable[[Scene, Fraction, bool], object] | None,
        on_start: Callable[[Scene, Fraction], object] | None,
        instants: Iterable[Fraction | float],
        probe: bool,
    ) -> Session:
        """:meth:`open`, of the file at ``path``, a resolved path, which
        the caller ``named`` so: what its errors name it.
        """
        settings = {
            **(config or {}),
            "dry_run": True,
            "disable_caching": True,
        }
        with tempconfig(settings):
            # The whole configuration the scene runs under, which every
            # execution of the session, a forward run's too, executes under,
            # whatever the process's is then.
            effective = global_config.copy()
        # Executing, probing (its restores run callbacks) and comparing
        # read that configuration too.
        with tempconfig(effective):
            options = {
                "grid": grid,
                "stepping": stepping,
                "on_step": on_step,
                "on_tick": on_tick,
                "on_start": on_start,
                "instants": tuple(instants),
            }
            # Each demoted play's identity, with its line's plays where it was
            # found (``None`` for a line demoted whole).
            demotions: dict[PlayIdentity, tuple[str, ...] | None] = {}
            session = cls(
                *_execute(path, scene, effective, options, demotions, named),
                draw_path=effective.draw_path,
            )
            session._opened = {
                "path": path,
                "scene": scene,
                "named": named,
                "config": dict(config) if config is not None else None,
                **options,
                "probe": probe,
            }
            if not probe:
                session._source = _source(session, path, scene, effective, stepping)
                return session
            # Probe the record. On a demotion, record the scene again with the
            # play ticked, a new execution compared with the first, and probe
            # that: each pass ticks at least one play more, while the executions
            # play the same plays of the same source. When they don't, the probe
            # stops, and the session holds the last execution. A demotion follows
            # its play's identity; where its line's plays changed, which play it
            # was is unknown, and every play of the line ticks.
            latest = found = probe_record(session.record)
            described = session.record
            probing = True
            first: ExecutionSummary | None = None
            # A backstop: each pass ticks one more play, or every play of one
            # more line a play was demoted at, so this many executions suffice
            # while the executions play the same plays.
            cap = 2 * len(session.record.plays) + 1
            executions = 1
            settled = True
            while True:
                record = session.record
                digest = record.identity.source_digest
                demoting: dict[PlayIdentity, tuple[str, ...] | None] = {}
                if probing:
                    for divergence in latest.divergences:
                        session._keep(divergence)
                    identities = play_identities(record.plays)
                    lines = line_plays(record.plays)
                    for index in latest.demoted:
                        identity = identities[index]
                        if identity not in demotions:
                            demoting[identity] = lines[identity[0]]
                demoting.update(
                    (whole_line(line), None)
                    for line in _changed_lines({**demotions, **demoting}, record)
                )
                if not demoting:
                    break
                demotions.update(demoting)
                if digest is not None:
                    for identity, line in demoting.items():
                        admission.demote(digest, record.scene_class, identity, line)
                if executions >= cap:
                    settled = False
                    break
                if first is None:
                    first = session.summary()
                session._executed_again(
                    *_execute(path, scene, effective, options, demotions, named)
                )
                executions += 1
                with contextlib.suppress(Incomparable):
                    session.compare(first)
                if not _same_source(record, session.record):
                    # Another source is stale, never a divergence, whether or not
                    # the probe still runs.
                    session.stale = True
                    probing = False
                    settled = False
                elif probing and _schedule_shift(record, session.record):
                    # The same source playing other plays is where its
                    # executions differ.
                    changed = _schedule_change(record, session.record)
                    if changed is not None:
                        session._keep(changed)
                    probing = False
                    settled = False
                if probing:
                    latest = probe_record(session.record)
                    found = found.then(latest)
                    described = session.record
            whole = {line for line, kind, _ in demotions if kind is None}
            found = session._found_in(found, described, whole)
            session.probe = replace(found, settled=settled)
            session._source = _source(session, path, scene, effective, stepping)
            return session

    @_under_its_settings
    def state_at(self, t: Fraction | float) -> State:
        """The scene at ``t``, from the record: restored there when a
        state-function play holds ``t`` and the record did not sample it.
        """
        return self.record.state_at(t)

    @_under_its_settings
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
        (pixels,) = self._drawn(
            record, [_resolve(record, exact(t))], self._static_picture
        )
        return self.scene.renderer.camera.get_image(pixels)

    def _drawn(
        self,
        record: Record,
        answers: Iterable[_Resolved],
        static: Callable[[Drawing, PlayRecord], PixelArray | None],
    ) -> list[PixelArray]:
        """The frames of ``answers``, instants ``record`` answers, drawn as
        every frame a session gives is drawn: the session's own record's, and
        a forward run's (:mod:`.timeline.fallback`). ``static(drawing, play)``
        gives a play's static picture. Paths map around the centre the render
        froze at its first frame, whatever order frames are asked in
        (:mod:`.timeline.mapping`).

        Called inside an operation, so under the configuration the session
        was opened with (:func:`_under_its_settings`): what the camera reads
        while it draws (a point cloud's thickness reads the frame's pixel
        size), and what a restore's callbacks read, are the session's.
        """
        freeze_mapping_centre(record)
        reuse = self.draws if global_config.draw_reuse else None
        with Drawing(record, draw_path=self.draw_path, reuse=reuse) as drawing:
            return [
                _draw(drawing, found, lambda play: static(drawing, play))
                for found in answers
            ]

    @_under_its_settings
    def clear_draws(self) -> None:
        """Keep none of the frames drawn so far: the next frame asked at any
        instant is drawn, not reused (a timing that asks for drawing).
        """
        self.draws.clear()

    @_under_its_settings
    def answers(
        self,
        instants: Iterable[Fraction | float],
        quality: str | None = None,
        *,
        frames: bool = True,
    ) -> list[Answer]:
        """The state at each of ``instants``, and its frame unless ``frames``
        is false (at ``quality`` as :meth:`frame` draws it), in order.

        The record answers what it can. The instants it can't (a state play
        that can't be restored there, a record that refuses drawing) are
        answered by one forward run for them all (:mod:`.timeline.fallback`):
        a new execution of the scene, whose answers say ``"forward"`` and
        name why the record could not answer, and which is compared with
        the record's execution (:attr:`forward_runs`, :attr:`divergences`).
        A forward run of a source changed since the record is compared with
        nothing: its answers say ``source-changed``, and :attr:`stale` is set.
        """
        return answer_instants(self, instants, quality, frames=frames)

    @_under_its_settings
    def lint(self, **options: Any) -> LintResult:
        """Lint the session (:func:`arachne.lint.lint`, given ``options``):
        with ``rules="armed"``, the temporal rules its scene armed.
        """
        from .lint import lint

        return lint(self, **options)

    @_under_its_settings
    def query(self, t: Fraction | float) -> Query:
        """Numbers about the frame on screen at ``t``, measured as the lint
        measures it (:func:`arachne.lint.query.query`): nothing is drawn, but
        for a record that refuses drawing, read from a forward run that does.
        """
        from .lint.query import query

        return query(self, t)

    @_under_its_settings
    def key_frames(self, out_dir: StrPath, quality: str | None = None) -> KeyFrames:
        """One PNG per play or wait in ``out_dir``, its last frame
        (:func:`arachne.lint.keyframes.key_frames`).
        """
        from .lint.keyframes import key_frames

        return key_frames(self, out_dir, quality)

    @_under_its_settings
    def storyboard(self, out_dir: StrPath, quality: str | None = None) -> Storyboard:
        """Sheets of six labelled key frames in ``out_dir``
        (:func:`arachne.lint.keyframes.storyboard`).
        """
        from .lint.keyframes import storyboard

        return storyboard(self, out_dir, quality)

    @_under_its_settings
    def summary(self) -> ExecutionSummary:
        """The summary of the session's execution, which another execution
        of its source, in this process or another, is compared with
        (:meth:`compare`, :meth:`.ExecutionSummary.to_json`).
        """
        if self._summary is None or self._summary[0] is not self.record:
            self._summary = (self.record, summarize(self.record))
        return self._summary[1]

    @_under_its_settings
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

    def _found_in(
        self, found: Probe, before: Record, whole: set[SourceLine | None]
    ) -> Probe:
        """``found``, the probe of ``before``, moved to the record held, an
        execution recorded after it: each play at its index there by its
        identity, a play it does not play, or one of a line demoted
        ``whole``, left out and its divergence naming no play (only its
        line); the same for the probe's divergences kept.
        """
        places = {
            identity: index
            for index, identity in enumerate(play_identities(self.record.plays))
            if identity[0] not in whole
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

    def _executed_again(
        self, scene: Scene, record: Record, module: ModuleType, recorder: Recorder
    ) -> None:
        """Hold the scene executed again, ``record`` its record, in place of
        the one held.
        """
        sys.modules.pop(self.module.__name__, None)
        self.scene, self.record, self.module = scene, record, module
        self._recorder = recorder
        camera = scene.renderer.camera
        self._executed_shape = (camera.pixel_height, camera.pixel_width)
        self._quality = None
        self._static = None
        self.draws = DrawReuse(SESSION_DRAWS)

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
        resize(camera, height, width)
        self._quality = quality
        self._static = None

    @_under_its_settings
    def changed(self) -> bool:
        """Whether the files the session's execution came from hold anything
        else now: the scene's source (the scene file and the closure of its
        imports, as the files name them now: a file edited or gone, or a
        module newly imported) or a file the execution read, edited or gone.
        Each is digested every time asked, whatever its size and
        modification time say. A source the session can't show its record
        was made from counts as changed. Asking changes nothing: the session
        is not :attr:`stale` for it.
        """
        source = self._opened_source()
        if source.files is None:
            return True
        scene_file = os.fspath(source.path)
        scene_dir = os.path.dirname(scene_file)
        try:
            if file_digests(source_files(scene_file), scene_dir) != source.files:
                return True
            return any(
                file_digests([os.path.join(scene_dir, read.path)], scene_dir)[0][1]
                != read.digest
                for read in source.reads
            )
        except OSError:
            return True

    @_under_its_settings
    def reload(self, force: bool = False) -> bool:
        """Execute the scene again when its files changed (:meth:`changed`),
        or whatever they hold with ``force``, as :meth:`open` does, under
        the configuration it was opened with: whether it did. The session
        then holds that execution as its next :attr:`generation`, probed
        afresh, with no divergence, forward run or staleness of the one
        before: an edit and then a reload is a new program, not two
        executions of one differing.

        It executes the file it was opened from, wherever the working
        directory is now. An execution that raises, or ends otherwise,
        raises as :meth:`open` does, and the session keeps the generation it
        held: :attr:`stale` when its files had changed, since its record is
        then of a program they no longer are.
        """
        if self._opened is None:
            raise RuntimeError(self._not_opened)
        changed = self.changed()
        if not (changed or force):
            return False
        if changed:
            # Until the new execution is held, whatever ends it.
            self.stale = True
        executed = type(self)._open(**self._opened)
        sys.modules.pop(self.module.__name__, None)
        generation, settled = self.generation + 1, self._settled
        # Every attribute is the execution's, as opening it made them, but
        # the generation and whether an operation runs now.
        vars(self).update(vars(executed))
        self.generation, self._settled = generation, settled
        return True

    _not_opened = (
        "This session was not opened from its file, so its scene can't be"
        " executed again."
    )

    def _opened_source(self) -> Source:
        if self._source is None:
            raise RuntimeError(self._not_opened)
        return self._source

    @_under_its_settings
    def close(self) -> None:
        """Let go of the scene, its record and its module: the scene's
        module is dropped from ``sys.modules`` (the modules it imported
        from its directory were dropped as each execution ended,
        :func:`.loader.imports`), and the session holds neither. A closed
        session answers nothing; closing it again does nothing.
        """
        if self._closed:
            return
        self._closed = True
        sys.modules.pop(self.module.__name__, None)
        del self.scene, self.record, self.module, self._recorder
        self._summary = self._static = None
        self.draws.clear()


def resize(camera: Any, height: int, width: int) -> None:
    """Give ``camera`` ``height`` by ``width`` pixels. The frame keeps its
    size in scene units, as a render at another quality has it; only the
    pixels change.
    """
    camera.pixel_height, camera.pixel_width = height, width
    camera.init_background()
    camera.pixel_array_to_cairo_context.clear()
    camera.reset()


@dataclass(frozen=True)
class Source:
    """What a session executed: the scene class ``scene`` of ``path``, under
    the configuration ``settings``, its ticked plays stepping as
    ``stepping`` says; and each file of its source, as ``(path relative to
    the scene's directory, sha256)``, by path, which the record's source
    digest digests (``None`` when it can't be shown to be that); the files
    its execution read, each with the play it was first read for; and the
    identities of the plays it ticked as the probe demoted them, which a
    forward run ticks too, so that its routes are the record's.
    """

    path: Path
    scene: str
    settings: ManimConfig
    stepping: bool | Literal["brute_force"]
    files: tuple[tuple[str, str], ...] | None
    reads: tuple[Read, ...] = ()
    demoted: frozenset[PlayIdentity] = frozenset()


def _source_files(record: Record) -> tuple[tuple[str, str], ...] | None:
    """The record's source, file by file, when it is still the source its
    digest was taken of.
    """
    scene_file = record.scene_file
    if scene_file is None or record.identity.source_digest is None:
        return None
    scene_dir = os.path.dirname(scene_file)
    files = source_files(scene_file)
    try:
        found = file_digests(files, scene_dir)
    except OSError:
        return None
    if source_digest(files, scene_dir) != record.identity.source_digest:
        return None
    return found


def _reads(recorder: Recorder, record: Record) -> tuple[Read, ...]:
    """The files ``recorder``'s execution read, its source's but."""
    inputs = recorder._inputs
    if inputs is None or record.scene_file is None:
        return ()
    return inputs.reads(frozenset(source_files(record.scene_file)))


def execute(
    path: Path,
    scene: str,
    settings: ManimConfig | dict[str, Any],
    recorder: Callable[[Scene], Recorder],
    then: Callable[[Scene], object] | None = None,
) -> tuple[Scene, ModuleType]:
    """Import ``path`` under a module of its own and execute its scene class
    ``scene`` under ``settings``, recorded by ``recorder(scene)``: the scene
    and its module. The modules of the scene's directory it imports are its
    own, loaded from their source, and gone from ``sys.modules`` afterwards
    (:func:`.loader.imports`). ``then`` is called with the scene once its
    script ended, before they go, so that the scene's code it calls (a
    forward run's visits after the end) imports what the execution did. An
    execution that raises leaves its module unimported.
    """
    base_dir = str(path.parent)
    # Each execution starts as a fresh process does: another execution of
    # the source in this process left its construction caches filled.
    forget_construction_caches()
    with imports(path):
        with tempconfig(settings):
            ORIGINS.start(base_dir, str(path))
            # The pixel shape configured before the scene's file runs: what a
            # frame its record draws is drawn at.
            configured = (global_config.pixel_width, global_config.pixel_height)
            try:
                module = _import(path)
                try:
                    scene_class = getattr(module, scene, None)
                    if not isinstance(scene_class, type):
                        raise LookupError(
                            f"{path.name} defines no scene named {scene!r}."
                        )
                    instance: Scene = scene_class()
                    instance._arachne_timeline = recorder(instance)
                    instance._arachne_timeline.configured_pixel_shape = configured
                    instance.render()
                except BaseException:
                    sys.modules.pop(module.__name__, None)
                    raise
            finally:
                ORIGINS.stop()
        if then is not None:
            then(instance)
    return instance, module


def _source(
    session: Session,
    path: Path,
    scene: str,
    settings: ManimConfig,
    stepping: bool | Literal["brute_force"],
) -> Source:
    """What ``session`` executed last, the execution it holds, with the
    demotions that execution ticked (not those found after it, which a
    capped session registered but did not apply).
    """
    record = session.record
    recorder = session._recorder
    return Source(
        path,
        scene,
        settings,
        stepping,
        _source_files(record),
        _reads(recorder, record),
        recorder.demoted,
    )


def _execute(
    path: Path,
    scene: str,
    settings: ManimConfig,
    options: dict[str, Any],
    demotions: dict[PlayIdentity, tuple[str, ...] | None],
    named: StrPath,
) -> tuple[Scene, Record, ModuleType, Recorder]:
    """Execute ``path``'s scene ``scene`` under ``settings`` (:func:`execute`),
    recorded by a ``Recorder`` given ``options`` that ticks the plays of
    ``demotions`` and those the probe demoted in an earlier session of the
    source (:func:`.admission.demoted`), which join ``demotions``. The record
    is the recorder's: a scene class may name an attribute ``record`` too.
    What the execution raises is raised as it was, its explanation on it
    (:func:`~.errors.explain_on`): the scene file as the caller ``named`` it, at the
    scene clock's time once the scene exists.
    """
    made: list[Recorder] = []

    def recorder(instance: Scene) -> Recorder:
        for identity, line in admission.demoted(
            _source_digest(instance), type(instance).__name__
        ).items():
            demotions.setdefault(identity, line)
        grid = options["grid"]
        made.append(
            Recorder(
                instance,
                **{
                    **options,
                    "grid": global_config.frame_rate if grid == "render" else grid,
                },
                draws=False,
                demoted=demotions,
                exits="armed",
                states="armed",
            )
        )
        return made[-1]

    try:
        instance, module = execute(path, scene, settings, recorder)
    except BaseException as error:
        # A scene's exit too: a caller that serves on goes on with it.
        explain_on(error, named, made[-1].scene.time if made else None)
        raise
    record = made[-1].record
    assert record is not None
    return instance, record, module, made[-1]


def _changed_lines(
    demotions: dict[PlayIdentity, tuple[str, ...] | None], record: Record
) -> set[SourceLine | None]:
    """The lines a play was demoted at whose plays in ``record`` are not the
    kinds, in order, they were where it was found, and that are not demoted
    whole yet: which of them the demotion was for is unknown.
    """
    lines = line_plays(record.plays)
    whole = {line for line, kind, _ in demotions if kind is None}
    return {
        line
        for (line, kind, _), found in demotions.items()
        if kind is not None and line not in whole and lines.get(line, ()) != found
    }


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
    never share a module, compiled from what the file holds now.
    """
    name = f"arachne_session_{next(_MODULE_NUMBERS)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} is not a Python file.")
    if isinstance(spec.loader, SourceFileLoader):
        spec.loader = FromSource(name, spec.loader.path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
