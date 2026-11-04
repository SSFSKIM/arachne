"""The forward fallback: what a session's record can't answer, a new
execution of its scene answers.

A record can't answer an instant of a state play it can't restore there
(:class:`.restore.Unrestorable`), nor any instant of a scene whose frames it
can't draw (:class:`.routes.RecordRefused`). :func:`answer_instants` lets
the record answer what it can and gathers the rest into one batch, which one
forward run answers. The scene is executed again in process, as the session
executed it (its file, configuration and stepping), and stops at the first
play that begins after the last instant asked.

- **A state play that can't be restored.** The record draws its scene, and
  drawing it depends on nothing but the captures. The forward run, on the
  session's grid, evaluates every instant asked, and the frame at or before
  it, where a trusted or state play holds it (``Recorder(instants=…)``), as
  a non-frame evaluation. Each instant is then answered from the new
  execution's record by the route of the play holding it, as a session
  answers from its own: a capture of the new execution, drawn from it.
- **A record that refuses drawing.** Its frames are drawn while the scene
  runs, and what a frame shows may depend on every frame drawn before it (a
  zoomed display shows the last picture drawn into it). The forward run is a
  render, on the render's grid: every frame is drawn as the render draws it
  (from the record, or as the play runs), and those asked for are kept. An
  instant is answered as that render shows it: by the last frame at or
  before it, and the state that frame shows; after the script ended, by the
  scene as it left it.

:func:`visit_frames` runs the second kind of forward run for a record that
draws as well, without drawing: on the render's grid, its frames
evaluated as the render evaluates them, a visitor shown each one asked.

The answer is labelled ``"forward"`` and names why the record could not
answer (:class:`Fallback`). It is never an answer about the record: the new
execution is compared with the record's over what it ran
(:func:`.execution.compare`), and a difference is kept on the session as a
``replay.diverged`` divergence of kind ``execution`` naming the input files
the two executions read differently for the plays compared: each file is
kept with the digest of what each execution read and the first play it was
read for (:class:`.inputs.Read`), never with what the file holds now.

A forward run leaves the session as it found it: its record and answers,
its scene's module and the process's origin table and random generators.
The modules the scene imports beside it stay those already imported, so a
helper edited since the session opened runs as it was.
"""

from __future__ import annotations

__all__ = [
    "Answer",
    "Fallback",
    "ForwardRun",
    "Visit",
    "answer_instants",
    "visit_frames",
]

import bisect
import math
import os
import random
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .admission import whole_line
from .clock import exact
from .dispatch import _all_engine_classes
from .draw import Drawing, LiveFrames
from .execution import (
    Divergence,
    Execution,
    ExecutionSummary,
    Incomparable,
    _begun,
    compare,
    summarize,
)
from .inputs import Read, file_digests, source_files
from .origins import ORIGINS
from .recorder import Recorder
from .restore import Unrestorable
from .routes import RecordRefused, _last_frame, _Resolved, _route
from .state import build_state

if TYPE_CHECKING:
    from PIL import Image

    from arachne.scene.scene import Scene
    from arachne.session import Session, Source
    from arachne.typing import PixelArray

    from .admission import PlayIdentity
    from .record import Capture, PlayRecord, Record
    from .state import State

#: What a forward run shows a visitor at each instant it answers: the
#: instant asked, the run's record (unfinished while it runs), the play
#: holding the frame shown there (``None`` after the script ended) and the
#: capture of it, while the live scene holds that capture's state. For a
#: record that refuses drawing, the frame shown is the render's frame on
#: screen at the instant; otherwise the instant itself, which the run keeps
#: (a frame of the grid in a ticked play, or any instant of another play).
Visit = Callable[[Fraction, "Record", "PlayRecord | None", "Capture"], None]


@dataclass(frozen=True)
class Fallback:
    """Why the record could not answer an instant, so that a forward run did:
    ``kind`` is ``"refused"`` (the record draws no frame,
    :class:`.RecordRefused`), ``"unrestorable"``
    (:class:`.restore.Unrestorable`) or ``"source-changed"`` (the scene's
    source changed since the record was made, which is stale:
    :attr:`.Session.stale`); ``reason`` is the stable code
    (``camera.mapping_func``, ``restore.global_rebound``,
    ``source.changed``, …) and ``detail`` the message.
    """

    kind: str
    reason: str
    detail: str

    @classmethod
    def of(cls, error: RecordRefused | Unrestorable) -> Fallback:
        """The fallback a refusal of the record calls for."""
        if isinstance(error, Unrestorable):
            return cls("unrestorable", error.reason, str(error))
        detail = str(error)
        return cls("refused", detail.split(":", 1)[0], detail)


@dataclass(frozen=True)
class Answer:
    """The scene at ``t`` as a session answers it: its state, and its frame
    when asked for. ``fallback`` says why the record could not answer, for
    an answer a forward run gave (``state.answer == "forward"``), and is
    ``None`` for the record's own.
    """

    t: Fraction
    state: State
    frame: Image.Image | None
    fallback: Fallback | None = None


@dataclass(frozen=True)
class ForwardRun:
    """One forward run: the instants it answered, in time order, how many
    plays it ran before it stopped (every play, when it ran the script to
    its end), and where it differs from the record's execution if it does.

    What was compared: up to ``compared_until`` (the last instant asked, for
    a run that stopped; ``None``: to the end of both), and of those plays
    only the first ``compared_plays`` when a play was drawn as it ran, whose
    drawing may write into the scene from its start on (``None``: no such
    bound). ``source_changed``: the scene's source is not the record's (or
    one of the two is unknown), so the record is stale and nothing was
    compared.
    """

    instants: tuple[Fraction, ...]
    plays: int
    divergence: Divergence | None
    compared_until: Fraction | None
    compared_plays: int | None = None
    source_changed: bool = False


def answer_instants(
    session: Session,
    instants: Iterable[Fraction | float],
    quality: str | None,
    *,
    frames: bool,
    visit: Visit | None = None,
) -> list[Answer]:
    """:meth:`.Session.answers`: the record's answers, and one forward run's
    for every instant the record can't answer. ``visit`` is called with the
    live scene of that run at each instant it answers (:data:`Visit`), so
    that what a state does not hold can be read there.
    """
    asked = [exact(t) for t in instants]
    found: dict[Fraction, Answer] = {}
    missing: dict[Fraction, Fallback] = {}
    for t in dict.fromkeys(asked):
        try:
            state = session.state_at(t)
            frame = session.frame(t, quality) if frames else None
        except (Unrestorable, RecordRefused) as error:
            missing[t] = Fallback.of(error)
        else:
            found[t] = Answer(t, state, frame)
    if missing:
        if frames:
            # The size the forward run draws at, as frame() draws at it.
            session._use_quality(quality)
        answered, stale = _forward(session, sorted(missing), frames=frames, visit=visit)
        for t, (state, frame) in answered.items():
            found[t] = Answer(t, state, frame, stale or missing[t])
    return [found[t] for t in asked]


def visit_frames(
    session: Session,
    instants: Iterable[Fraction],
    visit: Visit,
    *,
    to_end: bool = False,
    states: bool = False,
) -> Record:
    """Execute the session's scene once more on its render's frame grid and
    show ``visit`` the live scene at each of ``instants``, frames of that
    grid (or the end of the scene), whether or not the record answers them:
    each evaluated as the render evaluates that frame, through the scene's
    own frame hooks, where the record may hold no capture of it. The run is
    kept and compared as :func:`answer_instants`' is. With ``to_end`` it
    runs to the scene's end rather than stopping past the last instant;
    with ``states`` its record keeps the states the lint's temporal rules
    read (``Record.states``). Returns the run's record.
    """
    executed: list[Record] = []
    _forward(
        session,
        sorted({exact(t) for t in instants}),
        frames=False,
        visit=visit,
        render_grid=True,
        to_end=to_end,
        states=states,
        runs=executed,
    )
    return executed[0]


class _Stopped(BaseException):  # noqa: N818 (an instruction, not an error)
    """The forward run is past the last instant asked. Not an ``Exception``,
    so that a scene catching its own errors lets it through.
    """


class _Unforeseen(BaseException):  # noqa: N818 (an instruction, not an error)
    """A play of the forward run the record would draw is found to need
    drawing as it runs, which the record's execution never found: the
    render would execute the scene again to draw it so.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _ForwardRecorder(Recorder):
    """Records a forward run as a session's recorder records, stopping at
    the first play that begins after ``stop_after``; when it ``draws``, it
    draws as a render's recorder does, with the plays ``known`` to refuse
    mid-play drawn as they run from the earliest on, as the render's last
    execution draws them. It ticks the plays ``demoted`` (identities, or
    whole lines, as :class:`.Recorder` takes them).
    """

    def __init__(
        self,
        scene: Scene,
        *,
        grid: Fraction | None,
        draws: bool,
        stepping: bool | Literal["brute_force"],
        instants: Iterable[Fraction],
        stop_after: Fraction | None,
        known: Mapping[int, str],
        demoted: Iterable[PlayIdentity],
        visit: Visit | None = None,
        shown: Mapping[Fraction, Fraction] | None = None,
        states: bool = False,
    ) -> None:
        super().__init__(
            scene,
            grid=grid,
            draws=draws,
            stepping=stepping,
            instants=instants,
            demoted=demoted,
            exits="all",
            states="all" if states else "none",
        )
        if known:
            self.execution = Execution(known=dict(known))
        self.stop_after = stop_after
        #: The index of the first play drawn as it ran, once one has.
        self.live_from: int | None = None
        self.visit = visit
        #: The instants asked, ordered by the instant whose frame shows each.
        self._shown = sorted(((at, t) for t, at in (shown or {}).items()))
        self._shown_at = [at for at, _ in self._shown]
        #: The first error a visit raised, raised again once the run is out
        #: of the scene's code, which might have caught it.
        self.visit_error: Exception | None = None

    def refusal(self) -> str | None:
        # The renderer asks this first as each play begins.
        if self.stop_after is not None and self.scene.time > self.stop_after:
            raise _Stopped
        return super().refusal()

    def play(
        self,
        *,
        skipped: bool,
        cache_key: str | None,
        live: LiveFrames | None = None,
    ) -> PlayRecord:
        if live is not None and self.live_from is None:
            self.live_from = len(self.ensure_record().plays)
        played = super().play(skipped=skipped, cache_key=cache_key, live=live)
        if self.draws and live is None and played.draw_refusal is not None:
            raise _Unforeseen(played.draw_refusal)
        return played

    def capture(
        self,
        kind: str,
        instant: Fraction,
        play: PlayRecord,
        extras: Any,
        until: Fraction | None = None,
    ) -> Any:
        captured = super().capture(kind, instant, play, extras, until)
        if self.visit is not None and kind == "sample":
            low = bisect.bisect_left(self._shown_at, instant)
            high = (
                len(self._shown)
                if until is None
                else bisect.bisect_left(self._shown_at, until)
            )
            for _, t in self._shown[low:high]:
                self.visited(t, play, captured)
        return captured

    def visited(self, t: Fraction, play: PlayRecord | None, capture: Capture) -> None:
        """Show the visitor the live scene at ``t``; a later capture of the
        frame shown there (the next play's, where one play ended early)
        visits it again.
        """
        assert self.visit is not None
        try:
            self.visit(t, self.ensure_record(), play, capture)
        except Exception as error:
            if self.visit_error is None:
                self.visit_error = error

    def stopped(self) -> None:
        """The run stopped before its script ended: its identity is still
        that of what it ran (its source, the files it read so far).
        """
        record = self.ensure_record()
        record.identity = self._identity(record)


def _forward(
    session: Session,
    instants: list[Fraction],
    *,
    frames: bool,
    visit: Visit | None = None,
    render_grid: bool = False,
    to_end: bool = False,
    states: bool = False,
    runs: list[Record] | None = None,
) -> tuple[dict[Fraction, tuple[State, Image.Image | None]], Fallback | None]:
    """Execute the session's scene again for ``instants`` (in time order),
    keep the run in :attr:`.Session.forward_runs` and any divergence in
    :attr:`.Session.divergences`, and answer each instant from it; with the
    fallback every answer takes when the source changed since the record.
    ``visit`` sees the live scene at each instant answered. With
    ``render_grid`` the run is on the render's grid whatever the record's;
    with ``states`` its record keeps the temporal rules' states; the run's
    record is appended to ``runs`` when given.
    """
    from arachne.session import _reads, execute

    source = session._source
    if source is None:
        raise RuntimeError(
            "This session was not opened from its file, so its scene can't be"
            " executed again to answer what its record can't."
        )
    record = session.record
    refused = record.refusal is not None
    camera = session.scene.renderer.camera
    shape = (camera.pixel_height, camera.pixel_width)
    written: dict[Fraction, PixelArray] = {}
    made: list[_ForwardRecorder] = []

    def recorder(scene: Scene) -> Recorder:
        forward_camera = scene.renderer.camera
        _resize(forward_camera, shape)
        if refused or render_grid:
            fps = exact(forward_camera.frame_rate)
            grid: Fraction | None = fps
            extra: frozenset[Fraction] = frozenset()
            if refused:
                # Drawn whether or not frames are asked for: what a play drawn
                # as it runs draws may be written into the scene.
                asked = _on_screen(fps, instants) if frames else []
                _keep_frames(scene, fps, asked, written)
            # Each instant is answered by the frame on screen there.
            shown = dict(zip(instants, _on_screen(fps, instants), strict=True))
        else:
            grid = record.grid
            extra = frozenset(instants) | {_last_frame(record, t) for t in instants}
            # Each instant is evaluated and kept, and answered by itself.
            shown = {t: t for t in instants}
        forward = _ForwardRecorder(
            scene,
            grid=grid,
            draws=refused,
            stepping=source.stepping,
            instants=extra,
            stop_after=None if to_end else instants[-1],
            known={
                play.index: play.draw_refusal
                for play in record.plays
                if play.draw_refusal is not None
            },
            # Every play of a line the probe demoted a play of: which of them
            # it was is unknown before the run plays them, and a ticked play
            # answers as the render shows it, demoted or not.
            demoted={whole_line(line) for line, _, _ in source.demoted},
            visit=visit,
            shown=shown,
            states=states,
        )
        made.append(forward)
        return forward

    with _kept():
        try:
            _, module = execute(source.path, source.scene, source.settings, recorder)
        except _Stopped:
            (forward,) = made
            forward.stopped()
            whole = False
        except _Unforeseen as unforeseen:
            raise RecordRefused(
                f"{unforeseen.reason}; the forward run found it where the record's"
                " execution did not, so it is another execution, whose frames"
                " from there on are drawn only by executing it again"
            ) from None
        else:
            (forward,) = made
            sys.modules.pop(module.__name__, None)
            whole = True
        executed = forward.ensure_record()
        if runs is not None:
            runs.append(executed)
        if visit is not None and whole:
            # After the script ended, the scene as it left it.
            for t in instants:
                if executed.play_at(t) is None and t >= executed.duration:
                    assert executed.final is not None
                    forward.visited(t, None, executed.final)
        if forward.visit_error is not None:
            raise forward.visit_error
        until = None if whole else instants[-1]
        # What a play drawn as it runs draws may be written into the scene (a
        # zoomed display's picture, a mapped mobject's points), from its
        # static picture on, which the record's execution, drawing nothing,
        # never holds: only the plays before the first are compared.
        before_play = forward.live_from
        mine, theirs = session.summary(), summarize(executed)
        stale: Fallback | None = None
        try:
            divergence = compare(mine, theirs, until=until, before_play=before_play)
        except Incomparable:
            # Another program than the record's: nothing to compare, and no
            # divergence (two executions of one source differing), but the
            # record is stale.
            divergence = None
            stale = _source_changed(source, record)
            session.stale = True
        if divergence is not None:
            # What the two executions read differently for the plays compared.
            divergence = replace(
                divergence,
                changed_inputs=_changed_inputs(
                    source.reads,
                    _reads(forward, executed),
                    (
                        _bound(mine, until, before_play),
                        _bound(theirs, until, before_play),
                    ),
                ),
            )
        session.forward_runs.append(
            ForwardRun(
                tuple(instants),
                len(executed.plays),
                divergence,
                until,
                before_play,
                source_changed=stale is not None,
            )
        )
        if divergence is not None:
            session._keep(divergence)
        if refused:
            answered = _rendered(forward, executed, instants, written, frames=frames)
        elif executed.refusal is not None:
            raise RecordRefused(
                f"{executed.refusal}; the forward run's record refuses drawing where"
                " the record's execution did not"
            )
        else:
            answered = _recorded(session, executed, instants, frames=frames)
        return answered, stale


def _source_changed(source: Source, record: Record) -> Fallback:
    """Why a forward run of another source than the record's answered: the
    scene's source changed since the record was made, which is no
    divergence. It names the files that changed (added and removed ones
    included), or says the source differs when they can't be told.
    """
    changed: list[str] = []
    scene_file = record.scene_file
    if source.files is not None and scene_file is not None:
        scene_dir = os.path.dirname(scene_file)
        try:
            now = dict(file_digests(source_files(scene_file), scene_dir))
        except OSError:
            now = None
        if now is not None:
            before = dict(source.files)
            changed = sorted(
                path
                for path in before.keys() | now.keys()
                if before.get(path) != now.get(path)
            )
    if changed:
        names = ", ".join(changed)
        detail = (
            f"the scene's source changed since the record was made ({names}): the"
            " record is stale, and these are the answers of the source as it is now"
        )
    else:
        detail = (
            "the scene's source differs from the record's, or is unknown for one"
            " of them: the record is stale, and these are the answers of the"
            " source as it is now"
        )
    return Fallback("source-changed", "source.changed", detail)


def _bound(
    summary: ExecutionSummary, until: Fraction | None, before_play: int | None
) -> int | None:
    """How many of the plays of ``summary`` :func:`.execution.compare`
    compared (``None``: all of them, and where its script ended).
    """
    if until is None and before_play is None:
        return None
    return len(_begun(summary.plays, until)[:before_play])


def _changed_inputs(
    mine: tuple[Read, ...],
    theirs: tuple[Read, ...],
    bounds: tuple[int | None, int | None],
) -> tuple[str, ...]:
    """The input files two executions read differently for the plays each
    compared, the first ``bounds[0]`` of ``mine`` and ``bounds[1]`` of
    ``theirs`` (every play, and the script's end, for ``None``), by path: a
    file that each first read for one of its compared plays, with different
    contents, or that only one of them did (a file gone, a conditional read,
    a read for a play only one of them compared). A file neither first read
    for a compared play changed nothing compared.
    """

    def compared(reads: tuple[Read, ...], bound: int | None) -> dict[str, str]:
        return {
            read.path: read.digest
            for read in reads
            if bound is None or read.play < bound
        }

    early_mine, early_theirs = compared(mine, bounds[0]), compared(theirs, bounds[1])
    return tuple(
        sorted(
            path
            for path in early_mine.keys() | early_theirs.keys()
            if early_mine.get(path) != early_theirs.get(path)
        )
    )


def _recorded(
    session: Session, record: Record, instants: list[Fraction], *, frames: bool
) -> dict[Fraction, tuple[State, Image.Image | None]]:
    """Each instant answered from a forward run's record, which evaluated
    and kept every one of them, drawn from it through the session's own
    drawing (:meth:`.Session._drawn`).
    """
    resolved = {t: _route(record, t) for t in instants}
    answered = {t: _state(record, t, resolved[t]) for t in instants}
    if not frames:
        return {t: (answered[t], None) for t in instants}
    kept_static: dict[int, PixelArray | None] = {}

    def static(drawing: Drawing, play: PlayRecord) -> PixelArray | None:
        # Kept for the play's next frame, as a session keeps its own; apart
        # from the session's, which belongs to its record.
        if play.index not in kept_static:
            kept_static[play.index] = drawing.static(play)
        return kept_static[play.index]

    drawn = session._drawn(record, [resolved[t] for t in instants], static)
    camera = record._scene.renderer.camera
    return {
        t: (answered[t], camera.get_image(pixels))
        for t, pixels in zip(instants, drawn, strict=True)
    }


def _rendered(
    forward: _ForwardRecorder,
    record: Record,
    instants: list[Fraction],
    written: dict[Fraction, PixelArray],
    *,
    frames: bool,
) -> dict[Fraction, tuple[State, Image.Image | None]]:
    """Each instant answered as the forward run's render shows it: by the
    last frame at or before it and the state that frame shows, or after the
    script ended, by the scene as it left it. Called inside a session's
    operation, so drawn under the configuration the run executed under,
    its draw path among it, after the run has left it.
    """
    scene = forward.scene
    camera = scene.renderer.camera
    fps = exact(camera.frame_rate)
    final: PixelArray | None = None
    answered: dict[Fraction, tuple[State, Image.Image | None]] = {}
    for t in instants:
        ended = record.play_at(t) is None
        if ended and frames and final is None:
            # As the render draws the scene its record can't: from the live
            # scene, which the script left so.
            scene.renderer._draw_final(scene)
            final = camera.pixel_array.copy()
        shown = t if ended else _on_screen(fps, [t])[0]
        state = _state(record, t, _route(record, shown))
        if not frames:
            answered[t] = (state, None)
            continue
        pixels = final if ended else written[shown]
        answered[t] = (state, camera.get_image(pixels))
    return answered


def _state(record: Record, t: Fraction, resolved: _Resolved) -> State:
    """The forward answer at ``t``: the state ``resolved`` reads, a capture of
    the forward run.
    """
    assert resolved.prepare is None, "a forward run keeps every instant it answers"
    play = resolved.play
    return build_state(
        record,
        t,
        play,
        resolved.capture,
        route=play.route if play is not None else None,
        answer="forward",
        as_of=resolved.as_of,
    )


def _on_screen(fps: Fraction, instants: Iterable[Fraction]) -> list[Fraction]:
    """The frame of the grid at ``fps`` on screen at each instant: the last
    one at or before it.
    """
    return [Fraction(math.floor(t * fps)) / fps for t in instants]


def _keep_frames(
    scene: Scene,
    fps: Fraction,
    wanted: Iterable[Fraction],
    written: dict[Fraction, PixelArray],
) -> None:
    """Keep in ``written``, by instant, the frames of ``wanted`` the scene's
    renderer writes. A render writes every frame of the grid at ``fps``, in
    order, a repeated one once with its count; on the writer instance, as a
    class of the engine's patched would take every play off its route.
    """
    wanted = frozenset(wanted)
    count = 0

    def write_frame(pixels: PixelArray, *, repeat: int = 1) -> None:
        nonlocal count
        for k in range(count, count + repeat):
            if Fraction(k) / fps in wanted:
                written[Fraction(k) / fps] = pixels
        count += repeat

    scene.renderer.file_writer.write_frame = write_frame


def _resize(camera: Any, shape: tuple[int, int]) -> None:
    """Give ``camera`` the pixel size ``(height, width)``, as a session does
    for a quality.
    """
    from arachne.session import resize

    if (camera.pixel_height, camera.pixel_width) != shape:
        resize(camera, *shape)


@contextmanager
def _kept() -> Iterator[None]:
    """Whatever another execution does inside the block to the process's
    origin table, random generators and engine classes, they hold what they
    held before, however the block ends.
    """
    generators = (random.getstate(), np.random.get_state())  # noqa: NPY002
    with ORIGINS.kept(), _classes_kept():
        try:
            yield
        finally:
            random.setstate(generators[0])
            np.random.set_state(generators[1])  # noqa: NPY002


_MISSING = object()


@contextmanager
def _classes_kept() -> Iterator[None]:
    """The engine's classes hold afterwards what they held before: a scene
    sets a default (``Dot.set_default``, which replaces ``__init__``) or
    adds an animation override, and a run that stops or raises never
    reaches the scene's own reset. Each class's attributes are put back,
    those it gained removed, and the dicts, lists and sets they hold given
    back their items.
    """
    saved = []
    for cls in _all_engine_classes():
        members = dict(vars(cls))
        items = {
            key: value.copy()
            for key, value in members.items()
            if type(value) in (dict, list, set)
        }
        saved.append((cls, members, items))
    try:
        yield
    finally:
        for cls, members, items in saved:
            current = vars(cls)
            for key in [key for key in current if key not in members]:
                delattr(cls, key)
            for key, value in members.items():
                if current.get(key, _MISSING) is not value:
                    setattr(cls, key, value)
            for key, copy in items.items():
                _put_back(members[key], copy)


def _put_back(container: Any, copy: Any) -> None:
    """Give a dict, list or set the items of its ``copy`` again, when they
    are not the very items it holds now.
    """
    if isinstance(container, dict):
        if len(container) != len(copy) or any(
            copy.get(key, _MISSING) is not value for key, value in container.items()
        ):
            container.clear()
            container.update(copy)
    elif isinstance(container, list):
        if len(container) != len(copy) or any(
            a is not b for a, b in zip(container, copy, strict=True)
        ):
            container[:] = copy
    elif container != copy:
        container.clear()
        container.update(copy)
