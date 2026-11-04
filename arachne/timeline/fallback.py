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

The answer is labelled ``"forward"`` and names why the record could not
answer (:class:`Fallback`). It is never an answer about the record: the new
execution is compared with the record's over what it ran
(:func:`.execution.compare`), and a difference is kept on the session as a
``replay.diverged`` divergence of kind ``execution`` naming the input files
whose digests differ.

A forward run leaves the session as it found it: its record and answers,
its scene's module and the process's origin table and random generators.
The modules the scene imports beside it stay those already imported, so a
helper edited since the session opened runs as it was.
"""

from __future__ import annotations

__all__ = ["Answer", "Fallback", "ForwardRun", "answer_instants"]

import math
import random
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .clock import exact
from .draw import Drawing, LiveFrames
from .execution import Divergence, Execution, compare, summarize
from .origins import ORIGINS
from .recorder import Recorder
from .restore import Unrestorable
from .routes import RecordRefused, _draw, _last_frame, _Resolved, _route
from .state import build_state

if TYPE_CHECKING:
    from PIL import Image

    from arachne.scene.scene import Scene
    from arachne.session import Session
    from arachne.typing import PixelArray

    from .record import PlayRecord, Record
    from .state import State


@dataclass(frozen=True)
class Fallback:
    """Why the record could not answer an instant, so that a forward run did:
    ``kind`` is ``"refused"`` (the record draws no frame,
    :class:`.RecordRefused`) or ``"unrestorable"``
    (:class:`.restore.Unrestorable`); ``reason`` is the stable code
    (``camera.mapping_func``, ``restore.global_rebound``, …) and ``detail``
    the message.
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
    its end), where it differs from the record's execution if it does, and
    up to which instant the two were compared (``None``: all of both).
    """

    instants: tuple[Fraction, ...]
    plays: int
    divergence: Divergence | None
    compared_until: Fraction | None


def answer_instants(
    session: Session,
    instants: Iterable[Fraction | float],
    quality: str | None,
    *,
    frames: bool,
) -> list[Answer]:
    """:meth:`.Session.answers`: the record's answers, and one forward run's
    for every instant the record can't answer.
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
        answered = _forward(session, sorted(missing), frames=frames)
        for t, (state, frame) in answered.items():
            found[t] = Answer(t, state, frame, missing[t])
    return [found[t] for t in asked]


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
    execution draws them.
    """

    def __init__(
        self,
        scene: Scene,
        *,
        grid: Fraction | None,
        draws: bool,
        stepping: bool | Literal["brute_force"],
        instants: Iterable[Fraction],
        stop_after: Fraction,
        known: Mapping[int, str],
    ) -> None:
        super().__init__(
            scene, grid=grid, draws=draws, stepping=stepping, instants=instants
        )
        if known:
            self.execution = Execution(known=dict(known))
        self.stop_after = stop_after
        #: Where the first play drawn as it ran began, once one has.
        self.live_from: Fraction | None = None

    def refusal(self) -> str | None:
        # The renderer asks this first as each play begins.
        if self.scene.time > self.stop_after:
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
            self.live_from = self.scene.time
        played = super().play(skipped=skipped, cache_key=cache_key, live=live)
        if self.draws and live is None and played.draw_refusal is not None:
            raise _Unforeseen(played.draw_refusal)
        return played

    def stopped(self) -> None:
        """The run stopped before its script ended: its identity is still
        that of what it ran (its source, the files it read so far).
        """
        record = self.ensure_record()
        record.identity = self._identity(record)


def _forward(
    session: Session, instants: list[Fraction], *, frames: bool
) -> dict[Fraction, tuple[State, Image.Image | None]]:
    """Execute the session's scene again for ``instants`` (in time order),
    keep the run in :attr:`.Session.forward_runs` and any divergence in
    :attr:`.Session.divergences`, and answer each instant from it.
    """
    from arachne.session import execute

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
        if frames:
            _resize(forward_camera, shape)
        if refused:
            fps = exact(forward_camera.frame_rate)
            grid: Fraction | None = fps
            extra: frozenset[Fraction] = frozenset()
            if frames:
                _keep_frames(scene, fps, _on_screen(fps, instants), written)
        else:
            grid = record.grid
            extra = frozenset(instants) | {_last_frame(record, t) for t in instants}
        forward = _ForwardRecorder(
            scene,
            grid=grid,
            draws=refused and frames,
            stepping=source.stepping,
            instants=extra,
            stop_after=instants[-1],
            known={
                play.index: play.draw_refusal
                for play in record.plays
                if play.draw_refusal is not None
            },
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
        # What a play drawn as it runs draws may be written into the scene (a
        # zoomed display's picture), which the record's execution, drawing
        # nothing, never holds: the two are compared before the first.
        ends = [
            end
            for end in (None if whole else instants[-1], forward.live_from)
            if end is not None
        ]
        until = min(ends) if ends else None
        divergence = compare(session.summary(), summarize(executed), until=until)
        if divergence is not None and not whole:
            # A file only the record's execution read, the forward run having
            # stopped before it, is no difference between them.
            read = {path for path, _ in executed.identity.input_digests}
            divergence = replace(
                divergence,
                changed_inputs=tuple(
                    path for path in divergence.changed_inputs if path in read
                ),
            )
        session.forward_runs.append(
            ForwardRun(tuple(instants), len(executed.plays), divergence, until)
        )
        if divergence is not None:
            session.divergences.append(divergence)
        if refused:
            return _rendered(forward, executed, instants, written, frames=frames)
        if executed.refusal is not None:
            raise RecordRefused(
                f"{executed.refusal}; the forward run's record refuses drawing where"
                " the record's execution did not"
            )
        return _recorded(executed, instants, frames=frames)


def _recorded(
    record: Record, instants: list[Fraction], *, frames: bool
) -> dict[Fraction, tuple[State, Image.Image | None]]:
    """Each instant answered from a forward run's record, which evaluated
    and kept every one of them, drawn from it as a session draws.
    """
    resolved = {t: _route(record, t) for t in instants}
    answered = {t: _state(record, t, resolved[t]) for t in instants}
    if not frames:
        return {t: (answered[t], None) for t in instants}
    camera = record._scene.renderer.camera
    pictures: dict[Fraction, Image.Image] = {}
    kept_static: dict[int, PixelArray | None] = {}
    with Drawing(record) as drawing:

        def static(play: PlayRecord) -> PixelArray | None:
            # Kept for the play's next frame, as a session keeps it.
            if play.index not in kept_static:
                kept_static[play.index] = drawing.static(play)
            return kept_static[play.index]

        for t in instants:
            pixels = _draw(drawing, record, resolved[t], static)
            pictures[t] = camera.get_image(pixels)
    return {t: (answered[t], pictures[t]) for t in instants}


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
    script ended, by the scene as it left it.
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
    origin table and random generators, they hold what they held before.
    """
    generators = (random.getstate(), np.random.get_state())  # noqa: NPY002
    with ORIGINS.kept():
        try:
            yield
        finally:
            random.setstate(generators[0])
            np.random.set_state(generators[1])  # noqa: NPY002
