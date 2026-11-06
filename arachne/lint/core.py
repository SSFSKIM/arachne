"""The lint over a session.

:func:`lint` reads the session's record at its lint instants
(:mod:`.instants`): the parts drawn at each (:mod:`.parts`), which its rules
measure. A record that refuses drawing, or an instant it can't answer, is
read through one forward run of the scene on the render's grid, whose live
scene is read at each instant it answers: so is every instant the record
kept no capture of, rather than evaluated there by the lint, and every
instant of a record kept on another grid than the render's, whose frame
hooks and dt updaters ran at other instants than the render's. The session's divergences become ``replay.diverged``
findings (a probe's a note, an execution's an error), and a stale session a
``source.changed`` note. The caller's time range and exemptions then leave
findings out, look-alikes fold, and the result is data: nothing the lint
finds ends the process.
"""

from __future__ import annotations

__all__ = [
    "RULES",
    "LintContext",
    "LintResult",
    "Rule",
    "Span",
    "lint",
    "parts_reader",
]

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne.timeline.clock import exact

from .bridge import Checks, terms_at
from .exempt import Exemptions
from .findings import Finding, fold, in_range
from .frames import FrameReader, Frames
from .instants import LintInstant, lint_instants
from .parts import Parts, build_parts

if TYPE_CHECKING:
    from arachne.session import Session
    from arachne.timeline.execution import Divergence
    from arachne.timeline.record import Capture, PlayRecord, Record

    from .rules import Terms


@dataclass(frozen=True, eq=False)
class LintContext:
    """What a rule reads: the session, its record, the lint rate, the lint
    instants in time order, the parts drawn at each, and the lint's frames
    (:class:`.Frames`), where a rule finds the frames it asked for.
    """

    session: Session
    record: Record
    fps: Fraction
    instants: tuple[LintInstant, ...]
    parts: tuple[Parts, ...]
    frames: Frames

    def terms_at(self, t: Fraction, base: Terms) -> Terms:
        """``base`` under the terms the scene declared at or before ``t``
        (:func:`.bridge.terms_at`), in the execution the frames come from.
        """
        return terms_at(self.frames.execution, t, base)


#: A rule: the findings it makes of a lint's context. A rule that reads
#: frames of its own also has ``request_frames(frames)``, which the lint
#: calls before any frame is read, so that one execution of the scene serves
#: the parts and every rule (:mod:`.frames`); it then finds them in
#: ``context.frames.result(...)`` under the owner it asked as.
Rule = Callable[[LintContext], Iterable[Finding]]

#: Who the parts read at the lint instants are asked for by.
_PARTS = "parts"

#: The rules a lint runs unless told others.
RULES: tuple[Rule, ...] = ()


@dataclass(frozen=True)
class Span:
    """A play of the timeline, as a report lists it."""

    index: int
    kind: str
    t_start: Fraction
    t_end: Fraction


@dataclass(frozen=True, eq=False)
class LintResult:
    """What a lint found, in time order, look-alikes folded; the instants it
    read; the session's timeline; whether the record is stale; how long the
    lint took, in seconds; and the resolution, ``(width, height)``, its
    findings' pixel boxes are at: the session's.
    """

    findings: tuple[Finding, ...]
    instants: tuple[LintInstant, ...]
    timeline: tuple[Span, ...]
    stale: bool
    duration_s: float
    target_resolution: tuple[int, int]

    @property
    def passed(self) -> bool:
        """Whether no finding is an error."""
        return not any(finding.severity == "error" for finding in self.findings)


def lint(
    session: Session,
    *,
    fps: Fraction | float = 4,
    t_range: Sequence[float] | None = None,
    exempt: Iterable[Any] = (),
    at: str | Iterable[Fraction | float] | None = None,
    rules: Iterable[Rule] = RULES,
) -> LintResult:
    """Lint ``session``: its ``rules`` read at its lint instants, at ``fps``
    (``at`` ``"held+sampled"``, the default) or at the frames on screen at
    the times ``at`` lists; then its divergences and staleness. Findings
    that hold at no time in ``t_range`` are left out, and so are those the
    selectors and pairs of ``exempt`` exempt (:mod:`.exempt`).
    """
    started = time.perf_counter()
    exemptions = Exemptions.parse(exempt)
    record = session.record
    instants = lint_instants(record, fps, at=at)
    rules = tuple(rules)
    frames = Frames(session)
    _request_parts(frames, instants)
    # The scene's own checks (:mod:`.bridge`), whatever the rules.
    checks = Checks(rules)
    checks.request_frames(frames)
    for rule in rules:
        request = getattr(rule, "request_frames", None)
        if request is not None:
            request(frames)
    frames.read()
    parts = tuple(frames.result(_PARTS))
    context = LintContext(session, record, exact(fps), instants, parts, frames)
    found = [finding for rule in rules for finding in rule(context)]
    found.extend(checks(context))
    # After the parts: a forward run that read them may have found one.
    found.extend(_diverged(session))
    if session.stale:
        found.append(_stale(session))
    kept = [f for f in found if in_range(f, t_range) and not exemptions.exempts(f)]
    kept.sort(key=lambda f: (f.t_start is None, f.t_start or 0))
    timeline = tuple(
        Span(play.index, play.kind, play.start, play.end) for play in record.plays
    )
    return LintResult(
        tuple(fold(kept, timeline=[span.t_start for span in timeline])),
        instants,
        timeline,
        session.stale,
        time.perf_counter() - started,
        target_resolution(session),
    )


def target_resolution(session: Session) -> tuple[int, int]:
    """The pixel size, ``(width, height)``, the session's configuration
    renders at: what a finding's pixel box is given in.
    """
    height, width = session._executed_shape
    return int(width), int(height)


def _request_parts(frames: Frames, instants: tuple[LintInstant, ...]) -> None:
    """Ask for the parts drawn at each instant: from the record, and what it
    can't answer from the lint's forward run, read from its live scene.
    """
    frames.request(
        _PARTS, [instant.t for instant in instants], parts_reader(frames.session)
    )


def parts_reader(session: Session) -> FrameReader[Parts]:
    """The reader of the parts a frame of ``session`` draws, as the lint reads
    them (:mod:`.frames`): from the record, or a forward run's live scene.
    """
    target = target_resolution(session)

    def read(
        record: Record,
        t: Fraction,
        play: PlayRecord | None,
        capture: Capture,
        live: bool,
    ) -> Parts:
        return build_parts(
            record,
            t,
            play,
            capture,
            answer="forward" if live else "recorded",
            target=target,
            live=live,
        )

    return read


def _diverged(session: Session) -> list[Finding]:
    """The session's divergences, as ``replay.diverged`` findings over the
    play they name, or the whole scene.
    """
    return [divergence_finding(session.record, d) for d in session.divergences]


def divergence_finding(record: Record, divergence: Divergence) -> Finding:
    """``divergence`` as a ``replay.diverged`` finding over the play of
    ``record`` it names, or the whole scene: a probe's a note, an
    execution's an error.
    """
    play = (
        record.plays[divergence.play]
        if divergence.play is not None and divergence.play < len(record.plays)
        else None
    )
    start, end = (play.start, play.end) if play else (Fraction(0), record.duration)
    where = f"play {divergence.play}" if divergence.play is not None else "the scene"
    what = (
        "the probe ticked a play whose restore differed from its record"
        if divergence.kind == "probe"
        else "two executions of this source differ"
    )
    return Finding(
        divergence.code,
        start,
        end,
        message=f"{what}, at {where}: {divergence.reason}",
        line=divergence.source_line,
        severity="note" if divergence.kind == "probe" else "error",
        detail={
            "kind": divergence.kind,
            "play": divergence.play,
            "reason": divergence.reason,
            "changed_inputs": list(divergence.changed_inputs),
        },
    )


def _stale(session: Session) -> Finding:
    """The note that the session's record is of a source since changed."""
    from arachne.timeline.fallback import _source_changed

    record = session.record
    message = (
        "the scene's source changed since the record was made: the record is stale"
    )
    if session._source is not None:
        message = _source_changed(session._source, record).detail
    return Finding("source.changed", Fraction(0), record.duration, message=message)
