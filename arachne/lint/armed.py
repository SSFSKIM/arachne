"""A rule a scene arms: it watches the scene from an instant on, under
the terms the scene declared there.

The hold watch (:mod:`.watch`) and the departure sweep (:mod:`.departures`)
read the whole timeline rather than single instants, so they are armed
rather than asked at times::

    lint(session, rules=(hold_watch.where(since=0, text_gap=0.3),))

Nothing arms them unasked. A rule that samples frames of its own says
which (``reads``); they are read with the lint's other frames, from the
record or the lint's one forward run (:mod:`.frames`). Between frames an
armed rule reads the states its execution keeps (``Record.states``): the
scene as the script called each play, as each play ended (finished and
cleaned up) and as the script removed from it; it asks that the execution
keep every one of them from its arming on.

The terms a rule is held to at an instant are those of the scene's arming
in force there (:func:`.bridge.arm`, as the scene armed it last by then):
its ``pad`` and ``text_gap``, and its exemptions as the value it was
handed stands there, beside the rule's own; before any arming, the rule's
own terms under what the scene declared by then (:func:`.bridge.terms_at`).

``sample_fps`` is the rate a hold watch samples a play's frames at when
another engine's pass is matched: that pass reads the first frame of the
render's grid at or after each ``1/sample_fps`` of a play, and its last; by
default every frame of the grid is read. Durations are measured on the
engine's clock either way.
"""

from __future__ import annotations

__all__ = ["ArmedRule"]

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne.timeline.clock import exact

from .bridge import armed_entries, armed_terms, exemptions, terms_at
from .exempt import Exemptions
from .rules._rule import DEFAULT_PAD, Terms

if TYPE_CHECKING:
    from arachne.timeline.record import Capture, PlayRecord, Record, States

    from .core import LintContext
    from .findings import Finding
    from .frames import Frames
    from .parts import Parts


@dataclass(frozen=True)
class ArmedRule:
    """A :data:`~arachne.lint.core.Rule` armed at ``since`` (the scene's
    start by default) under ``terms``: ``pad``, ``text_gap`` and the
    exemptions the scene declared, read as one declaration for the whole
    scene. ``watch`` makes its findings; ``reads``, when given, names the
    frames it samples (render-grid instants, or the scene's end), whose
    parts it finds in :meth:`frames`.
    """

    watch: Callable[[LintContext, ArmedRule], Iterable[Finding]]
    since: Fraction = Fraction(0)
    terms: Terms = field(default_factory=lambda: Terms(mid_scene=True))
    reads: Callable[[Record, ArmedRule], Sequence[Fraction]] | None = None
    sample_fps: Fraction | None = None

    def where(
        self,
        *,
        since: Fraction | float = 0,
        pad: float = DEFAULT_PAD,
        text_gap: float | None = None,
        exempt: Iterable[Any] = (),
        sample_fps: Fraction | float | None = None,
    ) -> ArmedRule:
        """This rule armed at ``since`` under the terms given. Raises
        :class:`ValueError` for a ``sample_fps`` that is not positive.
        """
        rate = exact(sample_fps) if sample_fps is not None else None
        if rate is not None and rate <= 0:
            raise ValueError(f"sample_fps is a rate above 0, not {sample_fps!r}")
        terms = Terms(pad, text_gap, Exemptions.parse(exempt), mid_scene=True)
        return replace(self, since=exact(since), terms=terms, sample_fps=rate)

    def terms_at(
        self,
        record: Record,
        t: Fraction,
        play: int | None = None,
        seq: int | Fraction | None = None,
    ) -> Terms:
        """The terms ``record``'s scene is held to at ``t``, an instant of
        ``play`` when given (as the script called it, while it ran, or as it
        ended; a call the script makes after it holds from the next), and
        with ``seq`` at the state of that seq the record keeps (what the
        script does after it, at the same instant, holds from then on): its
        arming in force there, its exemptions beside the rule's own; else
        the rule's own under what the scene declared by then.
        """
        arming = armed_terms(record, t, play=play, seq=seq)
        if arming is None:
            return terms_at(record, t, self.terms, play=play, seq=seq)
        return replace(
            self.terms,
            pad=arming.pad,
            text_gap=arming.text_gap,
            exempt=_joined(self.terms.exempt, arming.exempt),
        )

    def exemptions(
        self,
        record: Record,
        checked: Iterable[tuple[Fraction, int, int | Fraction]] = (),
    ) -> Exemptions:
        """What a check of ``record``'s whole scene is held to: the rule's
        own exemptions, and every one the scene stated (declared, armed or
        swept with), each read at the scene's end (:func:`.bridge.exemptions`);
        and for each state ``checked`` (a time, within a play, at a seq)
        where a check of the scene was held to the scene's arming, the
        entries the arming's value held there, read at the end
        (:func:`.bridge.armed_entries`).
        """
        kept = _joined(self.terms.exempt, exemptions(record))
        for t, play, seq in checked:
            kept = _joined(kept, armed_entries(record, t, play=play, seq=seq))
        return kept

    def request_frames(self, frames: Frames) -> None:
        """Ask the lint for the frames this rule samples, before any is read,
        and that its execution keep its states, to the scene's end.
        """
        frames.require_states(self.since)
        frames.complete(self._owner)
        if self.reads is None:
            return
        target = _target(frames)

        def read(
            record: Record,
            t: Fraction,
            play: PlayRecord | None,
            capture: Capture,
            live: bool,
        ) -> Parts:
            from .parts import build_parts

            return build_parts(
                record,
                t,
                play,
                capture,
                answer="forward" if live else "recorded",
                target=target,
                live=live,
            )

        frames.request(self._owner, self.reads(frames.session.record, self), read)

    def frames(self, context: LintContext) -> dict[Fraction, Parts]:
        """The parts of each frame :attr:`reads` named, by its time."""
        times = self.reads(context.record, self) if self.reads is not None else ()
        return dict(zip(times, context.frames.result(self._owner), strict=True))

    @staticmethod
    def states(context: LintContext) -> States:
        """The states of the execution the frames come from, which keeps
        every one from the rule's arming on (:meth:`.Frames.require_states`).
        """
        states = context.frames.execution.states
        assert states is not None, "an execution read without its states"
        return states

    def state(self, context: LintContext, capture: Capture) -> Parts:
        """The parts of a state the execution keeps (``Record.states``)."""
        from .parts import build_parts

        execution = context.frames.execution
        answer = "recorded" if execution is context.record else "forward"
        return build_parts(
            execution,
            capture.instant,
            None,
            capture,
            answer=answer,
            target=_target(context.frames),
        )

    @property
    def _owner(self) -> tuple[str, int]:
        return ("armed", id(self))

    def __call__(self, context: LintContext) -> list[Finding]:
        return list(self.watch(context, self))


def _joined(first: Exemptions, second: Exemptions) -> Exemptions:
    """The exemptions of both, each once."""
    return Exemptions(
        tuple(dict.fromkeys((*first.bare, *second.bare))),
        tuple(dict.fromkeys((*first.pairs, *second.pairs))),
    )


def _target(frames: Frames) -> tuple[int, int]:
    from .core import target_resolution

    return target_resolution(frames.session)
