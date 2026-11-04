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
scene as each play ended, finished and cleaned up, and as the script
removed from it; it asks that the execution keep them.

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

from .exempt import Exemptions
from .rules._rule import DEFAULT_PAD, Terms

if TYPE_CHECKING:
    from arachne.timeline.record import Capture, PlayRecord, Record

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
        """This rule armed at ``since`` under the terms given."""
        terms = Terms(pad, text_gap, Exemptions.parse(exempt), mid_scene=True)
        return replace(
            self,
            since=exact(since),
            terms=terms,
            sample_fps=exact(sample_fps) if sample_fps is not None else None,
        )

    def request_frames(self, frames: Frames) -> None:
        """Ask the lint for the frames this rule samples, before any is read,
        and that its execution keep its states, to the scene's end.
        """
        frames.require_states()
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


def _target(frames: Frames) -> tuple[int, int]:
    from .core import target_resolution

    return target_resolution(frames.session)
