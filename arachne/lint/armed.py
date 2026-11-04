"""A rule a scene arms: it watches the scene from an instant on, under
the terms the scene declared there.

The hold watch (:mod:`.watch`) and the departure sweep (:mod:`.departures`)
read the whole timeline rather than single instants, so they are armed
rather than asked at times::

    lint(session, rules=(hold_watch.where(since=0, text_gap=0.3),))

Nothing arms them unasked. A rule that samples frames of its own says
which (``reads``); they are read with the lint's other frames, from the
record or the lint's one forward run (:mod:`.frames`).
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
    reads: Callable[[Record, Fraction], Sequence[Fraction]] | None = None

    def where(
        self,
        *,
        since: Fraction | float = 0,
        pad: float = DEFAULT_PAD,
        text_gap: float | None = None,
        exempt: Iterable[Any] = (),
    ) -> ArmedRule:
        """This rule armed at ``since`` under the terms given."""
        terms = Terms(pad, text_gap, Exemptions.parse(exempt), mid_scene=True)
        return replace(self, since=exact(since), terms=terms)

    def request_frames(self, frames: Frames) -> None:
        """Ask the lint for the frames this rule samples, before any is read."""
        if self.reads is None:
            return
        from .core import target_resolution
        from .parts import build_parts

        target = target_resolution(frames.session)

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

        frames.request(self._owner, self.reads(frames.session.record, self.since), read)

    def frames(self, context: LintContext) -> list[Parts]:
        """The parts of each frame :attr:`reads` named, in its order."""
        return context.frames.result(self._owner)

    @property
    def _owner(self) -> tuple[str, int]:
        return ("armed", id(self))

    def __call__(self, context: LintContext) -> list[Finding]:
        return list(self.watch(context, self))
