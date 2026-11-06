"""A rule asked of single instants: its terms, the instants it is asked at,
and how its findings at consecutive instants become one finding.
"""

from __future__ import annotations

__all__ = ["DEFAULT_PAD", "InstantRule", "Terms"]

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from arachne.timeline.clock import exact

from ..exempt import Exemptions
from ..findings import Finding

if TYPE_CHECKING:
    from arachne.timeline.source import SourceLine

    from ..core import LintContext
    from ..frames import Frames
    from ..parts import Parts

#: The clearance two things that must not touch keep, in home-frame units.
DEFAULT_PAD = 0.05


@dataclass(frozen=True)
class Terms:
    """What a sweep is held to: ``pad``, the clearance things that must not
    touch keep; ``text_gap``, the clearance two texts keep, asked only when
    declared; ``exempt``, the scene's exemptions (selectors, and pairs of
    them; :mod:`arachne.lint.exempt`); and ``mid_scene``, whether the frame
    is one the scene moves on from, where a text laid on a filled shape or
    over a stroke inside its own halo reads clean.
    """

    pad: float = DEFAULT_PAD
    text_gap: float | None = None
    exempt: Exemptions = field(default_factory=Exemptions)
    mid_scene: bool = False


#: A check of one instant's parts under some terms.
Check = Callable[["Parts", Terms], Iterable[Finding]]


@dataclass(frozen=True)
class InstantRule:
    """A :data:`~arachne.lint.core.Rule` made of a check of single
    instants.

    Called with a lint's context it checks the parts at each time ``at``
    names, each read at the frame of the render on screen then (the last at
    or before it, or the scene's end past it), as a query of that time reads
    it, whatever instants the lint samples; or at every lint instant when
    ``at`` is ``None``. Findings
    about the same objects with the same code at consecutive instants it
    checks are one finding, from the first of them to the last. ``line`` is
    the scene line its findings name: where the scene asked for it. At each
    instant its terms are those the scene declared by then
    (:meth:`.LintContext.terms_at`): the declared ``pad`` and ``text_gap``
    in place of its own, the declared exemptions beside its own.
    """

    check: Check
    at: tuple[Fraction, ...] | None = None
    terms: Terms = field(default_factory=Terms)
    line: SourceLine | None = None

    def where(
        self,
        *,
        at: Iterable[Fraction | float] | None = None,
        pad: float = DEFAULT_PAD,
        text_gap: float | None = None,
        exempt: Iterable[Any] = (),
        mid_scene: bool = False,
        line: SourceLine | None = None,
    ) -> InstantRule:
        """This rule asked at the times ``at`` (every instant when
        ``None``), under the terms given.
        """
        return replace(
            self,
            at=None if at is None else tuple(exact(t) for t in at),
            terms=Terms(pad, text_gap, Exemptions.parse(exempt), mid_scene),
            line=line,
        )

    def request_frames(self, frames: Frames) -> None:
        """Ask for the frames on screen at the times ``at`` names, through the
        lint's one source of frames (nothing when ``at`` is ``None``: the
        lint's own instants serve).
        """
        if self.at is None:
            return
        from ..core import parts_reader
        from ..instants import lint_instants

        times = [i.t for i in lint_instants(frames.session.record, at=self.at)]
        frames.request(self._owner, times, parts_reader(frames.session, frames.parts))

    @property
    def _owner(self) -> tuple[str, int]:
        return ("instant rule", id(self))

    def __call__(self, context: LintContext) -> list[Finding]:
        chosen = (
            list(context.parts)
            if self.at is None
            else context.frames.result(self._owner)
        )
        runs: dict[tuple[Any, ...], list[Any]] = {}
        done: list[Finding] = []
        for step, parts in enumerate(chosen):
            # The terms the scene declared by then hold beside the rule's own.
            terms = context.terms_at(parts.t, self.terms)
            for finding in self.check(parts, terms):
                if self.line is not None:
                    finding = replace(finding, line=self.line)
                key = (finding.code, tuple(o.id for o in finding.objects))
                run = runs.get(key)
                if run is not None and run[1] == step - 1:
                    run[1] = step
                    run[2] = parts.t
                    continue
                if run is not None:
                    done.append(replace(run[0], t_end=run[2]))
                runs[key] = [finding, step, parts.t]
        done.extend(replace(first, t_end=end) for first, _, end in runs.values())
        done.sort(key=lambda f: (f.t_start, f.code))
        return done
