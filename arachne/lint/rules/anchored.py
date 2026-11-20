"""Attachment: every annotation measured at its anchor.

At each instant it reads, the rule shows the capture there and compares,
for every annotation the scene holds that is attached to an anchor
(:mod:`arachne.mobject.annotation.attach`), where it points with where its
anchor is (:meth:`~arachne.assets.Anchor.point`, a function of the anchored
object's state alone): the end of its leader line, or, for one without a
leader that follows its anchor, its body's centre less its offset. A
distance over :data:`ALLOWED` pixels of the 1080-tall reference picture, as
the camera shows the two points then, is ``annotation.detached``; findings
about one annotation at consecutive instants it reads are one finding.

The usual cause is the order of updaters: an annotation added to the scene
before something whose own updater moves its target reads the target a
frame late. An annotation that names nothing (an anchor the asset does not
have, which the lint reports as ``annotation.anchor_unknown``) is not
measured, nor one that neither follows its anchor nor has a leader. Nor is
one whose anchor is on an object the scene does not hold at that instant:
the record keeps no state of it, so no instant can say where it was (the
annotation itself still follows it as it is, and is drawn so).

It reads every instant the lint reads (its sampled frames, or the frames
on screen at the times the lint is asked ``at``:
:attr:`~arachne.lint.frames.Frames.lint_instants`), and with them its own:
the frames the lint would choose at
:data:`~arachne.lint.codes.MOTION_REFERENCE_FPS`, the clock motion is
measured on, so a lint at a low rate still reads an attachment at least
that often. Asked :meth:`~AnchoredRule.where` ``at`` times, it reads the
frames on screen then and no others. A scene with no attached annotation in
its record is not read.
"""

from __future__ import annotations

__all__ = ["ALLOWED", "AnchoredRule", "anchored"]

from collections.abc import Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.camera.projection import pixel_points, record_view, view_of
from arachne.timeline.clock import exact
from arachne.timeline.draw import Drawing

from ..codes import MOTION_REFERENCE_FPS, THRESHOLD_HEIGHT
from ..findings import Finding, FindingObject

if TYPE_CHECKING:
    from arachne.mobject.mobject import Mobject
    from arachne.timeline.record import Capture, PlayRecord, Record
    from arachne.timeline.source import SourceLine

    from ..core import LintContext
    from ..frames import Frames
    from ..instants import LintInstant

#: How far, in pixels of the 1080-tall reference picture, an attachment may
#: point from its anchor.
ALLOWED = 1.0


@dataclass(frozen=True)
class _Miss:
    """An attachment pointing ``distance`` reference pixels from its anchor."""

    thing: FindingObject
    line: SourceLine | None
    distance: float
    anchor: str | None


def holds_attached(record: Record) -> bool:
    """Whether ``record`` holds an annotation attached to an anchor."""
    from arachne.mobject.annotation.attach import Annotation

    return any(
        isinstance(mob, Annotation) and mob.attached is not None
        for mob in record.mobjects()
    )


def _read(
    record: Record,
    t: Fraction,
    play: PlayRecord | None,
    capture: Capture,
    live: bool,
) -> tuple[_Miss, ...]:
    """The attachments at ``capture`` that point further than
    :data:`ALLOWED` from their anchors.
    """
    from arachne.mobject.annotation.attach import Annotation, Attached

    names = play.names if play is not None else record.final_names
    found: list[tuple[Mobject, Attached]] = []
    reached: set[int] = set()
    for mob, state in record._walk(capture, capture.draw_list()):
        reached.add(id(mob))
        attached = state.get("attached")
        if isinstance(mob, Annotation) and isinstance(attached, Attached):
            found.append((mob, attached))
    # An anchor on what the scene does not hold has no state here.
    found = [(m, a) for m, a in found if id(a.anchor.mobject) in reached]
    if not found:
        return ()
    camera = record._scene.renderer.camera
    with Drawing(record) as drawing:
        if live:
            # A forward run holds the capture: read it through the camera's
            # recorded inputs, running none of its callbacks.
            drawing.show_camera(capture)
            timeline = record._scene._arachne_timeline
            view = view_of(
                camera,
                home=timeline.home,
                configured_pixels=timeline.configured_pixels,
            )
        else:
            drawing.show(capture, [*capture.draw_list(), *capture.moving])
            view = record_view(record)
        scale = THRESHOLD_HEIGHT / min(view.pixel_width, view.pixel_height)
        misses: list[_Miss] = []
        for mob, attached in found:
            target = attached.point()
            pointed = _pointed(mob, attached)
            if target is None or pointed is None:
                continue
            ends = pixel_points(camera, view, mob, np.array([pointed, target]))
            distance = float(np.linalg.norm(ends[0] - ends[1])) * scale
            if distance <= ALLOWED:
                continue
            origin = record.origin(mob)
            misses.append(
                _Miss(
                    FindingObject(
                        id=origin.id,
                        source_line=origin.scene_line,
                        kind=type(mob).__name__,
                        label_or_text=names.get(id(mob)),
                        bbox=None,
                        screen_bbox=None,
                        name=names.get(id(mob)),
                    ),
                    origin.source_line,
                    distance,
                    attached.anchor.name,
                )
            )
    return tuple(misses)


def _pointed(mob: Any, attached: Any) -> np.ndarray | None:
    """Where an annotation points: its leader's end, or, when it follows,
    the anchor's point its body's placement says
    (:meth:`~arachne.mobject.annotation.attach.Annotation.points_at`: its
    held point less where that stood from the offset, the reference
    following places it by); ``None`` when it does neither.
    """
    leader = mob.leader
    if leader is not None and len(leader.points):
        return np.array(leader.points[-1])
    if attached.follow and any(len(m.points) for m in mob.body.get_family()):
        return mob.points_at()
    return None


@dataclass(frozen=True)
class AnchoredRule:
    """The rule :data:`anchored`, read at its instants or, from
    :meth:`where`, at the frames on screen at the times ``at``.
    """

    at: tuple[Fraction, ...] | None = None

    def where(self, *, at: Iterable[Fraction | float] | None = None) -> AnchoredRule:
        """This rule read at the frames on screen at the times ``at`` (its
        own instants when ``None``).
        """
        return replace(self, at=None if at is None else tuple(exact(t) for t in at))

    @property
    def _owner(self) -> tuple[str, int]:
        return ("anchored", id(self))

    def _times(
        self, record: Record, lint_instants: Iterable[LintInstant]
    ) -> list[Fraction]:
        """The instants it reads: those it was asked ``at``; else the
        instants the lint reads (``lint_instants``) and its own.
        """
        from ..instants import lint_instants as instants_of

        if self.at is not None:
            return [i.t for i in instants_of(record, at=self.at)]
        own = instants_of(record, MOTION_REFERENCE_FPS)
        return sorted({i.t for i in (*lint_instants, *own)})

    def request_frames(self, frames: Frames) -> None:
        """Ask for the frames it reads: none for a scene with no attached
        annotation in its record.
        """
        record = frames.session.record
        if not holds_attached(record):
            return
        frames.request(self._owner, self._times(record, frames.lint_instants), _read)

    def __call__(self, context: LintContext) -> list[Finding]:
        if not context.frames.answered(self._owner):
            return []
        times = self._times(context.session.record, context.frames.lint_instants)
        read = context.frames.result(self._owner)
        runs: dict[str, list[Any]] = {}
        done: list[Finding] = []
        for step, (t, misses) in enumerate(zip(times, read, strict=True)):
            for miss in misses:
                run = runs.get(miss.thing.id)
                if run is not None and run[1] == step - 1:
                    run[1], run[2] = step, t
                    run[3] = max(run[3], miss.distance)
                    continue
                if run is not None:
                    done.append(_finding(*run))
                runs[miss.thing.id] = [miss, step, t, miss.distance, t]
        done.extend(_finding(*run) for run in runs.values())
        done.sort(key=lambda f: (f.t_start, f.objects[0].id))
        return done


def _finding(
    miss: _Miss, _step: int, end: Fraction, worst: float, start: Fraction
) -> Finding:
    """The finding of a run of misses of one annotation, from ``start`` to
    ``end``, at its ``worst`` distance.
    """
    what = f"its anchor {miss.anchor!r}" if miss.anchor else "its anchor"
    return Finding(
        "annotation.detached",
        start,
        end,
        objects=(miss.thing,),
        message=(
            f"{miss.thing.kind} {miss.thing.id} points up to {worst:.1f} reference"
            f" pixels from {what}, beyond the {ALLOWED:g} it may: it reads its"
            " anchor before what moves the anchor; add it to the scene after"
            " what it points at"
        ),
        line=miss.line,
        detail={"distance": round(worst, 3), "allowed": ALLOWED, "anchor": miss.anchor},
    )


#: The rule: every attachment within a reference pixel of its anchor.
anchored = AnchoredRule()
