"""One attachment for every annotation.

An annotation is a group with a **body** (a callout's pill, a readout's
text, a gauge) and, optionally, a **leader**: a line from the body's edge to
the point the annotation names. It is attached to an
:class:`~arachne.assets.Anchor` with an offset. Its engine updater,
:func:`keep_attached`, which takes no ``dt``, puts the body at the anchor's
point plus the offset and the leader's end on the point; with
``follow=False`` the body stays where it stands and the leader stretches to
the point. :func:`anchor` attaches to a point of any drawn object: of its
box, along its outline, or one of its points by index; a rig's anchors are
its manifest's (:meth:`~arachne.assets.Rig.anchor`).

The updater runs where CE runs a mobject's updaters, after the play's
animations and in the order of the scene's mobjects: an annotation added
before something whose own updater moves its target reads the target as it
was before that updater ran, a frame late. The lint rule
:data:`~arachne.lint.rules.anchored` measures every attachment at its
anchor and reports that lag as ``annotation.detached``. An animation of the
annotation itself suspends its updater for the play, as CE suspends any
mobject's, except :class:`~.readout.CountUp`, which keeps it running.

An anchor that names nothing (an unknown name of a rig, which the lint
reports) has no point: what is attached to it draws nothing. An anchor is
measured by the lint only on an object the scene holds, whose state the
record keeps; an annotation attached to an object the scene does not hold
still follows it, unmeasured.
"""

from __future__ import annotations

__all__ = [
    "Along",
    "Annotation",
    "Attached",
    "BoxPoint",
    "LeaderLine",
    "PointAt",
    "anchor",
    "checked_hold",
    "keep_attached",
    "placed_point",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from arachne.assets.anchor import Anchor
from arachne.constants import ORIGIN, UR
from arachne.mobject.types.vectorized_mobject import VGroup, VMobject
from arachne.mobject.values import FrozenArray, freeze
from arachne.utils.color import WHITE, ParsableManimColor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arachne.mobject.mobject import Mobject

#: The offset an annotation sits at from its anchor by default.
DEFAULT_OFFSET: np.ndarray = UR


@dataclass(frozen=True)
class BoxPoint:
    """The point of a mobject's box in the direction ``direction`` (its
    centre for ``ORIGIN``), as :meth:`~.Mobject.get_critical_point` reads
    it.
    """

    direction: tuple[float, float, float]

    def point(self, mobject: Mobject) -> np.ndarray | None:
        if not _has_points(mobject):
            return None
        return np.array(mobject.get_critical_point(np.array(self.direction)))


@dataclass(frozen=True)
class Along:
    """The point ``proportion`` (0 to 1) of the way along a mobject's own
    outline, as :meth:`~.VMobject.point_from_proportion` reads it.
    """

    proportion: float

    def point(self, mobject: Mobject) -> np.ndarray | None:
        if not len(mobject.points):
            return None
        return np.array(mobject.point_from_proportion(self.proportion))


@dataclass(frozen=True)
class PointAt:
    """A mobject's own point ``index``."""

    index: int

    def point(self, mobject: Mobject) -> np.ndarray | None:
        points = mobject.points
        if not -len(points) <= self.index < len(points):
            return None
        return np.array(points[self.index])


def _has_points(mobject: Mobject) -> bool:
    return any(len(m.points) for m in mobject.get_family())


def anchor(
    mobject: Mobject,
    *,
    at: Sequence[float] | np.ndarray | None = None,
    along: float | None = None,
    index: int | None = None,
) -> Anchor:
    """An anchor on ``mobject``, read from its state as it is whenever the
    anchor is asked: the point of its box in the direction ``at`` (``UR``
    its upper right corner, ``ORIGIN`` its centre), the point ``along``
    (0 to 1) of the way along its own outline, or its own point ``index``.
    With none given, its centre.
    """
    given = [
        name
        for name, v in (("at", at), ("along", along), ("index", index))
        if v is not None
    ]
    if len(given) > 1:
        raise ValueError(
            f"An anchor is attached one way: give one of at, along or index, not"
            f" {' and '.join(given)}."
        )
    attachment: BoxPoint | Along | PointAt
    if along is not None:
        proportion = float(along)
        if not 0 <= proportion <= 1:
            raise ValueError(
                f"along is a proportion of the outline, between 0 and 1, not {along!r}."
            )
        if not len(mobject.points):
            raise ValueError(
                f"A {type(mobject).__name__} with no points of its own has no outline"
                " to be along: attach at a point of its box instead."
            )
        attachment = Along(proportion)
    elif index is not None:
        count = len(mobject.points)
        if not -count <= int(index) < count:
            raise IndexError(
                f"A {type(mobject).__name__} with {count} points has no point {index}."
            )
        attachment = PointAt(int(index))
    else:
        given_at = np.asarray(ORIGIN if at is None else at, dtype=float).ravel()
        direction = np.zeros(3)
        direction[: min(len(given_at), 3)] = given_at[:3]
        x, y, z = (float(c) for c in direction)
        attachment = BoxPoint((x, y, z))
    return Anchor(mobject, attachment)


@dataclass(frozen=True)
class Attached:
    """How an annotation is attached: to ``anchor``, its body at the
    anchor's point plus ``offset`` when it follows.
    """

    anchor: Anchor
    offset: tuple[float, float, float]
    follow: bool

    def point(self) -> np.ndarray | None:
        """The anchor's point now; ``None`` when it names nothing."""
        return self.anchor.point()


class LeaderLine(VMobject):
    """A straight line from ``start`` to ``end``: an annotation's leader,
    which ends on its anchor's point.
    """

    def __init__(
        self,
        start: Sequence[float] | np.ndarray = ORIGIN,
        end: Sequence[float] | np.ndarray = ORIGIN,
        *,
        color: ParsableManimColor = WHITE,
        stroke_width: float = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(color=color, stroke_width=stroke_width, **kwargs)
        self.put_between(start, end)

    def put_between(
        self, start: Sequence[float] | np.ndarray, end: Sequence[float] | np.ndarray
    ) -> None:
        """Run the line from ``start`` to ``end``."""
        self.set_points_as_corners(
            np.array([np.asarray(start, dtype=float), np.asarray(end, dtype=float)])
        )


def keep_attached(mobject: Mobject) -> None:
    """An annotation's own updater: keep it at its anchor
    (:meth:`Annotation.attach`). It takes no ``dt``, so it ticks no wait.
    """
    assert isinstance(mobject, Annotation)
    mobject.attach()


class Annotation(VGroup):
    """A group holding a ``body`` and, with ``leader``, a
    :class:`LeaderLine` drawn below it, attached to the anchor ``at``
    (``None``: placed by hand, and no leader) at ``offset``, following it
    unless ``follow`` is false.

    ``hold`` names the point of the body that following keeps in place, as
    one of the nine points of its box (:func:`checked_hold`): ``ORIGIN``, its
    centre, by default; ``LEFT``, the middle of its left edge; ``UR``, its
    upper right corner. The body is first put with its centre at the
    anchor's point plus the offset; from then on, whatever its size does,
    following keeps :meth:`held_point` where it stood then from the anchor's
    point plus the offset. With ``follow`` false the body is placed so once
    and stays. :meth:`points_at` reads the anchor's point back from the
    body, as the lint rule ``anchored`` does.
    """

    def __init__(
        self,
        body: VMobject,
        *,
        at: Anchor | None = None,
        offset: Sequence[float] | np.ndarray = DEFAULT_OFFSET,
        follow: bool = True,
        leader: bool = False,
        leader_color: ParsableManimColor = WHITE,
        hold: Sequence[float] | np.ndarray = ORIGIN,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if at is not None and not isinstance(at, Anchor):
            raise TypeError(
                f"at is an Anchor (anchor(mobject, ...) or a rig's anchor(name)),"
                f" not a {type(at).__name__}"
            )
        #: What the annotation shows, placed at the anchor.
        self.body = body
        #: The line from the body to the anchor, or ``None``.
        self.leader: LeaderLine | None = (
            LeaderLine(color=leader_color) if leader and at is not None else None
        )
        if self.leader is not None:
            self.add(self.leader)
        self.add(body)
        #: The point of the body's box that following keeps in place: a
        #: read-only array, so that it is part of the content key.
        self.hold = checked_hold(hold)
        #: Where the held point stands from the anchor's point plus the
        #: offset, read-only for the same reason.
        self._held_from_place: FrozenArray = freeze(np.zeros(3), FrozenArray)
        #: Where the held point stood from the body's centre when placement
        #: last measured it: what the lint reads, from the record alone,
        #: in place of calling :meth:`held_point`.
        self._held_from_centre: FrozenArray = freeze(np.zeros(3), FrozenArray)
        #: How the annotation is attached, or ``None`` when it is not.
        self.attached: Attached | None = None
        if at is None:
            return
        place = np.zeros(3)
        given = np.asarray(offset, dtype=float).ravel()
        place[: min(len(given), 3)] = given[:3]
        self.attached = Attached(
            at, (float(place[0]), float(place[1]), float(place[2])), bool(follow)
        )
        point = at.point()
        if point is None:
            # Attached to nothing: it draws nothing.
            for member in self.get_family():
                member.points = np.zeros((0, self.dim))
            return
        # Placed at its anchor first, whether it follows or stays there.
        body.move_to(point + place)
        held = self.held_point(body)
        self._held_from_place = freeze(held - (point + place), FrozenArray)
        self.note_held(body, held)
        if self.attached.follow or self.leader is not None:
            self.add_updater(keep_attached)
        self.attach()

    def held_point(self, body: Mobject) -> np.ndarray:
        """Where ``body`` (this annotation's body, or one about to replace
        it) puts the point that following holds: here, the point
        :attr:`hold` of its box. A subclass that holds another point
        overrides it with this signature. It is called while
        :class:`Annotation` is constructed, once the body is first placed,
        so what an override reads must be set before ``Annotation.__init__``
        runs; then at every :meth:`attach` that follows. It runs only while
        the scene runs: the lint never calls it.
        """
        return np.array(body.get_critical_point(self.hold), dtype=float)

    def note_held(self, body: Mobject, held: np.ndarray) -> None:
        """Record that ``held`` is where ``body`` (the body as it now
        stands, or a replacement about to become it) puts the held point,
        as where it stands from the body's centre: what :meth:`points_at`
        and the lint read. Placement calls it each time it measures; a
        subclass that replaces its body calls it for the replacement.
        """
        self._held_from_centre = freeze(
            np.asarray(held, dtype=float) - body.get_center(), FrozenArray
        )

    def points_at(self) -> np.ndarray | None:
        """The anchor's point that the body's placement says
        (:func:`placed_point`); ``None`` when it is not attached.
        """
        return placed_point(self)

    def attach(self) -> None:
        """Put the body at the anchor's point plus the offset, when it
        follows (its :attr:`hold` point where it stood from there), and the
        leader from the body's edge to the point.
        """
        attached = self.attached
        if attached is None:
            return
        point = attached.point()
        if point is None:
            return
        if attached.follow:
            place = point + np.array(attached.offset) + self._held_from_place
            held = self.held_point(self.body)
            self.note_held(self.body, held)
            self.body.shift(place - held)
        if self.leader is not None:
            self.leader.put_between(_edge_toward(self.body, point), point)


def placed_point(annotation: Annotation) -> np.ndarray | None:
    """The anchor's point that ``annotation``'s placement says: its body's
    centre plus where the held point stood from it when placement last
    measured, less where the held point stands from the anchor's point plus
    the offset, less the offset; ``None`` when it is not attached. It reads
    the annotation's fields and its body's points alone, never a method a
    subclass overrides, so it answers from a capture as from the live
    object; following puts the body where it is the anchor's point.
    """
    attached = annotation.attached
    if attached is None:
        return None
    return (
        annotation.body.get_center()
        + np.asarray(annotation._held_from_centre)
        - np.asarray(annotation._held_from_place)
        - np.array(attached.offset)
    )


def checked_hold(given: Sequence[float] | np.ndarray) -> FrozenArray:
    """``given`` as a read-only point of a box, when it is one: two or three
    numbers, the first two each -1, 0 or 1 (``LEFT``, ``ORIGIN``, ``UR``,
    ...) and the third, if given, 0; :class:`ValueError` naming it
    otherwise.
    """
    try:
        # Complex input would lose its imaginary part to the conversion:
        # refused before it.
        raw = np.asarray(given)
        flat = np.full(1, np.nan) if raw.dtype.kind == "c" else raw.astype(float)
    except (TypeError, ValueError):
        flat = np.full(1, np.nan)
    if (
        flat.ndim != 1
        or len(flat) not in (2, 3)
        or not np.isin(flat[:2], (-1.0, 0.0, 1.0)).all()
        or (len(flat) == 3 and flat[2] != 0)
    ):
        raise ValueError(
            "hold is a point of a box: two or three numbers, each of the first"
            " two -1, 0 or 1 and the third 0, as LEFT, ORIGIN or UR; not"
            f" {given!r}."
        )
    return freeze(np.array([flat[0], flat[1], 0.0]) + 0.0, FrozenArray)


def _edge_toward(body: Mobject, point: np.ndarray) -> np.ndarray:
    """Where the line from ``body``'s centre to ``point`` leaves its box;
    ``point`` itself when it lies inside the box.
    """
    centre = body.get_center()
    half = (body.get_critical_point(UR) - body.get_critical_point(-UR)) / 2
    away = point - centre
    scales = [abs(half[i] / away[i]) for i in (0, 1) if abs(away[i]) > 1e-12]
    if not scales or min(scales) >= 1:
        return np.array(point, dtype=float)
    return centre + away * min(scales)
