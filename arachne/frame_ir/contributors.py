"""What each pixel of a frame is made of: the drawn leaves that contribute to
it, in the order they were composited.

A leaf is one drawn thing of a frame's XF stream, ``(object, part)``: a
record's object and its order entry's part (a face of a face group), an
``instances`` record's member reporting its own object. Both painters report
leaves the same way, so the two reports of one frame can be set side by side:

- the core (``arachne_core.Stream.contributors``) reads them from the
  fragments it composites, in depth order, per sub-sample: its leaves carry
  their share of the pixel's sub-samples (``coverage``) and their colour, and a
  pixel whose sub-samples order two leaves both ways (two surfaces crossing
  inside it) is a ``crossing`` with each order;
- Cairo (:meth:`.cairo_painter.CairoDrawer.contributors`) draws each leaf alone
  over a transparent frame, with the camera's own code, and reads what it left
  in each pixel, in paint order.

Hidden leaves are reported too (``shown`` false): a leaf behind an opaque
front is evidence that something lies behind it, where a leaf the painter did
not draw is absent. A leaf's ``id`` is the stable id its stream gives its
object (``StreamState.objects``), which matches a leaf across two streams that
number their objects apart.

What Cairo is handed for each leaf is kept too (:class:`Outline`,
:meth:`.cairo_painter.CairoDrawer.outlines`): each fill and stroke the
camera makes, its path in pixels as Bézier segments, its width and its
colours. It is the geometry a difference at an edge is measured against.
"""

from __future__ import annotations

__all__ = [
    "Box",
    "Leaf",
    "LiveFrame",
    "Outline",
    "PixelContributors",
    "Subpath",
    "from_core",
    "over",
]

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

#: A pixel box, ``(x0, y0, x1, y1)``: the pixels ``x0 <= x < x1``,
#: ``y0 <= y < y1``.
Box = tuple[int, int, int, int]
RGBA = tuple[int, int, int, int]


class LiveFrame(RuntimeError):
    """Contributors asked of a painter of captures drawn as the scene runs,
    of a capture drawn over an earlier picture.

    Such a capture is drawn over the pixels of the pictures before it, so
    their leaves are flattened into its background and no report of it can
    name them: the frame is attributed from its state extracted whole,
    through both painters (``FrameStream.frame``). A capture its live
    painter records as whole (begun on its camera's background as the
    camera's reset left it, or continuing a picture that was) is reported
    as any frame is (``CorePainter.whole``).
    """


@dataclass(frozen=True)
class Leaf:
    """One leaf of a pixel.

    ``drawn`` is what the leaf alone draws over a transparent pixel, all its
    paint (a path's fill and strokes) composited, premultiplied RGBA: its
    alpha is the leaf's coverage of the pixel times its opacity. ``covers``
    is whether nothing beneath it reaches the pixel (what it leaves there
    does not depend on what was there). The core also knows coverage and
    colour apart: ``coverage`` is its share of the pixel's 16 sub-samples (0
    to 255) and ``colour`` its nearest fragment's premultiplied colour;
    Cairo's leaves carry ``None`` for both. ``painter`` is ``"core"`` or
    ``"cairo"``: on the core path, the overlay's leaves are Cairo's.
    """

    object: int
    part: int
    id: str | None
    drawn: RGBA
    shown: bool
    painter: str
    coverage: int | None = None
    colour: RGBA | None = None
    covers: bool = False

    @property
    def key(self) -> tuple[str | int, int]:
        """The leaf's identity across streams: its object's stable id (the
        object's number where the stream gave none) and its part.
        """
        return (self.object if self.id is None else self.id, self.part)


@dataclass(frozen=True)
class PixelContributors:
    """The leaves of one pixel, bottom first.

    ``leaves`` holds each leaf once. ``orders`` are the distinct sequences of
    leaves the pixel's sub-samples see, bottom first, each entry ``(index
    into leaves, coverage there)``, a leaf once per run of its fragments, so a
    leaf another lies between comes twice (Cairo: one sequence, each leaf's
    drawn alpha). ``crossing`` is whether the sequences order two leaves both
    ways, or admit no one order; the leaves are then in their depth order at
    the pixel's centre. ``decided`` is how the core ordered each pair of
    leaves, ``(a, b, rule)`` for indices ``a < b`` into ``leaves``: ``"paint"``
    (no sub-sample both cover: paint order), ``"tie"`` (one depth within the
    key arithmetic's error wherever both cover: paint order), ``"depth"``, or
    ``"majority"`` (their depths cross inside the pixel: the one in front at
    more of the sub-samples both cover). Cairo paints in paint order and
    reports none.
    """

    leaves: tuple[Leaf, ...]
    orders: tuple[tuple[tuple[int, int], ...], ...]
    crossing: bool = False
    decided: tuple[tuple[int, int, str], ...] = ()

    @property
    def keys(self) -> tuple[tuple[str | int, int], ...]:
        """The leaves' identities, bottom first."""
        return tuple(leaf.key for leaf in self.leaves)

    @property
    def shown(self) -> tuple[Leaf, ...]:
        """The leaves not hidden behind an opaque front."""
        return tuple(leaf for leaf in self.leaves if leaf.shown)


@dataclass(frozen=True)
class Subpath:
    """One subpath of an outline, in pixels (x right, y down): its start and
    its Bézier segments, each row ``(control 1, control 2, end)`` (``(k, 3,
    2)``; a straight segment's controls on its chord), and whether it was
    closed.
    """

    start: tuple[float, float]
    segments: np.ndarray
    closed: bool


@dataclass(frozen=True)
class Outline:
    """One paint operation of a leaf as Cairo is handed it: ``op`` is
    ``"fill"`` or ``"stroke"``; ``colours`` the source's colours in the
    frame's channel order, ``0`` to ``1`` and not premultiplied (one for a
    solid source, a gradient's stops otherwise); ``width`` a stroke's width in
    pixels (``0`` for a fill); ``join`` and ``cap`` Cairo's names for a
    stroke's (``""`` for a fill).
    """

    op: str
    colours: tuple[tuple[float, float, float, float], ...]
    width: float
    subpaths: tuple[Subpath, ...]
    join: str = ""
    cap: str = ""


def from_core(
    found: Mapping[tuple[int, int], tuple],
    objects: Mapping[int, tuple[str, str]],
) -> dict[tuple[int, int], PixelContributors]:
    """The core's report (``Stream.contributors``) as :class:`PixelContributors`,
    each leaf's stable id found in ``objects`` (``StreamState.objects``).
    """
    out: dict[tuple[int, int], PixelContributors] = {}
    for at, (leaves, orders, crossing, decided) in found.items():
        out[at] = PixelContributors(
            leaves=tuple(
                Leaf(
                    object=obj,
                    part=part,
                    id=objects.get(obj, (None,))[0],
                    drawn=tuple(drawn),
                    shown=shown,
                    painter="core",
                    coverage=coverage,
                    colour=tuple(colour),
                    covers=covers,
                )
                for obj, part, coverage, colour, drawn, shown, covers in leaves
            ),
            orders=tuple(tuple(tuple(e) for e in order) for order in orders),
            crossing=bool(crossing),
            decided=tuple(tuple(d) for d in decided),
        )
    return out


def over(
    under: Mapping[tuple[int, int], PixelContributors],
    top: Mapping[tuple[int, int], PixelContributors],
) -> dict[tuple[int, int], PixelContributors]:
    """``top``'s leaves composited over ``under``'s, pixel by pixel: the core
    path's frame, the core's leaves under Cairo's overlay. A leaf of
    ``under`` is hidden where a leaf of ``top`` covers the pixel; every
    sequence of ``under`` continues with ``top``'s, and ``top``'s leaves are
    over ``under``'s in paint order.
    """
    out = dict(under)
    for at, above in top.items():
        below = under.get(at)
        if below is None:
            out[at] = above
            continue
        covered = any(leaf.covers for leaf in above.leaves)
        shift = len(below.leaves)
        lifted = tuple(
            tuple((i + shift, c) for i, c in order) for order in above.orders
        )
        out[at] = PixelContributors(
            leaves=tuple(
                replace(leaf, shown=leaf.shown and not covered) for leaf in below.leaves
            )
            + above.leaves,
            orders=tuple(order + extra for order in below.orders for extra in lifted),
            crossing=below.crossing,
            decided=below.decided
            + tuple(
                (a, b + shift, "paint")
                for a in range(shift)
                for b in range(len(above.leaves))
            )
            + tuple((a + shift, b + shift, rule) for a, b, rule in above.decided),
        )
    return out
