"""A rig: an asset's parts as named vector objects in a scene, moved through
the parameters its maker tested.

``Rig(ref)`` loads the asset (:func:`~.reader.load_asset`) and holds one
group per part and per slot, in drawing order (the manifest's ``order``,
then its order of entries), each the part's shapes as ordinary
:class:`~arachne.mobject.types.vectorized_mobject.VMobject` objects: crisp
at any zoom, styled and transformed as any other. A slot's group holds one
group per drawing; the drawing it shows has its points, and every member of
the others holds none, so nothing draws them, while a style given to the
rig still reaches them.

Three things are kept apart, each with one writer:

- the parameter values and slot choices, immutable values replaced on
  write, written by :meth:`Rig.pose`, :meth:`Rig.show` and the animation
  :class:`~.motion.Pose`;
- where the rig is placed, ``F``, the affine map from the canvas to the
  scene: its **carrier**, a first child of three points with no paint, the
  images under ``F`` of three canvas points 10⁻⁴ scene units apart at the
  asset's size (:class:`~.pose.Model`). Whatever moves the rig as a whole
  moves it, and nothing that moves one part does;
- the art, every part's points, written by :meth:`Rig.settle` alone, from
  rest (:mod:`.pose`).

An animation pairs a rig with its copies member by member, by the members
that hold points. A copy shows what its rig shows: a copy aligned with the
rig for an animation takes its slot choices (:meth:`Rig.align_data`), and a
:class:`~.motion.Pose` that switches a slot switches it in the copies the
play's animations already hold. So a whole-rig animation and a slot switch
in one play pair each shape with its own copy, in either order.

A rig settles at once when it is posed or shown, from its own updater
(:func:`keep_settled`: after a play, and after the scene's code moved it),
and during a play that poses it from the scene's settle (:mod:`.motion`). A
part moved directly returns to its posed place at the next settle: parts
move through parameters. A rig scaled to nothing has no ``F`` and is not
settled until it has a size again.
"""

from __future__ import annotations

__all__ = ["CarrierPoint", "Rig", "keep_settled"]

import difflib
import math
import os
import weakref
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from ..mobject.types.vectorized_mobject import VectorizedPoint, VGroup, VMobject
from ..mobject.values import FrozenPoints, adopt
from ..utils.paths import straight_path
from .manifest import PartSpec, refuse
from .pose import CanvasMap, RigValues, model_of, posed_with, transforms
from .reader import Asset, load_asset

if TYPE_CHECKING:
    from typing import Self

    from ..mobject.mobject import Mobject
    from .anchor import Anchor

#: The points of a shape that draws nothing.
_EMPTY = adopt(np.zeros((0, 3)), FrozenPoints)


class CarrierPoint(VectorizedPoint):
    """One of a carrier's three points: moved with its rig, drawn by
    nothing, and no part of any box or measure of a family that holds it
    (its bounds, extremes and anchors are none), so a rig's box, its size
    and where it stands are its art's.
    """

    #: Passed over by the boxes the lint and the state read of a family.
    _scaffold = True

    def _point_bounds(self, anchors_every: int | None = None) -> Any:
        return None

    def reduce_across_dimension(self, reduce_func: Any, dim: int) -> float | None:
        return None

    def get_merged_array(self, array_attr: str) -> np.ndarray:
        if array_attr == "points":
            return np.zeros((0, self.dim))
        return super().get_merged_array(array_attr)

    def get_anchors(self) -> np.ndarray:
        return np.zeros((0, self.dim))

    def interpolate(
        self,
        mobject1: Mobject,
        mobject2: Mobject,
        alpha: float,
        path_func: Any = straight_path(),
    ) -> Self:
        super().interpolate(mobject1, mobject2, alpha, path_func)
        # A carrier's first point runs between its rig's copies: so do the
        # paints of the drawings its rig hides, which no animation pairs.
        rig, start, end = _rig_of(self), _rig_of(mobject1), _rig_of(mobject2)
        if rig is not None and start is not None and end is not None:
            rig._interpolate_hidden(start, end, alpha)
        return self


#: The rig of each rig's carrier's first point, made or copied, held weakly.
_CARRIED: weakref.WeakKeyDictionary[Mobject, weakref.ref[Rig]] = (
    weakref.WeakKeyDictionary()
)


def _rig_of(point: Mobject) -> Rig | None:
    held = _CARRIED.get(point)
    return None if held is None else held()


def _carry(rig: Rig) -> None:
    _CARRIED[rig.carrier.submobjects[0]] = weakref.ref(rig)


def keep_settled(rig: Mobject) -> None:
    """A rig's own updater: settle it (:meth:`Rig.settle`). It runs after a
    play's animations are done, and wherever the mobjects' updaters run; it
    takes no ``dt``, so it ticks no wait.
    """
    assert isinstance(rig, Rig)
    rig.settle()


class Rig(VGroup):
    """The asset ``asset`` (an :class:`~.reader.Asset`, or a reference
    :func:`~.reader.load_asset` takes) at its default pose, placed as its
    manifest's ``size`` places it. ``vmobject_kwargs`` go to the group, as a
    ``VGroup``'s do.
    """

    def __init__(
        self, asset: Asset | str | os.PathLike[str], **vmobject_kwargs: Any
    ) -> None:
        asset = load_asset(asset)
        model = model_of(asset)
        super().__init__(**vmobject_kwargs)
        self.asset = asset
        self.model = model
        manifest = asset.manifest
        self.carrier = VGroup(
            *(CarrierPoint(point) for point in model.natural.carrier(model)),
            name="carrier",
        )
        entries: list[tuple[int, int, str, VGroup]] = [
            (part.order, i, part.name, VGroup(*asset.rest(part.name), name=part.name))
            for i, part in enumerate(manifest.parts)
        ]
        entries += [
            (
                slot.order,
                len(manifest.parts) + i,
                slot.name,
                VGroup(
                    *(
                        VGroup(*asset.drawing(slot.name, d.name), name=d.name)
                        for d in slot.drawings
                    ),
                    name=slot.name,
                ),
            )
            for i, slot in enumerate(manifest.slots)
        ]
        members = [
            (name, group)
            for *_, name, group in sorted(entries, key=lambda entry: entry[:2])
        ]
        self._members = tuple(members)
        groups = [group for _, group in members]
        self.add(self.carrier, *groups)
        # What a settle needs whole: each group the rig made, with its
        # children as made.
        holders: list[VMobject] = [self.carrier, *groups]
        for name, group in members:
            if not isinstance(model.placed[name], PartSpec):
                holders += group.submobjects
        self._skeleton = tuple(
            (holder, tuple(holder.submobjects)) for holder in holders
        )
        self._values = _values(model.defaults)
        self._shown = ",".join(slot.default for slot in manifest.slots)
        self._settled_from: tuple[Any, ...] | None = None
        self.add_updater(keep_settled)
        _carry(self)
        self.settle()

    def __deepcopy__(self, clone_from_id: dict[int, Any]) -> Self:
        copied = super().__deepcopy__(clone_from_id)
        _carry(copied)
        return copied

    # Names

    def part(self, name: str) -> VMobject:
        """The group of the part or slot ``name``; ``rig.name_unknown`` for
        a name the asset's manifest does not have.
        """
        for found, group in self._members:
            if found == name:
                return group
        self._unknown("part", name, [found for found, _ in self._members])

    def _unknown(self, kind: str, name: str, names: list[str]) -> NoReturn:
        nearest = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
        hint = f", and the nearest is {' or '.join(nearest)}" if nearest else ""
        what = "part or slot" if kind == "part" else kind
        refuse(
            "rig.name_unknown",
            f"the asset {self.asset.name} has no {what} {name!r}: it has"
            f" {', '.join(names) or 'none'}{hint}",
            asset=self.asset.name,
            kind=kind,
            name=name,
            nearest=nearest,
        )

    # Values

    def value(self, parameter: str) -> float:
        """The value of ``parameter`` now."""
        return float(self._values[self._index(parameter)])

    def shown(self, slot: str) -> str:
        """The drawing the slot ``slot`` shows now."""
        return self._choices()[self._slot(slot)]

    def _index(self, parameter: str) -> int:
        index = self.model.index.get(parameter)
        if index is None:
            self._unknown("parameter", parameter, list(self.model.index))
        return index

    def _slot(self, slot: str) -> int:
        names = [s.name for s in self.asset.manifest.slots]
        if slot not in names:
            self._unknown("slot", slot, names)
        return names.index(slot)

    def _choices(self) -> list[str]:
        return self._shown.split(",") if self._shown else []

    def checked_values(
        self, values: dict[str, Any], *, eased: bool = False
    ) -> dict[int, float]:
        """``values``, parameter names to values, as values by parameter
        index, each in its tested range: ``rig.name_unknown`` for a name the
        manifest does not have, ``rig.range`` for a value outside the range
        (``eased``: one a pose's rate function carried there).
        """
        checked: dict[int, float] = {}
        for name, given in values.items():
            index = self._index(name)
            spec = self.model.parameters[index]
            value = float(given)
            if not (math.isfinite(value) and spec.min <= value <= spec.max):
                how = (
                    f"carried to {value:g} by its pose's rate function"
                    if eased
                    else f"given {given!r}"
                )
                refuse(
                    "rig.range",
                    f"the parameter {name} of the asset {self.asset.name} is {how},"
                    f" outside its tested range from {spec.min:g} to {spec.max:g}",
                    asset=self.asset.name,
                    parameter=name,
                    value=value if math.isfinite(value) else repr(value),
                    min=spec.min,
                    max=spec.max,
                )
            checked[index] = value
        return checked

    def checked_drawings(self, drawings: dict[str, Any]) -> dict[int, str]:
        """``drawings``, slot names to drawing names, by slot index:
        ``rig.name_unknown`` for a slot or a drawing the manifest does not
        have.
        """
        checked: dict[int, str] = {}
        for slot, drawing in drawings.items():
            index = self._slot(slot)
            names = [d.name for d in self.asset.manifest.slots[index].drawings]
            if drawing not in names:
                self._unknown("drawing", str(drawing), names)
            checked[index] = drawing
        return checked

    def pose(self, /, **values: float) -> Self:
        """Set the parameters ``values`` (names to values) and settle."""
        self.write_values(self.checked_values(values))
        return self.settle()

    def show(self, /, **drawings: str) -> Self:  # type: ignore[override]
        """Have each slot named in ``drawings`` show the drawing given, and
        settle.
        """
        self.write_choices(self.checked_drawings(drawings))
        return self.settle()

    def write_values(self, values: dict[int, float]) -> None:
        """Replace the values at the indices of ``values`` (checked, as
        :meth:`checked_values` gives them); nothing is settled.
        """
        if values:
            updated = np.array(self._values, dtype=float)
            for index, value in values.items():
                updated[index] = value
            self._values = _values(updated)

    def write_choices(self, drawings: dict[int, str]) -> None:
        """Replace the choices of the slots at the indices of ``drawings``
        (checked, as :meth:`checked_drawings` gives them); nothing is
        settled.
        """
        if drawings:
            choices = self._choices()
            for index, drawing in drawings.items():
                choices[index] = drawing
            self._shown = ",".join(choices)

    # Anchors

    def anchor(self, name: str) -> Anchor:
        """The anchor ``name`` of the asset: the point of its part it names,
        followed wherever the part goes (:class:`~.anchor.Anchor`). A name
        the manifest does not have is found by the lint
        (``annotation.anchor_unknown``) and refuses drawing; it gives an
        anchor with no point.
        """
        from .anchor import Anchor, attachments

        attachment = attachments(self.model).get(name)
        if attachment is None:
            from ..lint.raised import found

            names = list(self.asset.anchors)
            nearest = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
            hint = f", and the nearest is {' or '.join(nearest)}" if nearest else ""
            found(
                "annotation.anchor_unknown",
                f"the asset {self.asset.name} has no anchor {name!r}: it has"
                f" {', '.join(names) or 'none'}{hint}",
                detail={"asset": self.asset.name, "name": name, "nearest": nearest},
                objects=(self,),
            )
        return Anchor(self, attachment, name)

    # Placement and art

    def placement(self) -> CanvasMap | None:
        """``F``, where the rig is placed, read from its carrier; ``None``
        for a rig scaled to nothing.
        """
        return CanvasMap.from_carrier(
            self.model, [point.points[0] for point in self.carrier.submobjects]
        )

    def place(self, placement: CanvasMap) -> Self:
        """Put the rig where ``placement`` places its canvas, and settle."""
        for point, at in zip(
            self.carrier.submobjects, placement.carrier(self.model), strict=True
        ):
            point.points = at[None]
        return self.settle()

    def settle(self) -> Self:
        """Compute the art from rest, the values, the slots' choices and the
        carrier (:mod:`.pose`). Nothing is done when those, and the art,
        are what the art was last settled from, or when the rig has no
        placement. A rig changed into something its asset does not draw (a
        transform into another shape) raises :class:`ValueError`.
        """
        self._whole()
        carrier = tuple(point.points for point in self.carrier.submobjects)
        shapes = self._shapes()
        last = self._settled_from
        if (
            last is not None
            and last[0] is self._values
            and last[1] == self._shown
            and _same(last[2], carrier)
            and _same(last[3], tuple(shape.points for shape in shapes))
        ):
            return self
        placement = self.placement()
        if placement is None:
            return self
        model, values = self.model, np.asarray(self._values)
        affines = transforms(model, values)
        choices = dict(
            zip(
                (slot.name for slot in self.asset.manifest.slots),
                self._choices(),
                strict=True,
            )
        )
        for name, group in self._members:
            if isinstance(model.placed[name], PartSpec):
                posed = posed_with(model, name, values, affines[name])
                _write(group.submobjects, posed, placement)
                continue
            for drawing in group.submobjects:
                if drawing.name == choices[name]:
                    posed = posed_with(model, name, values, affines[name], drawing.name)
                    _write(drawing.submobjects, posed, placement)
                else:
                    for shape in drawing.submobjects:
                        shape.points = _EMPTY
        self._settled_from = (
            self._values,
            self._shown,
            carrier,
            tuple(shape.points for shape in shapes),
        )
        return self

    # Pairing with copies

    def align_data(self, mobject: Mobject, skip_point_alignment: bool = False) -> Self:
        """Align with ``mobject`` for an animation between them, as any
        group is. A copy of the rig (a rig of its asset, both whole) first
        takes the rig's slot choices: which drawing a slot shows is the
        rig's to say, through :meth:`show` and :class:`~.motion.Pose`, never
        an animation's, so each shape is paired with its own copy and keeps
        its own paint.
        """
        if (
            isinstance(mobject, Rig)
            and mobject.model is self.model
            and mobject._shown != self._shown
            and self._intact()
            and mobject._intact()
        ):
            mobject.adopt_choices(self)
        return super().align_data(mobject, skip_point_alignment)

    def _interpolate_hidden(self, start: Rig, end: Rig, alpha: float) -> None:
        """Give each shape this rig hides the paint between its own in the
        copies ``start`` and ``end``, which hide it too, at ``alpha``: an
        animation pairs only shapes with points, so a drawing switched in
        later shows the paint the animation left on the rig.
        """
        if not (
            start.model is self.model
            and end.model is self.model
            and self._intact()
            and start._intact()
            and end._intact()
        ):
            return
        for shape, first, last in zip(
            self._shapes(), start._shapes(), end._shapes(), strict=True
        ):
            if not (
                shape.get_num_points()
                or first.get_num_points()
                or last.get_num_points()
            ):
                shape.interpolate_color(first, last, alpha)

    def adopt_choices(self, rig: Rig) -> None:
        """Show what ``rig``, a rig of the same asset, shows, and settle."""
        self._shown = rig._shown
        self.settle()

    def _shapes(self) -> list[VMobject]:
        """Every shape of the art, the parts' and every drawing's."""
        found: list[VMobject] = []
        for name, group in self._members:
            if isinstance(self.model.placed[name], PartSpec):
                found += group.submobjects
            else:
                for drawing in group.submobjects:
                    found += drawing.submobjects
        return found

    def _intact(self) -> bool:
        """Whether the rig holds its carrier and parts as it made them."""
        return (
            self.carrier in self.submobjects
            and all(group in self.submobjects for _, group in self._members)
            and all(
                _same(tuple(holder.submobjects), children)
                for holder, children in self._skeleton
            )
            and all(len(point.points) == 1 for point in self.carrier.submobjects)
        )

    def _whole(self) -> None:
        """Raise unless the rig holds its carrier and parts as it made them."""
        if not self._intact():
            raise ValueError(
                f"A rig of the asset {self.asset.name} was changed into something its"
                " asset does not draw (its carrier or its parts are not as the rig"
                " made them, as a transform into another shape leaves them), so it"
                " can't be posed or settled. Transform a copy, or morph two rigs"
                " part by part with MorphParts."
            )


def _values(values: np.ndarray) -> RigValues:
    return adopt(np.array(values, dtype=float), RigValues)


def _same(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


def _write(
    shapes: list[VMobject], posed: tuple[np.ndarray, ...], placement: CanvasMap
) -> None:
    for shape, points in zip(shapes, posed, strict=True):
        shape.points = adopt(placement(points), FrozenPoints)
